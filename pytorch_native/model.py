import math
from typing import Tuple

import torch
import torch.nn.functional as F
import torch.nn as nn


# The meaning of feed forward network is processing the output from multi-attentions
class FeedForwardNN(nn.Module):
    def __init__(self, d_model: int, dropout: float):
        super().__init__()
        self.d_model: int = d_model
        self.dropout: float = dropout
        self.ffn: nn.Module = nn.Sequential(
            nn.Linear(self.d_model, self.d_model * 4),
            nn.ReLU(),
            nn.Linear(self.d_model * 4, self.d_model),
            nn.Dropout(self.dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ffn(x)

class Attention(nn.Module):
    def __init__(self, d_model: int, head_size: int, context_len: int, dropout: float):
        super().__init__()
        self.d_model: int = d_model
        self.head_size: int = head_size
        self.context_len: int = context_len
        self.dropout: float = dropout
        self.scale: float = math.sqrt(self.head_size)
        self.Wq: nn.Module = nn.Linear(self.d_model, self.head_size, bias=False)
        self.Wk: nn.Module = nn.Linear(self.d_model, self.head_size, bias=False)
        self.Wv: nn.Module = nn.Linear(self.d_model, self.head_size, bias=False)
        # torch.tril will return the lower triangle of a matrix
        self.register_buffer('mask', torch.tril(torch.ones(self.context_len, self.context_len)))
        self.Dropout: nn.Module = nn.Dropout(self.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape  # Dim x: [batch_size, context_len, d_model]
        q: torch.Tensor = self.Wq(x)  # Dim q: [batch_size, context_len, head_size]
        k: torch.Tensor = self.Wk(x)  # Dim k: [batch_size, context_len, head_size]
        v: torch.Tensor = self.Wv(x)  # Dim v: [batch_size, context_len, head_size]
        out: torch.Tensor = torch.matmul(q, k.transpose(-2, -1))  # q @ k.T
        out = torch.div(out, self.scale)   # (q @ k.T) / sqrt(head_size) <- square matrix [con_len * con_len]
        out = out.masked_fill(self.mask[:T, :T] == 0, float('-inf'))
        #          (word Key) every word should only see the previous ones and itself
        #          t1    t2    t3     t4
        # (word Query)
        #    t1  [1.2, -inf,  -inf,  -inf] -> softmax [1.00,  0.00,  0.00,  0.00]
        #    t2  [0.9,  1.8,  -inf,  -inf] -> softmax [0.29,  0.71,  0.00,  0.00]
        #    t3  [0.4,  1.3,   2.1,  -inf] -> softmax [0.10,  0.27,  0.63,  0.00]
        #    t4  [1.1,  0.5,   1.9,   1.4] -> softmax [0.20,  0.10,  0.40,  0.30]
        out = F.softmax(out, dim=-1)
        out = self.Dropout(out)
        out = torch.matmul(out, v)
        return out


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, head_size: int, context_len: int, dropout: float):
        super().__init__()
        self.d_model: int = d_model
        self.num_heads: int = num_heads
        self.head_size: int = head_size
        self.context_len: int = context_len
        self.dropout: float = dropout
        self.heads: nn.ModuleList = nn.ModuleList(
            [Attention(self.d_model, self.head_size, self.context_len, self.dropout) for _ in range(self.num_heads)]
        )
        self.Wo: nn.Module = nn.Linear(self.d_model, self.d_model)
        self.Dropout: nn.Module = nn.Dropout(self.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out: torch.Tensor = torch.cat([head(x) for head in self.heads], dim=-1)
        out = self.Wo(out)
        out = self.Dropout(out)
        return out


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, head_size: int, context_len: int, dropout: float):
        super().__init__()
        self.d_model: int = d_model
        self.num_heads: int = num_heads
        self.head_size: int = head_size
        self.context_len: int = context_len
        self.dropout: float = dropout
        self.ln1: nn.Module = nn.LayerNorm(self.d_model)
        self.ln2: nn.Module = nn.LayerNorm(self.d_model)
        self.mha: nn.Module = MultiHeadAttention(self.d_model, self.num_heads, self.head_size, self.context_len, self.dropout)
        self.ffn: nn.Module = FeedForwardNN(self.d_model, self.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out: torch.Tensor = x + self.mha(self.ln1(x))
        out = out + self.ffn(self.ln2(out))
        return out


class Transformer(nn.Module):
    def __init__(self, params: dict):
        super().__init__()
        self.context_len: int = params['context_len']
        self.d_model: int = params['d_model']
        self.num_blocks: int = params['num_blocks']
        self.num_heads: int = params['num_heads']
        self.head_size: int = self.d_model // self.num_heads
        self.dropout: float = params['dropout']
        self.device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.max_token_value: int = params['max_token_value']
        self.vocab_linear: nn.Module = nn.Linear(self.d_model, self.max_token_value)
        self.token_embed_tab: nn.Embedding = nn.Embedding(self.max_token_value, self.d_model)
        self.transformer_blocks: nn.Module = nn.Sequential(
            # after we get num_blocks of Transformer block, we get  the LayerNorm
            *([TransformerBlock(self.d_model, self.num_heads, self.head_size, self.context_len, self.dropout) for _ in range(self.num_blocks)] +
              [nn.LayerNorm(self.d_model)])
        )

    def forward(self, x_batch: torch.Tensor, y_batch: torch.Tensor=None) -> Tuple[torch.Tensor, torch.Tensor | None]:
        # token indices: [batch_size, context_len]
        B, T = x_batch.shape

        pe_lookup_tab: torch.Tensor = torch.zeros(self.context_len, self.d_model, device=self.device)
        position: torch.Tensor = torch.arange(0, self.context_len, dtype=torch.float, device=self.device).unsqueeze(1)
        div_term: torch.Tensor = torch.exp(-math.log(10000.0) *
                                           torch.arange(0, self.d_model, 2).float().to(self.device) / self.d_model)
        pe_lookup_tab[:, 0::2] = torch.sin(position * div_term)
        pe_lookup_tab[:, 1::2] = torch.cos(position * div_term)

        token_embeddings = self.token_embed_tab(x_batch)  # [B, T, d_model]
        out: torch.Tensor = token_embeddings + pe_lookup_tab[:T, :]

        out = self.transformer_blocks(out)
        logits: torch.Tensor = self.vocab_linear(out)
        loss: None | torch.Tensor = None

        if y_batch is not None:  # <- training process
            B, T, D = logits.shape
            logits_2d: torch.Tensor = logits.view(B * T, D)
            y_batch_2d: torch.Tensor = y_batch.view(B * T)
            loss: torch.Tensor = F.cross_entropy(logits_2d, y_batch_2d)

        return logits, loss

    # x_batch: [batch_size=1, context_len, d_model]
    def generate(self, x_batch: torch.Tensor, max_new_tokens=100, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            x_crop: torch.Tensor = x_batch[:, -self.context_len:]  # cropped substr len of context_len
            logits, _ = self.forward(x_crop)  # Dim [batch_size, [Timestep], vocab_dim]
            logits = logits[:, -1, :]  / temperature  # we get the last predicted token
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('inf')
            prob: torch.Tensor = torch.softmax(logits, dim=-1)
            pred_token: torch.Tensor = torch.multinomial(prob, num_samples=1)  # token index
            x_batch = torch.cat((x_batch, pred_token), dim=1)

        return x_batch