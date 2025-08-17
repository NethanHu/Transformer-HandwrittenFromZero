import math
from typing import Tuple

import torch
import torch.nn.functional as F
import torch.nn as nn

# Define hyperparameters of the Transformer
batch_size: int = 4
d_model: int = 512  # d_model can define the width of embedding, i.e. how much info does a word have
context_len: int = 16  # constrain the word length of one input
num_heads: int = 8
num_blocks: int = 12  # num of transformer blocks
head_size: int = d_model // num_heads  # divide the embedding to each heads to focus on different aspects
scale: float = math.sqrt(head_size)
device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dropout: float = 0.1


# The meaning of feed forward network is processing the output from multi-attentions
class FeedForwardNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.ffn: nn.Module = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ffn(x)

class Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.Wq: nn.Module = nn.Linear(d_model, head_size, bias=False)
        self.Wk: nn.Module = nn.Linear(d_model, head_size, bias=False)
        self.Wv: nn.Module = nn.Linear(d_model, head_size, bias=False)
        # torch.tril will return the lower triangle of a matrix
        self.register_buffer('mask', torch.tril(torch.ones(context_len, context_len)))
        self.Dropout: nn.Module = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, T, _ = x.shape  # Dim x: [b, [vary by timestep], d_model]
        q: torch.Tensor = self.Wq(x)  # Dim q: [b, context_len, head_size]
        k: torch.Tensor = self.Wk(x)  # Dim k: [b, context_len, head_size]
        v: torch.Tensor = self.Wv(x)  # Dim v: [b, context_len, head_size]
        out: torch.Tensor = torch.matmul(q, k.transpose(-2, -1))  # q @ k.T
        out = torch.div(out, scale)   # (q @ k.T) / sqrt(head_size) <- square matrix [con_len * con_len]
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
    def __init__(self):
        super().__init__()
        self.heads: nn.ModuleList = nn.ModuleList([Attention() for _ in range(num_heads)])
        self.Wo: nn.Module = nn.Linear(d_model, d_model)
        self.Dropout: nn.Module = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out: torch.Tensor = torch.cat([head(x) for head in self.heads], dim=1)
        out = self.Wo(out)
        out = self.Dropout(out)
        return out


class TransformerBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1: nn.Module = nn.LayerNorm(d_model)
        self.ln2: nn.Module = nn.LayerNorm(d_model)
        self.mha: nn.Module = MultiHeadAttention()
        self.ffn: nn.Module = FeedForwardNN()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out: torch.Tensor = self.ln1(x)
        out = x + self.mha(out)
        out = out + self.ln2(self.ffn(out))
        return out


class Transformer(nn.Module):
    def __init__(self, max_token_val: int=100256):
        super().__init__()
        self.vocab_linear: nn.Module = nn.Linear(d_model, max_token_val)
        self.token_embed_tab: nn.Embedding = nn.Embedding(max_token_val, d_model)
        self.transformer_blocks: nn.Module = nn.Sequential(
            # after we get num_blocks of Transformer block, we get  the LayerNorm
            *([TransformerBlock() for _ in range(num_blocks)] +
              [nn.LayerNorm(d_model)])
        )

    def forward(self, x_batch: torch.Tensor, y_batch: torch.Tensor=None) -> Tuple[torch.Tensor, torch.Tensor | None]:
        _, T, _ = x_batch.shape
        pe_lookup_tab: torch.Tensor = torch.zeros(context_len, d_model, device=device)
        position: torch.Tensor = torch.arange(0, context_len, dtype=torch.float, device=device).unsqueeze(1)
        div_term: torch.Tensor = torch.exp(-math.log(10000.0) *
                                           torch.arange(0, d_model, 2).float() / d_model)
        pe_lookup_tab[:, 0::2] = torch.sin(position * div_term)
        pe_lookup_tab[:, 1::2] = torch.cos(position * div_term)
        out: torch.Tensor = self.token_embed_tab(x_batch) + pe_lookup_tab
        out = self.transformer_blocks(out)
        logits: torch.Tensor = self.vocab_linear(out)
        loss: None | torch.Tensor = None

        if y_batch is not None:  # <- training process
            B, T, D = logits.shape
            logits_2d: torch.Tensor = logits.view(B * T, D)
            y_batch_2d: torch.Tensor = logits.view(B * T)
            loss: torch.Tensor = F.cross_entropy(logits_2d, y_batch_2d)

        return logits, loss
