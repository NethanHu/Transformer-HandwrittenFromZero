from typing import Tuple

from model import Transformer

import tiktoken
from tiktoken.core import Encoding
import torch

params = {
    'context_len': 256,  # constrain the word length of one input
    'batch_size': 32,
    'd_model': 512,  # embedding size
    'learning_rate': 1e-3,
    'num_blocks': 16,
    'num_heads': 8,
    'dropout': 0.1,
    'max_token_value': 100256,
    'epochs': 1,
    'device': torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    'eval_interval': 50,
    'eval_iters': 20,
    'max_iters': 1000,
    'seed': 1024,
    'saved_path': './ckpt/model.pt',
    'dataset_path': './data/tinystories.csv'
}

assert params['d_model'] % params['num_heads'] == 0, 'd_model must be divisible by num_heads!'

torch.manual_seed(params['seed'])

with open(params['dataset_path'], 'r') as f:
    text: str = f.read()

tokenizer: Encoding = tiktoken.get_encoding('cl100k_base')
tokenizer_text: torch.Tensor = torch.tensor(tokenizer.encode(text), dtype=torch.long, device=params['device'])

train_size: int = int(len(tokenizer_text) * 0.9)
train_data: torch.Tensor = tokenizer_text[0: train_size]
valid_data: torch.Tensor = tokenizer_text[train_size:]

def get_batch(split: str) -> Tuple[torch.Tensor, torch.Tensor]:
    data: torch.Tensor = train_data if split == 'train' else valid_data
    indices: torch.Tensor = torch.randint(low=0, high=len(data) - params['context_len'], size=(params['batch_size'],))
    x: torch.Tensor = torch.stack([data[idx: idx + params['context_len']] for idx in indices])
    y: torch.Tensor = torch.stack([data[idx + 1: idx + params['context_len'] + 1] for idx in indices])
    return x, y

def estimate_loss(cur_model: Transformer) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    cur_model.eval()

    for split in ['train', 'valid']:
        losses: torch.Tensor = torch.zeros(params['eval_iters'], dtype=torch.float)
        for k in range(params['eval_iters']):
            x, y = get_batch(split)
            with torch.no_grad():
                _, loss = cur_model(x, y)
                losses[k] = loss.item()
        out[split] = losses.mean()

    cur_model.train()
    return out


model: torch.nn.Module = Transformer(params).to(params['device'])
optimizer: torch.optim.AdamW = torch.optim.AdamW(model.parameters(), lr=params['learning_rate'])

for e in range(params['epochs']):
    for step in range(params['max_iters']):
        if step % params['eval_interval'] == 0 or step == params['max_iters'] - 1:
            losses: dict[str, torch.Tensor] = estimate_loss(model)
            print('Step:', step, ', training Loss:', round(losses['train'].item(), 3),
                  ', validation Loss:', round(losses['valid'].item(), 3))

        x, y = get_batch('train')
        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

# you need to `mkdir ckpt` first
torch.save(model.state_dict(), params['saved_path'])