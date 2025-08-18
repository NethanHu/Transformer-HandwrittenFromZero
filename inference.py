import tiktoken
import torch

from model import Transformer

params = {
    'context_len': 256,  # constrain the word length of one input
    'batch_size': 32,
    'd_model': 512,  # embedding size
    'num_blocks': 16,
    'num_heads': 8,
    'dropout': 0.1,
    'max_token_value': 100256,
    'device': torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    'seed': 1024,
    'saved_path': 'ckpt/model.pt'
}

SEED: int = 1024
torch.manual_seed(SEED)

checkpoint = torch.load(params['saved_path'], map_location=params['device'])
model: Transformer = Transformer(params)
model.load_state_dict(checkpoint)
model.eval()
model.to(params['device'])

tokenizer = tiktoken.get_encoding('cl100k_base')

start: str = 'A little princess cried when'
start_ids = tokenizer.encode(start)
x: torch.Tensor = torch.tensor(start_ids, dtype=torch.long, device=params['device']).unsqueeze(0)

with torch.no_grad():
    y: torch.Tensor = model.generate(x, max_new_tokens=30, temperature=1.0)
    print(tokenizer.decode(y[0].tolist()))