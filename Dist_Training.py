import torch
import torch.nn as nn
from torch.nn import functional as F
import mmap
import random
import pickle
import argparse
import os
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel
import torch.distributed as dist
import torch.multiprocessing as mp

parser = argparse.ArgumentParser(description='This is a demonstration program')
parser.add_argument('-batch_size', type=int, default=8, help='Please provide a batch_size')

args = parser.parse_args()
print(f'batch size: {args.batch_size}')

device = 'cuda' if torch.cuda.is_available() else 'cpu'

batch_size = int(args.batch_size)
block_size = 128
max_iters = 200
learning_rate = 3e-4
eval_iters = 200
n_embd = 2000
n_head = 8
n_layer = 50
dropout = 0.2
gradient_accumulation_steps = 4

print(device)

chars = ""
with open("/data/llm_from_scratch/vocab.txt", 'r', encoding='utf-8') as f:
    text = f.read()
    chars = sorted(list(set(text)))

vocab_size = len(chars)

string_to_int = {ch: i for i, ch in enumerate(chars)}
int_to_string = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [string_to_int[c] for c in s]
decode = lambda l: ''.join([int_to_string[i] for i in l])

def get_random_chunk(split):
    filename = "/data/llm_from_scratch/output_train.txt" if split == 'train' else "/data/llm_from_scratch/output_val.txt"
    with open(filename, 'rb') as f:
        with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
            file_size = len(mm)
            start_pos = random.randint(0, (file_size) - block_size * batch_size)

            mm.seek(start_pos)
            block = mm.read(block_size * batch_size - 1)

            decoded_block = block.decode('utf-8', errors='ignore').replace('\r', '')
            data = torch.tensor(encode(decoded_block), dtype=torch.long)

    return data

def get_batch(split):
    data = get_random_chunk(split)
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

def estimate_loss(model):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = []
        for k in range(eval_iters):
            X, Y = get_batch(split)
            with torch.cuda.amp.autocast():
                logits, loss = model(X, Y)
            losses.append(loss.unsqueeze(0))  # Unsqueeze to avoid scalar gather warning
        losses = torch.cat(losses, dim=0)
        out[split] = losses.mean().item()
    model.train()
    return out

class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        out = wei @ v
        return out

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedFoward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        y = self.sa(x)
        x = self.ln1(x + y)
        y = self.ffwd(x)
        x = self.ln2(x + y)
        return x

class GPTLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, index, targets=None):
        B, T = index.shape
        tok_emb = self.token_embedding_table(index)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, index, max_new_tokens):
        for _ in range(max_new_tokens):
            logits, loss = self.forward(index)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            index_next = torch.multinomial(probs, num_samples=1)
            index = torch.cat((index, index_next), dim=1)
        return index

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# Wrap your model with DataParallel
# model = nn.DataParallel(model)
# model = model.to(device)
# model = model.half()  # Convert the model to half precision (FP16)


def main(local_rank, world_size, args):
    # Set the device based on the local rank
    print(f'Local Rank: {local_rank}')
    device = torch.device('cuda', local_rank)
    torch.cuda.set_device(device)

    # Initialize DDP
    torch.distributed.init_process_group(backend='nccl', init_method='env://')
    
    model = GPTLanguageModel(vocab_size).to(device)

    model = DistributedDataParallel(model, device_ids=[local_rank])
    print('loading model parameters...')
    print(f'Number of parameters in the model: {count_parameters(model)}')
    
    # create a PyTorch optimizer after model creation
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scaler = torch.cuda.amp.GradScaler()

    # Your training loop
    for iter in range(args.max_iters):
        print(iter)
    if iter % args.eval_iters == 0:
        losses = estimate_loss(model)  # Pass the model as an argument
        print(f"step: {iter}, train loss: {losses['train']:.3f}, val loss: {losses['val']:.3f}")

    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = model.forward(xb, yb)
    loss.backward()
    if iter % args.gradient_accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    with open('model-01.pkl', 'wb') as f:
        # Save the model state_dict
        torch.save(model.state_dict(), f)

        print('model saved')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='This is a demonstration program')

    # Add your command-line arguments here
    parser.add_argument('-batch_size', type=int, required=True, help='Please provide a batch_size')
    parser.add_argument('-max_iters', type=int, default=200, help='Maximum number of iterations')
    parser.add_argument('-eval_iters', type=int, default=200, help='Number of iterations between evaluations')
    parser.add_argument('-gradient_accumulation_steps', type=int, default=1, help='Number of steps to accumulate gradients')

    args = parser.parse_args()

    # Set the world size based on the number of available GPUs
    world_size = torch.cuda.device_count()

    # Use torch.distributed.elastic.multiprocessing.spawn to spawn processes
    mp.spawn(main, args=(world_size, args), nprocs=world_size)
