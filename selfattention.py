import os
import time
import torch
import requests
import argparse
import numpy as np
import torch.nn as nn
from torch.nn import functional as F

parser = argparse.ArgumentParser(description="file contains raw implementation of a transformer")
parser.add_argument('--resume', help="Resume training or start new", default=False, type=bool)
args = parser.parse_args()


# for calculating script time
t0 = time.time()

# HyperParameters
learning_rate=3e-4
train_iters=1000
grad_accum_steps=25
n_embd = 384
n_heads = 6
n_layer =  6
block_size = 96
batch_size = 32
split = 0.9
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 20
eval_interval = 5
dropout=0.2
out_dir = './models'
model_args = dict(n_layer=n_layer, n_head=n_heads, n_embd=n_embd, block_size=block_size,
                 vocab_size=None, dropout=dropout)

# Reading data
if not os.path.exists('input.txt'):
    with open('input.txt',mode='w') as file:
        data_url_path = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
        text = requests.get(data_url_path).text
        file.write(text)
else:
    with open('input.txt', mode='r') as file:
        text = file.read()

# Encoder and Decoder
vocab = ''.join(sorted(list(set(text)))) # Contains all the distinct chars from the file
vocab_size = len(vocab)
vocab_to_id = {char: i for i,char in enumerate(vocab)}
id_to_vocab = {i: char for char, i in vocab_to_id.items()}
encode = lambda s : [vocab_to_id[c] for c in s]
decode = lambda s : ''.join([id_to_vocab[c] for c in s])

# Split Train and Validation
data = torch.tensor([vocab_to_id[char] for char in text])
N = len(data)
n = int(N * split)
train_data = data[:n]
val_data = data[n:]


# Estimate losses
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros((eval_iters, ))
        for i in range(eval_iters):
            x, y = get_batch(split)
            x, y = x.to(device=device), y.to(device=device)
            logits, loss = model(x, y)
            losses[i] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# Data Loader
def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix =  torch.randint(len(data) - block_size, (batch_size, ))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + 1 + block_size] for i in ix])
    x, y = x.to(device=device), y.to(device=device)
    return x, y

# Feed Forward simple ANN
class Feedforward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

# Single Attention Block
class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.head_size = head_size
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer('tril', torch.tril(torch.ones((block_size, block_size))))
    def forward(self, x):
        
        B, T, C = x.shape
        k = self.key(x) # (B, T, head_size)
        q = self.query(x) # (B, T, head_size)
        # Interaction of keys with queries.
        wei = q @ k.transpose(-2, -1) * self.head_size**-0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        wei = F.softmax(wei ,dim=-1)
        wei = self.dropout(wei)
        # Aggregating the results
        v = self.value(x)
        out = wei @ v
        return out # (B, T, head_size)

# Multiheaded Attention block
class MultiHeadedAttention(nn.Module):
    def __init__(self, n_heads, head_size):
        super().__init__()
        self.n_heads = n_heads
        self.heads = nn.ModuleList([Head(head_size) for _ in range(n_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        x = torch.cat([h(x) for h in self.heads], dim=-1)
        x = self.dropout(self.proj(x))
        return x

# Single Transformer Block
class Block(nn.Module):
    def __init__(self, n_heads, n_embd):
        super().__init__()
        head_size = n_embd // n_heads
        self.sa_heads = MultiHeadedAttention(n_heads=n_heads, head_size=head_size)
        self.ffwd = Feedforward(n_embd=n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        x = x + self.sa_heads(self.ln1(x)) # (B, T, C)
        x = x + self.ffwd(self.ln2(x)) # (B, T, C)
        return x

# Bigram Model
class LanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, n_embd) 
        self.position_embedding = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd=n_embd, n_heads=n_heads) for _ in range(n_layer)])
        self.lm_head = nn.Linear(n_embd, vocab_size)
    def forward(self, idx, targets=None):
        B, T = idx.shape
        # idx is (B,T)
        tok_embd = self.token_embedding(idx) # (B, T, C)
        pos_embd = self.position_embedding(torch.arange(T, device=device)) # (T, C)
        x = tok_embd + pos_embd # (B, T, C)
        x = self.blocks(x)
        #x = self.sa_heads(x)
        #x = self.ffwd(x) # Adding a linear head just before we compute logits to give it time to think
        logits = self.lm_head(x) # It is (B,T,C)

        if targets is None:
            loss = None
        else:
            # for computing the cross entropy loss we are supposed to change it to (B*T,C)
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss
    def generate(self, idx, max_new_tokens):
        # Again idx is (B, T)
        for _ in range(max_new_tokens):
            idx_cond = idx[: , -block_size: ]
            logits, loss = self.forward(idx_cond) # This is (B, T, C)
            logits = logits[:,-1,:] # Just taking the last chars preds (B, C)
            probs = F.softmax(logits, dim=-1)
            preds = torch.multinomial(probs, num_samples=1) # (B,1)
            idx = torch.cat((idx, preds), dim=-1)
        return idx

# Instantiating and moving the model to device.
model = LanguageModel()
# creating optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
# best_val_loss so far
best_val_loss = 4.00

if args.resume:
    print("Resuming Training!!")

    checkpoint = torch.load(os.path.join(out_dir, 'ckpt.pt'), map_location=device)
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
        # Move optimizer's state to the same device
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device)
    best_val_loss = checkpoint['best_val_loss']

model = model.to(device=device)
# training loop

for iter in range(train_iters):

    # every eval_interval try to evaluate the loss
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
    if losses['val'] < best_val_loss:
        best_val_loss = losses['val']
        if iter > 0:
            checkpoint = {
                'model' : model.state_dict(),
                'optimizer' : optimizer.state_dict(),
                'model_args' : model_args,
                'iter_num' : iter,
                'best_val_loss': best_val_loss,
            }
            print(f"saving model checkpoint to {out_dir}")
            torch.save(checkpoint, os.path.join(out_dir, 'ckpt.pt'))

    optimizer.zero_grad(set_to_none=True)
    lossi = 0.0
    for microstep in range(grad_accum_steps):
        x, y = get_batch('train')
        logits, loss = model(x, y)
        loss.backward() # accumulating the gradients
        lossi += loss.item() / grad_accum_steps
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

context = torch.zeros((1, 1), dtype=torch.int32, device=device)
print(decode(model.generate(context, max_new_tokens=500)[0].tolist()))

t1 = time.time()

print(f"Total time taken for script to run on {device}: {t1 - t0}")
