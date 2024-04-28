import os
import time
import torch
import requests
import numpy as np
import torch.nn as nn
from torch.nn import functional as F


# for calculating script time
t0 = time.time()

# HyperParameters
learning_rate=3e-4
train_iters=20000
block_size = 8
batch_size = 32
split = 0.9
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
eval_interval = 100

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

# Bigram Model
class BigramModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, vocab_size)
    
    def forward(self, idx, targets=None):

        # idx is (B,T)
        logits = self.embedding(idx) # It is (B,T,C)

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
            logits, loss = self.forward(idx) # This is (B, T, C)
            logits = logits[:,-1,:] # Just taking the last chars preds (B, C)
            probs = F.softmax(logits, dim=-1)
            preds = torch.multinomial(probs, num_samples=1) # (B,1)
            idx = torch.cat((idx, preds), dim=-1)
        return idx

# Instantiating and moving the model to device.
model = BigramModel(vocab_size)
model = model.to(device=device)

# creating optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# training loop

for iter in range(train_iters):

    # every eval_interval try to evaluate the loss
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
    
    x, y = get_batch('train')
    logits, loss = model(x, y)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

context = torch.zeros((1, 1), dtype=torch.int32, device=device)
print(decode(model.generate(context, max_new_tokens=500)[0].tolist()))

t1 = time.time()

print(f"Total time taken for script to run on {device}: {t1 - t0}")
