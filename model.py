import torch
import torch.nn as nn
import torch.nn.functional as F
from data import vocab_size


# Model Parameters
n_embd = 384
n_heads = 8
n_layer =  4
block_size = 128
device = 'cuda' if torch.cuda.is_available() else 'cpu'
dropout=0.3
model_args = dict(n_layer=n_layer, n_head=n_heads, n_embd=n_embd, block_size=block_size,
                 vocab_size=None, dropout=dropout)

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
        idx = idx.to('cuda')
        for _ in range(max_new_tokens):
            idx_cond = idx[: , -block_size: ]
            logits, loss = self.forward(idx_cond) # This is (B, T, C)
            logits = logits[:,-1,:] # Just taking the last chars preds (B, C)
            probs = F.softmax(logits, dim=-1)
            preds = torch.multinomial(probs, num_samples=1) # (B,1)
            idx = torch.cat((idx, preds), dim=-1)
        return idx