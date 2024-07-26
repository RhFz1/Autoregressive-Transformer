import os
import time
import torch
import argparse
from data import train_data, val_data, encode, decode
from model import LanguageModel

parser = argparse.ArgumentParser(description="file contains raw implementation of a transformer")
parser.add_argument('--resume', help="Resume training or start new", default=False, type=bool)
args = parser.parse_args()


# for calculating script time
t0 = time.time()

# HyperParameters
learning_rate=3e-5
train_iters=1000
grad_accum_steps=16
n_embd = 768
n_heads = 12
n_layer =  12
block_size = 512
batch_size = 64
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 20
eval_interval = 5
dropout=0.3
out_dir = 'home/ec2-user/FAIR/Autoregressive-Transformer/models'
model_args = dict(n_layer=n_layer, n_head=n_heads, n_embd=n_embd, block_size=block_size,
                 vocab_size=None, dropout=dropout)


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

torch.set_float32_matmul_precision('high')
# Instantiating and moving the model to device.
model = LanguageModel()
print("Compiling model!!!")
# creating optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
print("Print using fused optimizer!!")
# best_val_loss so far
best_val_loss = 4.00

if args.resume:
    print("Resuming Training!!")

    checkpoint = torch.load(os.path.join(out_dir, 'ckpt_00.pt'), map_location=device)
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
        # Move optimizer's state to the same device
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device)
    best_val_loss = checkpoint['best_val_loss']

model = model.to(device=device)
model = torch.compile(model)
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
            torch.save(checkpoint, os.path.join(out_dir, 'ckpt_01.pt'))

    optimizer.zero_grad(set_to_none=True)
    lossi = 0.0
    for microstep in range(grad_accum_steps):
        x, y = get_batch('train')
        logits, loss = model(x, y)
        loss = loss / grad_accum_steps
        loss.backward() # accumulating the gradients
        lossi += loss.detach()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

context = torch.zeros((1, 1), dtype=torch.int32, device=device)
print(decode(model.generate(context, max_new_tokens=500)[0].tolist()))

t1 = time.time()

print(f"Total time taken for script to run on {device}: {t1 - t0}")
