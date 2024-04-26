import os
import math
import time
import pickle

import torch
import numpy as np
from torch.nn.parallel import DistributedDataParallel as DDP


# I/O
out_dir = 'output'
eval_iters = 200
log_interval = 1
eval_interval = 2000

# data
dataset = 'shakespere'
batch_size = 32
block_size = 128

# system
device = 'cuda'

# learning rate decay settings
decay_lr = True # Set to true to decay the learning rate
warmup_iters = 2000
learning_rate = 3e-4
lr_decay_iters = 600000
min_lr = 3e-6

ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
device_type = 'cuda' if 'cuda' in device else 'cpu'
# Nice Joke!!
data_dir = os.path.join('data', dataset)
def get_batch(split):

    if split == 'train':
        data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
    else:
        data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')

    ix = torch.randint(len(data) - block_size, (batch_size, ))
    x = torch.stack([torch.from_numpy((data[i: i + block_size]).astype(np.int32)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i + 1:i + block_size + 1]).astype(np.int32)) for i in ix])

    if device_type == 'cuda':
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y

# placing these here, can get ovverided if resuming training
iter_num = 0
best_val_loss = 1e9

# LR scheduler (cosine warmup)
def get_lr(it):
    # 1) Linear for warmup iters
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    
    # 2) if it > decay, then min learning rate
    if it > lr_decay_iters:
        return min_lr
    
    # 3) if in between use cosine decay 
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1

    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)

# Loss Estimator
@torch.no_grad()
def estimate_losses():
    out = {}
    model.eval() # This sets the model in evaluation mode i.e., limiting gradient computations and backprop graphs.

    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            with ctx:
                logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out



# Train Loop
X, Y = get_batch('train')
t0 = time.time()
local_iter_num = 0

while True:

    # determine the learning rate of the current iteration
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    
    if iter_num % eval_interval == 0:
        losses = estimate_losses()
        print(f"step {iter_num}: train loss {losses['train']: .4f}, val loss {losses['val']: .4f}")

        if wandb_log:
            wandb.log({
                "iter": iter_num,
                "train/loss": losses['train'],
                "val/loss": losses['val'],
                "lr": lr,
                "mfu": running_mfu*100, # convert to percentage
            })
        if losses['val'] < best_val_loss or always_save_checkpoint:
            best_val_loss = losses['val']
            if iter_num > 0:
                checkpoint = {
                    'model' : raw_model.state_dict(),
                    'optimizer' : optimizer.state_dict(),
                    'model_args' : model_args,
                    'iter_num' : iter_num,
                    'best_val_loss' : best_val_loss,
                    'config' : config
                }
                print(f"saving checkpoint to {out_dir}")
                torch.save(checkpoint, os.path.join(out_dir, 'ckpt.pt'))
        if iter_num == 0 and eval_only:
            break

    for micro_step in range(gradient_accumulation_steps):

        logits, loss = model(X ,Y)

        loss = loss / gradient_accumulation_steps

        optimizer.zero_grad(set_to_none = True)

        loss.backward()

        optimizer.step()

        t1 = time.time()
        dt = t1 - t0
        t0 = t1
        if iter_num % log_interval == 0 and master_process:
            # get loss as float. note: this is a CPU-GPU sync point
            # scale up to undo the division above, approximating the true total loss (exact would have been a sum)
            lossf = loss.item() * gradient_accumulation_steps
            if local_iter_num >= 5: # let the training loop settle a bit
                mfu = raw_model.estimate_mfu(batch_size * gradient_accumulation_steps, dt)
                running_mfu = mfu if running_mfu == -1.0 else 0.9*running_mfu + 0.1*mfu
            print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%")
        iter_num += 1
        local_iter_num += 1

        if iter_num > max_iters:
            break
if ddp:
    ddp.clear_group()

        

