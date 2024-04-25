import os
import math
import time
import pickle

import torch
import numpy as np
from torch.nn.parallel import DistributedDataParallel as DDP

# data
dataset = 'shakespere'
batch_size = 32
block_size = 128

# system
device = 'cuda'

# learning rate decay settings
warmup_iters = 2000
learning_rate = 3e-4
lr_decay_iters = 600000
min_lr = 3e-6



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

    
