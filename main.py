import os
import torch
import numpy as np
import torch.nn.functional as F
from dotenv import load_dotenv
load_dotenv()

# data
block_size = 8
batch_size = 32

# device
device_type = 'cuda'

data_dir = os.environ.get('data_path') 
def data_loader(split):
    X, y = np.array(), np.array()
    if split == 'train':
        data_path = os.path.join(data_dir, 'train.bin')
        data = np.memmap(data_path, dtype=np.uint16, mode='r')
    elif split == 'val':
        data_path = os.path.join(data_dir, 'val.bin')
        data = np.memmap(data_path, dtype=np.uint16, mode='r')
    else:
        data_path = os.path.join(data_dir, 'test.bin')
        data = np.memmap(data_path, dtype=np.uint16, mode='r')
    
    ix = torch.randint(len(data) - block_size, (batch_size, ))
    x = torch.stack([torch.from_numpy(data[i : i + block_size].astype(np.int16)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i + 1 : i + block_size + 1].astype(np.int16)) for i in ix])

    if device_type == 'cuda':
        x, y = x.pin_memory().to(device=device_type,non_blocking=True),y.pin_memory().to(device=device_type,non_blocking=True)
    
    return x, y


        
