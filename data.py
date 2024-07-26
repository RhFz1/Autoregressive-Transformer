import os
import requests
import torch

split = 0.9

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

