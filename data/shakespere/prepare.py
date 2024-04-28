import os
import tiktoken
import requests
import numpy as np


train_bin_path = os.path.join(os.path.dirname(__file__), 'train.bin')
val_bin_path = os.path.join(os.path.dirname(__file__), 'val.bin')

if not os.path.exists(train_bin_path) or not os.path.exists(val_bin_path):
    
    input_file_path = os.path.join(os.path.dirname(__file__), 'input.txt')

    if not os.path.exists(input_file_path):
        # Downloading the shakespere corpus
        with open(input_file_path,mode='w') as file:
            data_url_path = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
            data = requests.get(data_url_path).text
            file.write(data)

    # Reading the input text
    with open(input_file_path, mode='r', encoding ='utf-8') as file:
        data = file.read()
    
    if os.path.exists(input_file_path):
        os.remove(input_file_path)

    # Parameters for splitting the data
    N = len(data)
    split = 0.8

    # Train and Val before conversion to embeddings
    train_data = data[:int(split * N)]
    val_data = data[int(split * N):]

    # Converting Train and Val to embeddings
    encoder = tiktoken.get_encoding('gpt2')
    train_encs = encoder.encode_ordinary(train_data)
    val_encs = encoder.encode_ordinary(val_data)

    # Converting Train and Val to np arrays
    train_encs = np.array(train_encs, dtype=np.uint32)
    val_encs = np.array(val_encs, dtype=np.uint32)

    # Saving embeds to bin files
    train_encs.tofile(os.path.join(os.path.dirname(__file__), 'train.bin'))
    val_encs.tofile(os.path.join(os.path.dirname(__file__), 'val.bin'))