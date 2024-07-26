import torch
import glob
from model import LanguageModel
from collections import OrderedDict
from data import vocab, vocab_to_id, id_to_vocab, encode, decode

model_path = glob.glob('/home/syednoor/Desktop/FAIR/Autoregressive-Transformer/models/*.pt')[-1]
# Load the model
model = LanguageModel()
checkpoint = torch.load(model_path, map_location='cuda')

new_keys = {key: key.replace('_orig_mod.', '') for key in checkpoint['model'].keys()}
print(checkpoint['model_args'])
checkpoint['model'] = OrderedDict((new_keys.get(k, k), v) for k, v in checkpoint['model'].items())
model.load_state_dict(checkpoint['model'])

# Move the model to the GPU
model = model.to('cuda')

# Inference
model.eval()
prompt = "The meaning of life is"
enc = encode(prompt)
inp = torch.tensor(enc, dtype=torch.long).unsqueeze(0).to('cuda') # adding a batch dimension

print(decode(model.generate(inp, max_new_tokens=500)[0].tolist()))

