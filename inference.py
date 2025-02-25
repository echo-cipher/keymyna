'''
Minimal inference code to predict key of audio using KeyMyna
'''

import matplotlib.pyplot as plt
import torch
from torch import nn
from transformers import AutoModel

# if using mac, switch to soundfile as backend
import torchaudio
torchaudio.set_audio_backend('soundfile')

# hyperparameters
myna_huggingface = 'oriyonay/myna-vertical'
adapter_path = 'keymyna-bb.pth'

# constant list of keys
keys = [
    'c major', 'c minor', 'db major', 'db minor', 'd major', 
    'd minor', 'eb major', 'eb minor', 'e major', 'e minor', 
    'f major', 'f minor', 'gb major', 'gb minor', 'g major', 
    'g minor', 'ab major', 'ab minor', 'a major', 'a minor', 
    'bb major', 'bb minor', 'b major', 'b minor'
]

# get device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'==> Using {device}')

# load embedding model
myna = AutoModel.from_pretrained(
    myna_huggingface, 
    trust_remote_code=True
).to(device)
myna.eval()

# load MLP adapter
proj = nn.Sequential(
    nn.Linear(384, 2048),
    nn.ReLU(),
    nn.Dropout(0.75),
    nn.Linear(2048, 24)
).to(device)
proj.load_state_dict(torch.load(adapter_path, map_location=device))
proj.eval()

# change number of samples per embedding from 50k to 100k
myna.config.n_samples = 100000 # KeyMyna constant
myna.config.n_frames = myna.config._get_n_frames(myna.config.n_samples)

@torch.no_grad()
def predict(filepath: str):
    embeds = myna.from_file(filepath)
    preds = proj(embeds).mean(dim=0).softmax(dim=0).cpu().numpy()

    plt.figure(figsize=(10, 6))
    plt.bar(keys, preds)
    plt.xticks(rotation=90)
    plt.xlabel('Keys')
    plt.ylabel('Predicted Values')
    plt.title('Histogram of Predicted Key Probabilities')
    plt.tight_layout()
    plt.show()

predict('your_file.wav')
