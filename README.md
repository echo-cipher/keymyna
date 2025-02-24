# KeyMyna

Official repository of **"Myna-Style Contrastive Pre-Training Improves Music Audio Key Detection"**

## Introduction
KeyMyna introduces a novel approach to music key detection by leveraging Myna-style contrastive pre-training on Mel spectrograms with vertical patches. This method enables competitive performance out of the box and achieves state-of-the-art (SOTA) results using shallow but wide MLPs trained on extracted features. Unlike previous approaches, KeyMyna does not require complex data augmentation policies due to the robustness of its self-supervised pre-training, which we demonstrate in the paper.

## Repository Structure
```
📂 KeyMyna
├── 📜 README.md        # This file
├── 📜 requirements.txt # Required dependencies
├── 📜 parse_args.py    # Parse command-line arguments for train/evaluation
├── 📜 ho.py            # Automated hyperparameter search, as detailed in the paper
├── 📜 hp_slurm.py      # Parallelized hyperparameter search on a SLURM cluster
└── 📜 train.py         # Main training script
```

## Installation
To get started with KeyMyna, first clone the repository and install dependencies:
```sh
git clone https://github.com/echo-cipher/KeyMyna.git
cd KeyMyna
pip install -r requirements.txt
```

## Embedding Files
This repository works with **embedding files**, which are pickle files with extracted (Myna) embeddings. We do this to avoid recomputing embeddings with every forward pass. 
This pickle file contains a dictionary with entries 'train', 'valid', and 'test'. Each entry in 'train', 'valid', and 'test' contains another dictionary containing 'indices', 'embeddings', and 'labels'. The 'embeddings' value is a PyTorch tensor of shape (n_embeds, dim). The 'labels' value is a list of length n (number of songs in the dataset). The 'indices' value is a PyTorch LongTensor of shape (n_embeds,) containing indices where indices[i] corresponds to the label index of embeddings[i] (since we have multiple embeddings per song, this is an easy way to aggregate predictions on the song, rather than embedding, level). In other words, labels[indices[i]] gives the label of the embedding vector found in embeddings[i].

## Usage
### Training an MLP on KeyMyna Features
```sh
python train.py --embeds_file PATH/TO/EMBEDS.pkl --arch 1024 d0.9 1024 --epochs 50 --batch_size 64 --checkpoint_path PATH/TO/CHECKPOINT
```

### Running Evaluation
All you need to do for evaluation is pass the embeddings file (pickle), the model checkpoint with --resume, and the --evaluate argument:
```sh
python train.py --embeds_file PATH/TO/EMBEDS.pkl --arch 1024 d0.9 1024 --resume PATH/TO/MODEL.pth --evaluate
```

## Pretrained Models
Pretrained models (MLP heads on top of [Myna-Vertical](https://github.com/ghost-signal/myna) are available for download:
- **KeyMyna MLP (GiantSteps)**: [Download](https://drive.google.com/file/d/1pgBB9rMd0fY8fS_ROonai7mBWDoDHLey/view?usp=sharing)
- **KeyMyna MLP (McGill Billboard)**: [Download](https://drive.google.com/file/d/1pgBB9rMd0fY8fS_ROonai7mBWDoDHLey/view?usp=sharing)

## Results
KeyMyna achieves state-of-the-art results on key detection benchmarks:
| Model | Dataset | MIREX Weighted Score |
|--------|------------|------------------|
| KeyMyna (ours) | GiantSteps | 75.91 |
| KeyMyna (ours) | McGill Billboard | 84.35 |

## Citation
If you use KeyMyna in your research, please cite:
```
@inproceedings{keymyna,
  author    = {Anonymous Authors},
  title     = {Myna-Style Contrastive Pre-Training Improves Music Audio Key Detection},
  booktitle = {Under anonymous submission},
  year      = {2025},
  url       = {https://github.com/echo-cipher/KeyMyna}
}
```

