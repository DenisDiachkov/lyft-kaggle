# Lyft autonomous driving competition

## Installation

```bash
conda create -n lyft python=3.7
conda activate lyft
conda install -c conda-forge pytorch-lightning=0.9.0
conda install torchvision=0.7.0
pip install -r requirements.txt
```

## Train model

```bash
python codes/main.py --gpu 0,1
```
