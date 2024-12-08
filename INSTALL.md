# Installation

## Dependency Setup
Create a new conda virtual environment
```
conda create -n sharpbalance python=3.8 -y
conda activate sharpbalance
```

Install [Pytorch](https://pytorch.org/)>=1.13.0, [torchvision](https://pytorch.org/vision/stable/index.html)>=0.14.0 following official instructions. For example:
```
pip install torch==1.13.0+cu116 torchvision==0.14.0+cu116 -f https://download.pytorch.org/whl/torch_stable.html
```
Install required packages:
```
pip install functorch==1.13
pip install timm==0.6.12 tensorboardX six

```