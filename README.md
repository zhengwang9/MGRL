# MGRL

## Installation
- Clone the repo:
```bash
- git clone https://github.com/zhengwang9/MGRL.git && cd MGRL
```
- Create a conda environment and activate it:
```bash
conda create -n env python=3.9
conda activate env
pip install -r requirements.txt
```
## Image Preprocession and Feature Extraction

- We used [CLAM](https://github.com/mahmoodlab/CLAM) to split the slides and extract featurers of patches by [Ctranspath](https://github.com/Xiyue-Wang/TransPath)   

## Training

```bash
# train the model
cd train
python train_multihead_DFS.py
```

## Citation
