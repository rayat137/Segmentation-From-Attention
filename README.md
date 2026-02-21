# Segmentation From Attention: Training-Free Layer Selection and One-Shot Tuning for Segmentation in Vision-Language Models (TMLR 2026) ![J2C](https://img.shields.io/badge/Certification-J2C-blue)

*Mir Rayat Imtiaz Hossain, Mennatullah Siam, Leonid Sigal, James J. Little*

This repository contains source code for our **TMLR 2026** paper titled, [Segmentation From Attention: Training-Free Layer Selection and One-Shot Tuning for Segmentation in Vision-Language Models](https://openreview.net/pdf?id=a5lAwubXro). ![J2C](https://img.shields.io/badge/Certification-J2C-blue)

---

## Overview

This repository provides implementations for:

- Training-free open-vocabulary segmentation using BLIP attention  
- Automatic layer ranking (InfoScore)  
- One-shot segmentation via lightweight tuning  

The code supports:

- COCO  
- PASCAL VOC  

and evaluates both **single-prompt** and **multi-prompt** settings.

## Dataset

You can download the dataset from [here](https://etsmtl365-my.sharepoint.com/personal/seyed-mohammadsina_hajimiri_1_ens_etsmtl_ca/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fseyed%2Dmohammadsina%5Fhajimiri%5F1%5Fens%5Fetsmtl%5Fca%2FDocuments%2FDIaM%2Fdatasets%2Ezip&parent=%2Fpersonal%2Fseyed%2Dmohammadsina%5Fhajimiri%5F1%5Fens%5Fetsmtl%5Fca%2FDocuments%2FDIaM&ga=1) provided by [DIaM](https://github.com/sinahmr/DIaM).

The data folder should look like this:

```
data
├── coco
│   ├── annotations
│   ├── train
│   ├── train2014
│   ├── val
│   └── val2014
└── pascal
|   ├── JPEGImages
|   └── SegmentationClassAug
```

#### The train/val split

The train/val split can be found in the diectory `src/lists/`. The list is borrowed from from https://github.com/Jia-Research-Lab/PFENet.

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/rayat137/Segmentation-From-Attention.git
cd Segmentation-From-Attention
```

---

### 2. Create environment (recommended)

```bash
python -m venv venv
source venv/bin/activate
```

(or use conda if preferred)

---

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

This installs **exact versions** for reproducibility.

---

### 4. Patch HuggingFace BLIP text module

Replace the HuggingFace BLIP text file with the provided modified version:

```bash
cp replace_huggingface/modeling_blip_text.py \
<your_python_env>/site-packages/transformers/models/blip/
```

Example:

```bash
cp replace_huggingface/modeling_blip_text.py \
venv/lib/python3.10/site-packages/transformers/models/blip/
```

This enables attention extraction required for segmentation.

---

## Training-Free Open Vocabulary Segmentation

#### COCO

```bash
python training_free_blip.py --config configs/coco_blip.yaml --multi_prompt
```

Remove `--multi_prompt` for single-prompt evaluation:

```bash
python training_free_blip.py --config configs/coco_blip.yaml
```

---

#### PASCAL VOC

```bash
python training_free_blip.py --config configs/pascal_blip.yaml --multi_prompt
```

Remove `--multi_prompt` for single-prompt evaluation, likewise:

```bash
python training_free_blip.py --config configs/pascal_blip.yaml 
```

---

## One-Shot Segmentation

#### COCO

```bash
python one_shot_blip.py --config configs/coco_blip_one_shot.yaml --multi_prompt
```

Single-prompt:

```bash
python one_shot_blip.py --config configs/coco_blip_one_shot.yaml
```

---

#### PASCAL VOC

```bash
python one_shot_blip.py --config configs/pascal_blip_one_shot.yaml --multi_prompt
```

Remove `--multi_prompt` for single-prompt evaluation, likewise:

```bash
python one_shot_blip.py --config configs/pascal_blip_one_shot.yaml
```


---

## Layer Ranking (InfoScore)

To compute attention layer ordering:

#### COCO

```bash
python compute_infoscore_blip.py --config configs/coco_blip.yaml
```

#### PASCAL VOC

```bash
python compute_infoscore_blip.py --config configs/pascal_blip.yaml
```

---

## Reproducibility Notes

- Always run `pip install -r requirements.txt` inside a fresh environment.
- CUDA version is not captured by pip. For full reproducibility, record:

```bash
nvidia-smi
python -c "import torch; print(torch.version.cuda)"
```

---

## Citation

If you use this code, please cite:

```bibtex
@article{
hossain2026segmentation,
title={Segmentation From Attention: Training-Free Layer Selection and One-Shot Tuning for Segmentation in {VLM}s},
author={Mir Rayat Imtiaz Hossain and Mennatullah Siam and Leonid Sigal and James J. Little},
journal={Transactions on Machine Learning Research},
issn={2835-8856},
year={2026},
url={https://openreview.net/forum?id=a5lAwubXro},
note={J2C Certification}
}
```

---

## License

MIT License.

---

## Contact

Mir Rayat Imtiaz Hossain 
rayat137@cs.ubc.ca
