# Segmentation-From-Attention

Source code for:

**Segmentation From Attention: Training-Free Layer Selection and One-Shot Tuning for Segmentation in Vision-Language Models**  
*TMLR 2026 (Journal-to-Conference / J2C)*  

Paper: https://openreview.net/pdf?id=a5lAwubXro

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

### COCO

```bash
python training_free_blip.py --config configs/coco_blip.yaml --multi_prompt
```

Remove `--multi_prompt` for single-prompt evaluation:

```bash
python training_free_blip.py --config configs/coco_blip.yaml
```

---

### PASCAL VOC

```bash
python training_free_blip.py --config configs/pascal_blip.yaml --multi_prompt
```

---

## One-Shot Segmentation

### COCO

```bash
python one_shot_blip.py --config configs/coco_blip_one_shot.yaml --multi_prompt
```

Single-prompt:

```bash
python one_shot_blip.py --config configs/coco_blip_one_shot.yaml
```

---

### PASCAL VOC

```bash
python one_shot_blip.py --config configs/pascal_blip_one_shot.yaml --multi_prompt
```

---

## Layer Ranking (InfoScore)

To compute attention layer ordering automatically:

```bash
python compute_infoscore_blip.py --config configs/coco_blip.yaml
```

This outputs ranked layers based on the InfoScore metric used in the paper.

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

Rayat Hossain  
rayat137@cs.ubc.ca
