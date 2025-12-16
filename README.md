### AN2DL_CV_Challenge
Repository destined for Kaggle Challenge on Breast Cancer Multi-Class Classification Task on Low Resolution Histopathological Images

# AN2DL Second Challenge - Bratwurst & Caipirinha Team

![Polimi Logo](polimi.png) ![AI Lab Logo](airlab.jpeg)

This repository contains the code, data preprocessing pipelines, models, and evaluation scripts for the **AN2DL Second Challenge**, a multiclass classification task of breast tissue images into four molecular subtypes: Luminal A, Luminal B, HER2+, and Triple Negative.

---

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Methods](#methods)
  - [Data Preprocessing](#data-preprocessing)
  - [Model Architecture](#model-architecture)
- [Experiments](#experiments)
- [Results](#results)
- [Requirements](#requirements)
- [Usage](#usage)
- [Team](#team)
- [References](#references)

---

## Project Overview
The goal of this project is to classify breast tissue images into molecular subtypes. Challenges include:
- Small dataset size (573 cleaned images)
- Low image resolution (~1 MB PNGs)
- Class imbalance, especially for Triple Negative
- Presence of distractor images (green blobs, unrelated content)

To address these challenges, the team implemented:
- Masked-image preprocessing
- Tiling of images to focus on local features
- Transfer learning using pretrained CNNs (DenseNet-121, ConvNeXt-Tiny)
- Soft-voting aggregation of tile predictions
- Customized Focal Loss to handle class imbalance

---

## Dataset
- **Training images:** 690 original images (after cleaning, 573)
- **Image type:** PNG, ~1 MB each
- **Classes:** Luminal A, Luminal B, HER2+, Triple Negative
- **Masks:** Provided for all tissue regions
- **Distractors:** Green blobs, Shrek images, and other irrelevant samples (filtered during preprocessing)

---

## Methods

### Data Preprocessing
1. Removal of irrelevant images using color-based scoring (green/brown ratio)
2. Cropping of images to squares around masked areas
3. Application of segmentation masks
4. Tiling: select tiles with sufficient masked area for training
5. Ensuring at least 3 tiles per test sample

### Model Architecture
- Input: 128 × 128 image tiles
- Backbone: DenseNet-121 (best performance), ConvNeXt-Tiny (alternative)
- Augmentations: Random rotations, flips
- Batch normalization used for DenseNet, layer normalization for ConvNeXt
- Loss: Customized Focal Loss with class weighting
- Optimizer: Lion with tuned learning rates (separate for backbone and classification head)
- Prediction: Soft-voting over tile-level outputs

---

## Experiments
- **One-vs-Rest classifiers:** Tried but did not improve F1-score
- **Hierarchical ensembles:** Separated Luminal vs aggressive subtypes; error propagation limited effectiveness
- **Two-branch ensembles:** Full image + tiles, masked + unmasked inputs; no performance gain
- **Final approach:** Single model with tiling + soft-voting

---

## Results
- Initial CNN from scratch: F1-score ~0.12
- Final model with tiling and DenseNet-121: F1-score ~0.3912
- Cross-validation was challenging due to limited computational resources
- Grad-CAM visualizations confirmed the model focuses on relevant tissue features

---

## Requirements
- Python >= 3.8
- PyTorch >= 2.0
- torchvision
- Optuna
- numpy, pandas, scikit-learn, matplotlib
- OpenCV
- [Optional] Google Colab or GPU for training

Install dependencies:
```bash
pip install -r requirements.txt
