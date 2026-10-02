# Satellite Land-Cover Classifier & Change Detection Pipeline

A deep learning pipeline for classifying satellite imagery and detecting land-cover change using Sentinel-2 multispectral data.

Built and trained on Google Colab (T4 GPU).

---

## Overview

This project has two stages:

**Stage 1 — Land-Cover Classifier**
Fine-tune a ResNet50 (ImageNet pre-trained) on the EuroSAT RGB dataset to classify satellite image patches into 10 land-cover categories with **97% test accuracy**.

**Stage 2 — Temporal Change Detection**
Apply the trained classifier to two Sentinel-2 L2A scenes from different years over the same geographic area and map land-cover transitions at patch level.

---

## Dataset

[EuroSAT](https://github.com/phelber/EuroSAT) — 27,000 labeled satellite image patches (64×64 px, 10m/pixel) across 10 classes:

`AnnualCrop` · `Forest` · `HerbaceousVegetation` · `Highway` · `Industrial` · `Pasture` · `PermanentCrop` · `Residential` · `River` · `SeaLake`

Split: **80% train / 10% val / 10% test**

---

## Model

- **Architecture:** ResNet50 with a replaced 10-way classification head
- **Pre-training:** ImageNet weights (frozen backbone for initial training)
- **Training strategy:**
  - Phase 1: 10 epochs, head only — Adam, lr = 1e-3
  - Phase 2: 5 epochs, full fine-tuning — Adam, lr = 1e-4
- **Augmentation:** random horizontal & vertical flips, resize to 224×224
- **Normalization:** ImageNet mean/std

---

## Results

| Metric | Value |
|---|---|
| Test Accuracy | **97%** |
| Macro Precision | 0.97 |
| Macro Recall | 0.97 |
| Macro F1 | 0.97 |

Per-class F1 scores:

| Class | F1 |
|---|---|
| AnnualCrop | 0.96 |
| Forest | **0.99** |
| HerbaceousVegetation | 0.97 |
| Highway | 0.94 |
| Industrial | 0.98 |
| Pasture | 0.95 |
| PermanentCrop | 0.96 |
| Residential | **0.99** |
| River | 0.97 |
| SeaLake | **1.00** |

![Confusion Matrix](images/confusion_matrix.png)

---

## Change Detection Pipeline

The trained classifier is applied to Sentinel-2 L2A imagery (bands B02, B03, B04) over two time periods to identify land-cover transitions.
**Example**
**Area of interest:** Western Ghats region, Karnataka, India  
**Period 1:** Jan–Mar 2018  
**Period 2:** Jan–Mar 2024

![2018 vs 2024 Scene](images/scene_2018_vs_2024.png)

**Steps:**

1. Load Sentinel-2 GeoTIFF bands with `rasterio`
2. Apply per-band 2nd–98th percentile stretch to normalize reflectance to 8-bit
3. Tile the scene into 64×64 px patches (~640 m ground coverage, matching EuroSAT's scale)
4. Resize each patch to 224×224 and classify with the fine-tuned ResNet50
5. Compare patch-level predictions across both dates
6. Flag transitions of interest (e.g. Forest → AnnualCrop / Pasture / Industrial)

**Key engineering decisions:**

- **Patch scale:** 64 px native crop → resize to 224, not a direct 224 px crop. A direct 224 px crop covers ~2.24 km, completely outside the model's training distribution.
- **Radiometric normalization:** percentile stretch per band instead of a fixed divisor, which would otherwise compress most values to near-black pixels.

---

## Project Structure

```
├── satellite_deforestation_pipeline.ipynb   # Full pipeline (Colab-ready)
├── requirements.txt
├── confusion_matrix.png
├── scene_2018_vs_2024.png
└── .gitignore
```

---

## How to Run

1. Open the notebook in [Google Colab](https://colab.research.google.com/) with a **T4 GPU** runtime (`Runtime > Change runtime type > T4 GPU`)
2. Download [EuroSAT RGB](https://github.com/phelber/EuroSAT) and upload the zip, or mount Google Drive
3. Set `dataset_root` to point at the extracted folder
4. For change detection: download Sentinel-2 L2A B02/B03/B04 GeoTIFFs for two dates from the [Copernicus Browser](https://browser.dataspace.copernicus.eu/) and update the band paths
5. Run all cells top to bottom

> **Note:** Dataset files, `.tiff` imagery, and model weights (`.pth`) are not included in this repo due to size. See `.gitignore`.

---

## Requirements

```
torch
torchvision
numpy
pillow
rasterio
scikit-learn
matplotlib
seaborn
```

Install with:
```bash
pip install -r requirements.txt
```

---

## References

- Helber et al. (2019). *EuroSAT: A Novel Dataset and Deep Learning Benchmark for Land Use and Land Cover Classification.* IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing.
- [Copernicus Sentinel-2 L2A](https://sentinels.copernicus.eu/web/sentinel/missions/sentinel-2)
- [EuroSAT Dataset](https://github.com/phelber/EuroSAT)