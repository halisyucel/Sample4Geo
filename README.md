# Sample4Geo — Cross-View Geo-Localization + XAI

This repository is a fork of [Sample4Geo (ICCV 2023)](https://arxiv.org/abs/2303.11851). Two XAI (Explainable AI) techniques have been added on top of the original model: **GradCAM** and **Occlusion Sensitivity**.

> **Hacettepe University — Computer Vision Course Project**

---

## What is this project?

**Problem:** Given a street-level photo, find where it was taken by searching a database of satellite images — without GPS.

**How it works:**
- The model passes both the street photo and the satellite image through the same encoder, producing an embedding vector for each.
- Embeddings of matching pairs end up close together; non-matching pairs end up far apart.
- Finding the nearest embedding to a query = finding the most likely location.

**XAI contribution:** Visualizing *where* the model looks when it finds a correct match. Answering questions like "Did it recognize that building? That intersection?"

---

## Setup

```bash
pip install -r requirements.txt
pip install opencv-python albumentations
```

---

## Download Pretrained Weights

No need to train from scratch. Download the pretrained weights from Google Drive:

**[Download Weights (Google Drive)](https://drive.google.com/drive/folders/1PMuUqvDnCb216D8_ZDDJzDD3FxeH5BoA?usp=drive_link)**

Place the files under `pretrained/`:

```
pretrained/
└── university/
    └── convnext_base.fb_in22k_ft_in1k_384/
        └── weights_e1_0.9515.pth
```

---

## Dataset Setup

### University-1652 (Available now)

Download from Hugging Face: [https://huggingface.co/datasets/Skyy93/University-1652](https://huggingface.co/datasets/Skyy93/University-1652)

Then run the conversion script:

```bash
python convert_u1652_dataset.py
```

Expected folder structure:

```
data/
└── U1652/
    ├── train/
    │   ├── drone/
    │   └── satellite/
    └── test/
        ├── query_drone/
        └── gallery_satellite/
```

### VIGOR (Access pending)

Street↔satellite pairs from NYC, Seattle, San Francisco, and Chicago.
Access requested via institutional email.

Once received, place under:

```
data/
└── VIGOR/
```

---

## Step-by-Step Usage

### 1. Get Baseline Metrics

Measure the model's raw performance before running XAI:

```bash
# University-1652
python eval_university.py

# VIGOR (same-area)
python eval_vigor_same.py

# VIGOR (cross-area)
python eval_vigor_cross.py
```

Output metrics:
- **R@1:** Is the correct match ranked first?
- **R@5:** Is the correct match in the top 5?
- **Hit Rate:** Overall success rate

### 2. Run GradCAM

```bash
python examples/run_gradcam.py \
    --query   data/U1652/test/query_drone/0001/image-00.jpg \
    --gallery data/U1652/test/gallery_satellite/0001/satellite.jpg \
    --checkpoint pretrained/university/convnext_base.fb_in22k_ft_in1k_384/weights_e1_0.9515.pth \
    --output  xai_results/gradcam
```

### 3. Run Occlusion Sensitivity

```bash
python examples/run_occlusion.py \
    --query   data/U1652/test/query_drone/0001/image-00.jpg \
    --gallery data/U1652/test/gallery_satellite/0001/satellite.jpg \
    --checkpoint pretrained/university/convnext_base.fb_in22k_ft_in1k_384/weights_e1_0.9515.pth \
    --output  xai_results/occlusion \
    --patch-size 64 --stride 16
```

Occlusion Sensitivity is slow (~500 forward passes on a 384×384 image). It automatically uses GPU if available.

### 4. View Results

```
xai_results/
├── gradcam/
│   ├── query_gradcam.png       ← Original | Heatmap | Overlay
│   └── gallery_gradcam.png
└── occlusion/
    ├── query_occlusion.png     ← Original | Importance Map | Overlay | Faithfulness Curve
    └── gallery_occlusion.png
```

---

## XAI Methods

### GradCAM (Gradient-weighted Class Activation Mapping)

Uses gradients from the last convolutional layer to show which regions influenced the matching decision. Fast, requires access to model internals (white-box).

Red regions = the model assigned high importance to those areas.

### Occlusion Sensitivity

Slides a black patch across the image, covering one region at a time, and measures how much the cosine similarity score drops. A large drop means that region was important. Does not require model internals (black-box).

Also produces a **faithfulness curve**: as the most important regions are progressively masked, how much does the model degrade? This curve quantitatively measures how trustworthy the XAI explanation is.

| | GradCAM | Occlusion Sensitivity |
|---|---|---|
| Type | Gradient-based | Perturbation-based |
| Speed | Fast | Slow |
| Model access | White-box | Black-box |
| Faithfulness | Indirect | Direct |

---

## Folder Structure

```
Sample4Geo/
├── sample4geo/          ← Model, loss, dataset, trainer
├── xai/                 ← XAI modules (GradCAM, Occlusion Sensitivity)
├── examples/            ← Runnable scripts
│   ├── run_gradcam.py
│   └── run_occlusion.py
├── pretrained/          ← Downloaded model weights (not in git)
├── data/                ← Datasets (not in git)
├── xai_results/         ← Output visualizations (not in git)
├── eval_university.py
├── eval_vigor_same.py
├── eval_vigor_cross.py
└── convert_u1652_dataset.py
```

---

## References

- [Sample4Geo Paper (ICCV 2023)](https://arxiv.org/abs/2303.11851)
- [Original Repository](https://github.com/Skyy93/Sample4Geo)
- [Pretrained Weights](https://drive.google.com/drive/folders/1PMuUqvDnCb216D8_ZDDJzDD3FxeH5BoA?usp=drive_link)
- [University-1652 on Hugging Face](https://huggingface.co/datasets/Skyy93/University-1652)
