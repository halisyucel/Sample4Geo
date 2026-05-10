# Project Progress Log

**Course:** Hacettepe University — Computer Vision  
**Topic:** Explainable AI (XAI) on Cross-View Geo-Localization  
**Base paper:** Sample4Geo (ICCV 2023)  
**Fork:** https://github.com/halisyucel/Sample4Geo

---

## Group

| Member | Dataset |
|---|---|
| Halis Yücel | VIGOR (NYC, Seattle, San Francisco, Chicago) |
| Kuzey Ersoy | University-1652 |

---

## Completed

### 1. XAI Method Selection

Two XAI techniques chosen based on methodological contrast:

- **GradCAM** — gradient-based, white-box. Uses gradients from the last convolutional layer to produce an attention heatmap. Fast, requires model internals.
- **Occlusion Sensitivity** — perturbation-based, black-box. Slides a patch over the image and measures cosine similarity drop at each position. Includes a faithfulness curve (progressively masks top-important regions and measures score degradation).

Rejected candidates: Saliency Maps, Integrated Gradients, Feature Visualization, LIME.  
Reason for rejection: either methodologically too close to GradCAM (gradient-based), or less interpretable for spatial geo-localization tasks.

---

### 2. XAI Module Implementation (`xai/`)

Built from scratch on top of the original codebase. Two files:

**`xai/gradcam.py` — `GradCAMExtractor`**
- Automatically finds the target layer in ConvNeXt
- Registers forward/backward hooks
- `generate_cam(image, target_embedding)` → heatmap
- `generate_pair_cam(query, gallery)` → both directions
- `save_visualization(image, cam, path)` → 3-panel figure (original / heatmap / overlay)

**`xai/occlusion_sensitivity.py` — `OcclusionSensitivity`**
- `compute_sensitivity(query, gallery_emb, patch_size, stride)` → importance map
- `compute_pair_sensitivity(query, gallery)` → both directions at once
- `compute_faithfulness(query, gallery_emb, importance_map, steps)` → faithfulness curve
- `save_visualization(image, map, path, faithfulness_data)` → 4-panel figure (original / map / overlay / faithfulness curve)

Removed files (not needed): `saliency.py`, `integrated_gradients.py`, `feature_viz.py`, `similarity_viz.py`

---

### 3. Example Scripts (`examples/`)

Two standalone runnable scripts, one per XAI method:

**`examples/run_gradcam.py`**
```bash
python examples/run_gradcam.py \
    --query data/U1652/test/query_drone/0001/image-00.jpg \
    --gallery data/U1652/test/gallery_satellite/0001/satellite.jpg \
    --checkpoint pretrained/university/convnext_base.fb_in22k_ft_in1k_384/weights_e1_0.9515.pth \
    --output xai_results/gradcam
```

**`examples/run_occlusion.py`**
```bash
python examples/run_occlusion.py \
    --query data/U1652/test/query_drone/0001/image-00.jpg \
    --gallery data/U1652/test/gallery_satellite/0001/satellite.jpg \
    --checkpoint pretrained/university/convnext_base.fb_in22k_ft_in1k_384/weights_e1_0.9515.pth \
    --output xai_results/occlusion \
    --patch-size 64 --stride 16
```

Both scripts support `--help` for full argument list.

---

### 4. README Updates

- **`README.md`** — Full rewrite. Covers: project explanation, setup, pretrained weights download, dataset setup (University-1652 + VIGOR), step-by-step usage, XAI method comparison table, folder structure.
- **`xai/README.md`** — Full rewrite. Covers: how each method works internally, code examples, parameter guide (`patch_size` / `stride` tradeoffs), output format explanation.

---

### 5. Pretrained Weights Download

Downloaded from Google Drive: https://drive.google.com/drive/folders/1PMuUqvDnCb216D8_ZDDJzDD3FxeH5BoA

All weights available under `pretrained/`:

| Dataset | Weights file | Status |
|---|---|---|
| University-1652 | `weights_e1_0.9515.pth` | ✅ |
| VIGOR (same-area) | `weights_e40_0.7786.pth` | ✅ |
| VIGOR (cross-area) | `weights_e40_0.6109.pth` | ✅ |
| CVUSA | `weights_e40_98.6830.pth` | ✅ |
| CVACT | `weights_e36_90.8149.pth` | ✅ |

---

## Waiting

- **VIGOR dataset:** Access request sent via form using institutional email (b2220356137@cs.hacettepe.edu.tr). Waiting for approval.

---

### 6. University-1652 Dataset

Downloaded directly from the original source (Google Drive, sent by dataset author via institutional email request to zdzheng12@gmail.com).

**Download link:** https://drive.google.com/file/d/1iVnP4gjw-iHXa0KerZQ1IfIO0i1jADsR/view

Placed under `data/U1652/`. Dataset arrived in the correct folder structure — no conversion needed.

```
data/U1652/
├── train/
│   ├── drone/
│   ├── satellite/
│   ├── google/
│   └── street/
└── test/
    ├── query_drone/
    ├── gallery_satellite/
    ├── query_satellite/
    ├── gallery_drone/
    └── ...
```

---

### 7. Baseline Evaluation — University-1652

**Environment:** MacBook Pro, Apple M4, 16GB RAM (no CUDA GPU)  
**Command:**
```bash
cd /Users/halisyucel/Projects/Sample4Geo
python3 eval_university.py
```

**Issues encountered and fixes applied to `eval_university.py`:**

1. **CUDA error on load** — Weights were saved on GPU. Fixed by adding `map_location=torch.device('cpu')` to `torch.load()`.

2. **Device selection** — Original code defaulted to CPU when CUDA unavailable. Updated to prefer MPS (Apple Silicon GPU) when available:
   ```python
   device = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
   ```

3. **`autocast` incompatibility** — `torch.cuda.amp.autocast` crashes on non-CUDA devices. Replaced with a device-aware context manager in `sample4geo/trainer.py` that only activates autocast on CUDA.

4. **`num_workers=4` deadlock on MPS** — macOS multiprocessing + MPS caused the DataLoader to freeze after the first batch. Fixed by setting `num_workers=0`.

5. **`pin_memory=True` warning on MPS** — MPS does not support pinned memory. Set `pin_memory=False` in both DataLoaders.

**Runtime:** ~35 minutes on M4 (CPU+MPS, no CUDA)

**Results:**

| Metric | Result |
|---|---|
| Recall@1 | 92.67% |
| Recall@5 | 97.69% |
| Recall@10 | 98.24% |
| Recall@top1% | 98.28% |
| AP | 93.82% |

Full results saved in `results_university.txt`.  
Note: Paper reports R@1 = 95.15% — small gap likely due to evaluation config differences.

---

## Up Next

1. Run XAI on sample query-gallery pairs  
   `python examples/run_gradcam.py ...`  
   `python examples/run_occlusion.py ...`

2. Receive VIGOR dataset → run baseline eval, then repeat XAI  
   (`eval_vigor_same.py` and `eval_vigor_cross.py`)

3. Analysis  
   - Compare GradCAM vs Occlusion Sensitivity heatmaps
   - Identify successful vs failed matches, compare their heatmaps
   - Compare University-1652 (drone↔satellite) vs VIGOR (street↔satellite) results
   - Compute and report faithfulness scores

4. Reports  
   - Progress report
   - Final report
   - Presentation
