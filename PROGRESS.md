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

### 5. Pretrained Weights Download (Partial)

Downloaded from Google Drive: https://drive.google.com/drive/folders/1PMuUqvDnCb216D8_ZDDJzDD3FxeH5BoA

**Status:**

| Dataset | Weights file | Status |
|---|---|---|
| CVUSA | `weights_e40_98.6830.pth` | ✅ Available |
| CVACT | `weights_e36_90.8149.pth` | ✅ Available |
| University-1652 | `weights_e1_0.9515.pth` | ⏳ Downloading manually |

The zip extraction partially failed — the university `.pth` file was not extracted. Downloading manually from Drive and placing at:  
`pretrained/university/convnext_base.fb_in22k_ft_in1k_384/weights_e1_0.9515.pth`

---

## In Progress

- [ ] University-1652 weights download (manual)

---

## Up Next

1. Download University-1652 dataset from Hugging Face  
   `load_dataset("layumi/university-1652")`

2. Run conversion script  
   `python convert_u1652_dataset.py`

3. Run baseline evaluation  
   `python eval_university.py`  
   → Get R@1, R@5, Hit Rate numbers before any XAI

4. Run XAI on sample query-gallery pairs  
   `python examples/run_gradcam.py ...`  
   `python examples/run_occlusion.py ...`

5. VIGOR dataset (waiting for institutional email access)  
   Once received: repeat steps 2–4 for VIGOR

6. Analysis  
   - Compare GradCAM vs Occlusion Sensitivity heatmaps
   - Identify successful vs failed matches, compare their heatmaps
   - Compare University-1652 (drone↔satellite) vs VIGOR (street↔satellite) results
   - Compute and report faithfulness scores

7. Reports  
   - Progress report
   - Final report
   - Presentation
