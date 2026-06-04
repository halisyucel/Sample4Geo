# Project Progress Log

**Course:** Hacettepe University — BBM416 Computer Vision  
**Topic:** Explainable AI (XAI) on Cross-View Geo-Localization  
**Base paper:** Sample4Geo (ICCV 2023)  
**Fork:** https://github.com/halisyucel/Sample4Geo  
**Upstream:** https://github.com/Skyy93/Sample4Geo

---

## Group

| Member | Dataset |
|---|---|
| Halis Yücel | VIGOR (NYC, Seattle, San Francisco, Chicago) |
| Kuzey Ersoy | University-1652 |

---

## Completed

### 1. Fork Setup

Forked the original Sample4Geo repository (Skyy93/Sample4Geo, ICCV 2023) into halisyucel/Sample4Geo.

The upstream already included a refactoring commit (by KonradHabel, 2023-09-26) that:
- Renamed `reident/` → `sample4geo/`
- Simplified all eval/train scripts
- Replaced `environment.yml` with `requirements.txt`

This refactored version was the starting point for all XAI work.

---

### 2. XAI Method Selection

Two XAI techniques chosen based on methodological contrast:

- **GradCAM** — gradient-based, white-box. Uses gradients from the last convolutional layer to produce an attention heatmap. Fast, requires model internals.
- **Occlusion Sensitivity** — perturbation-based, black-box. Slides a patch over the image and measures cosine similarity drop at each position. Includes a faithfulness curve (progressively masks top-important regions and measures score degradation).

Rejected candidates: Saliency Maps, Integrated Gradients, Feature Visualization, LIME.  
Reason for rejection: either methodologically too close to GradCAM (gradient-based), or less interpretable for spatial geo-localization tasks.

---

### 3. Pretrained Weights Download

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

### 4. University-1652 Dataset

Downloaded directly from the original source (Google Drive, sent by dataset author via institutional email request to zdzheng12@gmail.com).

**Download link:** https://drive.google.com/file/d/1iVnP4gjw-iHXa0KerZQ1IfIO0i1jADsR/view

Dataset is stored in Google Drive under `vision-datasets/University-Release/`.

```
University-Release/
├── train/
│   ├── drone/, satellite/, google/, street/
└── test/
    ├── query_drone/, gallery_satellite/
    ├── query_satellite/, gallery_drone/
    └── query_street/, gallery_street/, 4K_drone/
```

---

### 5. VIGOR Dataset

Access approved. Dataset stored in Google Drive under `vision-datasets/VIGOR/`.

```
VIGOR/
├── Chicago/, NewYork/, SanFrancisco/, Seattle/
│   ├── panorama/   ← street-view (360° panoramas)
│   └── satellite/  ← satellite tiles
└── splits/
    └── {city}/
        ├── satellite_list.txt
        ├── pano_label_balanced.txt
        ├── same_area_balanced_train.txt
        └── same_area_balanced_test.txt
```

**Note:** The `VigorDataset` class in the codebase expects `data_folder/ground/{city}/` but the Drive uses `panorama/`. The notebook creates symlinks at runtime to bridge this mismatch:
```
/content/vigor_data/ground/{city}/ → Drive/.../VIGOR/{city}/panorama/
/content/vigor_data/satellite/{city}/ → Drive/.../VIGOR/{city}/satellite/
/content/vigor_data/splits/{city}/ → Drive/.../VIGOR/splits/{city}/
```

---

### 6. Baseline Evaluation — University-1652

**Results (Colab A100, 2026-06-02):**

| Metric | Result | Paper |
|---|---|---|
| Recall@1 | 92.66% | 95.15% |
| Recall@5 | 97.69% | — |
| Recall@10 | 98.24% | — |
| Recall@top1% | 98.29% | — |
| AP | 93.81% | — |

Note: The ~2.5% gap vs paper is consistent across both Mac (MPS) and Colab (CUDA) runs, suggesting it is not a hardware/precision issue. Likely caused by a difference in eval split or test-time augmentation in the original paper setup.

**Runtime note:** Feature extraction took 2h 9m due to reading 37,855 files from Drive (~26s/batch on A100). Fixed by copying dataset to local Colab disk first (see Section 9.5 below).

---

### 7. Migration to Google Colab

**Reason:** VIGOR dataset arrived; running evaluation and XAI on a CUDA GPU is significantly faster than M4 MPS. Colab T4/A100 also ensures reproducibility with the original codebase.

**Changes made:**

1. **Reverted Mac/MPS fixes:**
   - `eval_university.py` — restored to upstream: `device = 'cuda' if ... else 'cpu'`, original `num_workers`, `pin_memory=True`
   - `sample4geo/trainer.py` — restored to upstream: removed custom `autocast` context manager, restored `from torch.cuda.amp import autocast`

2. **Removed Mac-specific and intermediate files:**
   - `xai/` — all XAI code moved into the notebook
   - `examples/` — all example scripts moved into the notebook
   - `find_xai_pairs.py` — pair selection logic moved into the notebook
   - `xai_pairs.txt` — superseded by notebook
   - `convert_u1652_dataset.py` — not needed (Drive has the dataset directly)
   - `results_university.txt` — archived above; will re-run on Colab

3. **Created `notebooks/sample4geo-xai.ipynb`** — single Colab notebook covering the full pipeline (see Section 8 below).

---

### 8. Colab Notebook (`notebooks/sample4geo-xai.ipynb`)

Full pipeline notebook designed for Google Colab with CUDA.

**Structure:**

| Section | Content |
|---|---|
| 0 | Environment setup: Drive mount, repo clone, `pip install`, path config |
| 1 | Pretrained weights — existence check (weights loaded directly from Drive) |
| 2 | Dataset path setup + VIGOR symlink fix |
| 3 | Baseline eval — University-1652 (D2S) |
| 4 | Baseline eval — VIGOR same-area and cross-area |
| 5 | XAI class definitions: `GradCAMExtractor`, `OcclusionSensitivity`, visualization helpers |
| 6 | XAI pair selection — University-1652 (5 successful + 3 failed) |
| 7 | GradCAM — University-1652 (successful + failed) |
| 8 | Occlusion Sensitivity — University-1652 (successful + failed) with faithfulness curves |
| 9 | XAI pair selection — VIGOR |
| 10 | GradCAM — VIGOR |
| 11 | Occlusion Sensitivity — VIGOR |
| 12 | Analysis scaffold + results summary table + save to Drive |

**Key design decisions:**
- All XAI classes defined inline in the notebook (no `xai/` module dependency)
- GradCAM uses `register_full_backward_hook` (safer than deprecated `register_backward_hook`)
- Hook cleanup via `remove_hooks()` to prevent memory accumulation across cells
- Faithfulness fix: uses `reshape` instead of `view` to handle non-contiguous tensors
- Results auto-saved to `MyDrive/vision-datasets/xai_results/` at the end
- `OcclusionSensitivity` uses batched forward passes (batch=32, stride=32) for ~110× speedup over naive per-position loop

---

### 9. Notebook Revision — Syntax, Structure & Optimization (2026-06-02)

A second pass over `notebooks/sample4geo-xai.ipynb` covering style, correctness, and performance.

---

#### 9.1 Pretrained Weights: from zip to direct Drive path

Originally the notebook downloaded a `pretrained.zip` from Drive and extracted it into the repo directory at runtime. This was replaced:

- Weights are now stored already-extracted at `MyDrive/vision-datasets/pretrained/` on Drive.
- `PRETRAINED_DIR` was changed from `f'{REPO_DIR}/pretrained'` → `f'{DRIVE_ROOT}/pretrained'`.
- The entire zip extraction cell was removed; only a checkpoint existence check remains.

**Why:** avoids re-extracting on every Colab session and keeps the repo directory clean.

---

#### 9.2 Code Style Pass

All code cells were updated for consistency:

- **Comments:** lowercase throughout; decorative separators (`# ── text ──────`) stripped down to plain `# text`.
- **Print strings:** lowercase throughout; f-string variable parts and path strings left untouched.
- **Informative prints added** at the end of key cells: checkpoint verification, symlink setup, GPU check, model loading, feature extraction, pair selection, and final save.
- **Cell outputs cleared** — notebook committed clean (no stale execution outputs).

---

#### 9.3 Intro Cell Updated

- Added course code: **BBM416**
- Added group members: Halis Yücel (VIGOR) and Kuzey Ersoy (University-1652)
- Added public Drive link for datasets & weights:  
  https://drive.google.com/drive/folders/1m85pmQhE_iMRUbc173Z81Gs3ID5-wD2C?usp=sharing

---

#### 9.4 Occlusion Sensitivity — Batched Rewrite (major speedup)

The original `compute_sensitivity` loop did one forward pass per patch position — 441 passes per image with `patch=64, stride=16` on a 384×384 input.

**Bottleneck analysis:**
- Positions per image: `((384-64)/16 + 1)² = 441`
- Per pair (query + gallery): 882 forward passes
- 8 U1652 pairs + VIGOR pairs + faithfulness steps → easily thousands of forward passes total
- On A100 this was still taking many minutes per dataset

**Changes made:**

| Change | Effect |
|---|---|
| Batch patch positions (batch_size=32) | ~110× fewer kernel launches |
| `stride` 16 → 32 | 441 → 121 positions per image |
| `torch.no_grad()` → `torch.inference_mode()` | small additional speedup |
| `STRIDE = 32` in parameter cell | consistent with new default |

New position count: `((384-64)/32 + 1)² = 121` → with batch=32: **4 forward passes per image** (was 441).

**Heatmap quality:** slightly coarser grid but fully adequate for presentation and qualitative analysis.

**Attempted during this session (abandoned):**

Tried to use `jupyter-mcp-server` (datalayer) to edit the notebook interactively instead of via raw JSON. Ran into two issues:
1. MCP server v1.0.0 introduced a breaking change requiring `MCP_TOKEN` in the client config — this was missing from `~/.config/opencode/opencode.json`. Added `"MCP_TOKEN": ""` to the environment block.
2. Even after connecting, `read_notebook` / `read_cell` returned 404 because those tools use the `/api/collaboration/session/` endpoint which requires the `jupyter-collaboration` extension to be active. The extension was installed (`jupyter-collaboration==4.0.2`) but the endpoint was not responding correctly with the running server config.

#### 9.5 Dataset Local Copy — Attempted and Reverted

After the first Colab run, feature extraction took **2h 9m** for U1652 on an A100 — entirely due to Drive I/O latency when reading 37,855 small files (~26s/batch). GPU utilization was near zero.

First attempt: add `shutil.copytree` cells to Section 2 to copy datasets to `/content/` before eval.

**Problem:** U1652 test ≈ 18GB, VIGOR ≈ 96GB. Copying from Drive to local via Colab's Fuse mount is just as slow as reading — both go through the same Drive API layer. Zipping first was also ruled out (too large to zip on user's machine, and zipping on Colab reads from Drive which hits the same bottleneck).

**Resolution:** Reverted entirely. Eval is a one-time operation per session — run overnight and move on. See 9.8 for the adopted optimization instead.

---

#### 9.6 `trainer.py` — autocast Fix

`from torch.cuda.amp import autocast` → `from torch.amp import autocast`  
`with autocast():` → `with autocast('cuda'):`  
Removes FutureWarning on PyTorch ≥ 2.4.

---

#### 9.7 `trainer.py` — inference_mode

`torch.no_grad()` → `torch.inference_mode()` in `predict()`.  
`inference_mode` disables autograd more aggressively, slightly faster for pure inference.

---

#### 9.9 Feature & Metric Caching (Drive persistence across sessions)

**Problem:** Colab sessions are ephemeral — closing the tab loses all computed state. Feature extraction takes 2h+ per dataset; running it on every session is not viable.

**Solution:** cache everything to `MyDrive/vision-datasets/xai_cache/` as `.pt` / `.json` files.

Cache files created:

| File | Contents | Size (est.) |
|---|---|---|
| `u1652_features.pt` | q_feats, g_feats, q_ids, g_ids (float32) | ~160MB |
| `u1652_metrics.json` | r1, result_str | bytes |
| `vigor_same_features.pt` | ref_feats, ref_labels, q_feats, q_labels | ~520MB |
| `vigor_same_metrics.json` | r1, result_str | bytes |
| `vigor_cross_features.pt` | same as above | ~520MB |
| `vigor_cross_metrics.json` | r1, result_str | bytes |

**Total cache size:** ~1.2GB (embedding vectors, not images — 96GB dataset compressed to ~1.2GB of features).

**Session flow after first run:**
- features: `.pt` → `torch.load` → seconds
- metrics: `.json` → `json.load` → instant
- `evaluate()`, `predict()`, model reload for eval: **entirely skipped**
- Only model load that remains: for GradCAM/Occlusion (model needed for inference)

**Code changes:**
- `evaluate/university.py`: added `precomputed` parameter — skips internal `predict()` when features are passed in
- `evaluate/vigor.py`: same
- Notebook cell 13: `predict()` wrapped with cache check; stores as `q_feats_u / g_feats_u / q_ids_u / g_ids_u`
- Notebook cell 14: passes `precomputed=` to `evaluate()`, wraps metrics in cache check
- Notebook cell 16 (`run_vigor_eval`): full cache support for both features and metrics
- Notebook cell 28: removed model reload + predict entirely; aliases `q_feats_u` directly
- Notebook cell 39 (VIGOR XAI): tries `precomputed_same/cross` from memory first, falls back to Drive cache, raises if neither exists

---

#### 9.10 Notebook review fixes

Full notebook review after all changes revealed:
- **Cell 35 markdown** was showing old forward-pass count (576 with stride=16); updated to "121 positions → 4 batched forward passes" (stride=32, batch=32)
- **VIGOR occlusion failed pairs cell was missing** entirely from Section 11; added after the successful-pairs cell
- **`cfg_tmp` in `run_vigor_eval`** was a redundant alias of `cfg`; removed

Instead of copying data, optimized the DataLoader pipeline to better overlap Drive I/O and GPU compute:

| Parameter | Before | After | Reason |
|---|---|---|---|
| `batch_size` | 128 | 256 | A100 40GB handles ConvNeXt-Base@384×384 at batch 256 easily; halves batch overhead |
| `num_workers` | 4 | 16 | More parallel Drive reads |
| `prefetch_factor` | 2 (default) | 4 | Workers pre-fetch more batches ahead of GPU |
| `persistent_workers` | False | True | Workers stay alive between batches, no respawn overhead |

Also applied to VIGOR DataLoaders (which were previously hardcoded at `batch_size=128, num_workers=4` instead of reading from `VigorConfig`).

**Realistic expectation:** Drive I/O remains the fundamental ceiling, but better pipelining should bring U1652 feature extraction from ~2h9m down to ~30-50 min.

#### 9.11 XAI Checkpoint — Drive-backed results with skip logic

**Problem:** XAI loops (GradCAM, Occlusion) take time per pair. If Colab session dies mid-loop, all computed heatmaps are lost (were saved to `/content/` local disk).

**Solution:**
1. `RESULTS_DIR` moved from `f'{REPO_DIR}/xai_results'` → `f'{DRIVE_ROOT}/xai_results'`. Every `plt.savefig()` call now writes directly to Drive in real time — no end-of-session copy needed.

2. Skip logic added to all 8 XAI loop cells (U1652 GradCAM successful/failed, U1652 Occlusion successful/failed, VIGOR GradCAM successful/failed, VIGOR Occlusion successful/failed):
```python
if os.path.exists(save_path):
    print(f'skipping #{i} — already saved.')
    continue
```
On resume, already-saved pairs are detected and skipped; only remaining pairs are computed.

3. Final "save to Drive" cell simplified — since `RESULTS_DIR` is already on Drive, the old `shutil.copytree` was removed; cell now just verifies the output directory and lists saved files.

**Combined with the feature cache (9.9), a resumed session now:**
- Loads features + metrics in seconds
- Skips already-generated heatmaps
- Only computes what is genuinely missing

1. **Run notebook on Colab**
   - ~~Section 3: University-1652 baseline~~ ✅ R@1=92.66%
   - Section 4: VIGOR same-area and cross-area baselines
   - Sections 6–11: full XAI pipeline for both datasets

2. **Debug path lookup for U1652DatasetEval**
   - Section 6 has a `TODO` to check the exact attribute name for image paths (`.samples`, `.img_paths`, etc.)

3. **Debug path lookup for VigorDatasetEval**
   - Section 9/10 uses `idx2ground_path` and `idx2sat_path` — verify these exist

4. **Analysis** (Section 12)
   - Compare GradCAM vs Occlusion heatmaps
   - Compare successful vs failed match attention patterns
   - Compare University-1652 (drone↔satellite) vs VIGOR (street↔satellite)
   - Quantitative faithfulness scores

5. **Reports**
   - Progress report
   - Final report
   - Presentation (video)
