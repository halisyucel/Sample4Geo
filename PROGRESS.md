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
   - ~~Section 4: VIGOR same-area baseline~~ ✅ R@1=77.86%
   - ~~Section 4: VIGOR cross-area baseline~~ ✅ R@1=61.71%
   - ~~Section 6: U1652 pair selection~~ ✅ 5 successful + 3 failed
   - Sections 7–11: XAI pipeline (GradCAM + Occlusion for both datasets)

2. ~~Debug path lookup for U1652DatasetEval~~ ✅ `.images` (see Section 12)

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

---

### 12. Baseline Evaluation — VIGOR Cross-Area

**Results (Colab A100, 2026-06-05):**

| Metric | Result | Paper |
|---|---|---|
| Recall@1 | 61.71% | 61.09% |
| Recall@5 | 83.51% | — |
| Recall@10 | 87.99% | — |
| Recall@top1% | 99.39% | — |
| Hit_Rate | 77.64% | — |

R@1 slightly above the paper value (61.71% vs 61.09%). Features loaded from Drive cache (`vigor_cross_features.pt`), metrics computed in ~15 seconds.

---

### 13. U1652 Pair Selection

5 successful and 3 failed query-gallery pairs selected from the test set similarity matrix for XAI visualization.

- Successful: all have sim ≥ 0.85, same ID matched correctly
- Failed: true IDs 0010 and 0013, top-1 retrieval returned wrong gallery (IDs -001 and 0113)

---

### 14. Bug Fix — U1652 Path Lookup (2026-06-05)

Cell 31 was using `query_dataset.samples` to get image paths. `U1652DatasetEval` does not have a `.samples` attribute — paths are stored in `.images` (a flat `list[str]`, one path per image, built in `__init__` at `dataset/university.py:202`).

Fixed: `q_paths = query_dataset.images`, `g_paths = gallery_dataset.images`. `get_path(paths, idx)` simplified to a direct index lookup.

**Results (Colab A100, 2026-06-05):**

| Metric | Result | Paper |
|---|---|---|
| Recall@1 | 77.86% | 77.86% |
| Recall@5 | 95.68% | — |
| Recall@10 | 97.22% | — |
| Recall@top1% | 99.61% | — |
| Hit_Rate | 89.83% | — |

R@1 matches the paper exactly (pretrained weight filename `weights_e40_0.7786.pth`). Feature extraction on A100: ref loader 354 batches × ~11s/batch ≈ 1h05m; query loader 206 batches × ~15s/batch ≈ 51m. Total ~1h57m. Features saved to Drive cache (`vigor_same_features.pt`).

---

### 11. Bug Fixes — `evaluate/vigor.py` (2026-06-05)

Three bugs found and fixed during VIGOR same-area eval run.

#### 11.1 `calculate_scores` — Q×R similarity matrix OOM

The original implementation built the full Q×R similarity matrix in CPU RAM before iterating:

```python
similarity = []
for i in range(steps):
    sim_tmp = query_features[start:end] @ reference_features.T
    similarity.append(sim_tmp.cpu())
similarity = torch.cat(similarity, dim=0)  # full Q×R in RAM
```

For VIGOR same-area: Q=52,605, R=90,618 → **~19 GB** CPU RAM. For cross-area: ~**38 GB**. This caused a ~40-minute hang (Colab paging to swap) with no visible progress.

**Fix:** streaming approach — process each batch, accumulate results immediately, delete batch:

```python
for s in range(steps):
    sim_batch = (query_features[start:end] @ reference_features.T).cpu()
    # accumulate results per query
    del sim_batch
```

Peak RAM during score computation now: one batch × R ≈ `step_size × R × 4 bytes` = 1000 × 90k × 4 ≈ **360 MB**.

#### 11.2 `evaluate()` — missing `result_str` in return value

`evaluate()` returned only `r1` (a `numpy.float64`), but notebook's `run_vigor_eval` expected `(r1, result_str)`. Caused `TypeError: cannot unpack non-iterable numpy.float64 object` after 2 hours of feature extraction.

Fixed: `calculate_scores` now returns `(results[0], ' - '.join(string))`, and `evaluate()` propagates the tuple.

#### 11.3 Notebook cell 16 — stale module import cache

After `git pull` in a live Colab session, Python's `sys.modules` retained the old `sample4geo.evaluate.vigor` module. Re-running the cell re-executed `from sample4geo.evaluate.vigor import evaluate` but the cached module was served, not the updated file.

Fixed: cell 16 now starts with:
```python
import importlib, sys
for _mod in list(sys.modules):
    if _mod.startswith('sample4geo'):
        del sys.modules[_mod]
```
This forces a clean re-import of all `sample4geo.*` submodules every time the cell runs.

#### 11.4 Notebook cell 16 — `IndentationError`

A previous cleanup pass (removing `cfg_tmp`) left an orphan `'        '` line (8 spaces, no newline) in the cell source JSON. Python tokenizer concatenated it with the next line, producing 16-space indentation inside an `else:` block.

Fixed: removed the orphan line from the notebook JSON directly.

---

### 15. Bug Fix — VIGOR Pair Selection `ValueError` (2026-06-07)

Cell 40 (`# pair selection for vigor`) crashed with:

```
ValueError: The truth value of an array with more than one element is ambiguous. Use a.any() or a.all()
```

**Root cause:** For VIGOR, `VigorDatasetEval` stores query labels as a 2D array of shape `(N, 4)` — one row per query, four columns: `[sat, sat_np1, sat_np2, sat_np3]` (primary positive + 3 near-positives). So `q_ids_v_np[i]` returns a `(4,)` array, not a scalar. The original code did `true_id = q_ids_v_np[i]` and then `if top1_id == true_id` — comparing a scalar to a length-4 array triggers the ambiguous truth value error.

**Fix:**

```python
# before
true_id  = q_ids_v_np[i]
if top1_id == true_id and len(vigor_successful) < 5:
    ...
elif top1_id != true_id and len(vigor_failed) < 3:
    correct_idxs = np.where(g_ids_v_np == true_id)[0]

# after
true_ids = q_ids_v_np[i]          # shape (4,): [sat, sat_np1, sat_np2, sat_np3]
true_id  = int(true_ids[0])       # primary positive satellite index
if top1_id in true_ids and len(vigor_successful) < 5:
    ...
elif top1_id not in true_ids and len(vigor_failed) < 3:
    correct_idxs = np.where(g_ids_v_np == true_id)[0]
```

`in` / `not in` checks against the full 4-element positive set, consistent with how VIGOR evaluation counts a hit. `true_id = int(true_ids[0])` extracts the primary positive as a scalar for the `np.where` lookup used in failed pairs.

---

### 16. Notebook Section 13 — Display Saved Results (2026-06-07)

XAI figures are saved directly to Drive as they are generated, but skipped pairs produce no inline output in the notebook — reopening the notebook after a completed run showed no visuals at all.

**Fix:** Added Section 13 (`## 13. Display Saved Results`) at the end of the notebook with a single code cell that reads all saved PNGs from Drive and renders them inline via `matplotlib.image.imread` + `plt.imshow`.

`display_folder(title, folder, filter_prefix)` filters by filename prefix so figures are shown in logical order:

| Call | Prefix | Folder |
|---|---|---|
| GradCAM — U1652 successful | `u1652_successful` | `gradcam/` |
| GradCAM — U1652 failed | `u1652_failed` | `gradcam/` |
| GradCAM — VIGOR same successful | `vigor_same_successful` | `gradcam/` |
| GradCAM — VIGOR same failed | `vigor_same_failed` | `gradcam/` |
| Occlusion — U1652 successful | `u1652_successful` | `occlusion/` |
| Occlusion — U1652 failed | `u1652_failed` | `occlusion/` |
| Occlusion — VIGOR same successful | `vigor_same_successful` | `occlusion/` |
| Occlusion — VIGOR same failed | `vigor_same_failed` | `occlusion/` |

Each figure is rendered at `20×10` inches. The section is purely read-only — no model or feature state required, can be run independently at any time.

---

### 17. XAI Output Review — Visual Observations (2026-06-12)

Full review of all 32 generated XAI figures after the complete pipeline run.

---

#### 17.1 GradCAM — General Observations

GradCAM results are visually compelling and semantically coherent across both datasets.

**University-1652 (drone ↔ satellite):**
- Successful pairs: the drone query consistently attends to building edges and rooftop transitions; the satellite gallery attends to the same building footprints. The correspondence is clear and meaningful.
- Failed pairs: the gallery (wrong satellite) shows diffuse, uniform activation spread across the entire image — the model cannot find a dominant discriminative region, which manifests as scattered high activation. This "confused" activation pattern is a useful XAI observation.

**VIGOR (street panorama ↔ satellite):**
- Successful pairs: street queries focus on landmark structures in the panorama (buildings, tree-lines); satellite gallery focuses on the corresponding road junction or structural boundary. Coherent cross-view attention.
- Failed pairs: the classic failure case is a street view under an elevated metro/rail structure — the model attends to the overhead steel structure, a feature that is not well-visible or well-matched in the satellite view. This illustrates why cross-view localization fails in structurally ambiguous scenes.

---

#### 17.2 Occlusion Sensitivity — Observations and Limitations

**Observations:**
- Important regions identified by occlusion broadly agree with GradCAM (same structural features highlighted), which cross-validates both methods.
- Faithfulness curves for **successful pairs** are mostly monotonically decreasing — masking the most important pixels progressively reduces cosine similarity, validating that the heatmap correctly identifies discriminative regions.
- Faithfulness curves for **failed pairs** show **non-monotonic behaviour** (especially in the gallery curve): similarity can temporarily *increase* when some pixels are masked. Interpretation: in failed matches, the model attends to misleading features that are artificially boosting similarity to the wrong target; removing those regions briefly improves the match quality before full masking degrades it. This is an interesting negative-explanation signal.

**Known limitation — coarse resolution:**
- Parameters: `patch_size=64, stride=32`, input `384×384` → `((384−64)/32 + 1)² = 121` positions → 4 batched forward passes per image.
- At this stride, each importance cell covers a 64×64 patch (1/6 of the image width), producing a visibly blocky heatmap. This is a deliberate trade-off between runtime and resolution (naive stride=8 would require ~300 forward passes per image).
- The coarseness is sufficient for qualitative analysis and localising general regions of interest, but should be acknowledged as a limitation in the report.

---

#### 17.3 U1652 Pair Selection — Same-Class Bias

All 5 successful University-1652 pairs belong to `id=0003` (5 different drone images of the same building, all correctly retrieving the same satellite image). This is a result of greedy sequential selection — the code picks the first 5 successful queries it encounters, and since queries are sorted by class ID, the first class fills all 5 slots.

This is **not a bug** and is actually analytically interesting: it demonstrates that the model is robust to viewpoint variation within a single location (5 different drone angles all match correctly with high similarity ≥ 0.86). However, for presentation purposes the diversity is limited — all 5 figures show the same building from slightly different angles. This should be noted in the report/video.

---

#### 17.4 Video / Presentation Preparation — Pending

The following notebook additions are needed to make the notebook presentation-ready for the project video:

1. **Section 5 (XAI Setup)** — add a "What to expect" markdown after the class definitions, explaining what each method should highlight for geo-localization.
2. **Sections 7–8 (U1652 XAI)** — add observation markdown after GradCAM and Occlusion cells summarising what was found.
3. **Sections 10–11 (VIGOR XAI)** — same as above for VIGOR.
4. **Section 12 (Analysis)** — fill in the analysis scaffold with the observations above; update the results summary to use actual VIGOR metrics (done in revision 9.10).
