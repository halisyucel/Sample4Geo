# XAI Module

Two XAI techniques for cross-view geo-localization: **GradCAM** and **Occlusion Sensitivity**.

---

## GradCAM

When the model matches a street photo to a satellite image, which pixels did it focus on? GradCAM answers this.

**How it works:**
1. The image passes through the model, producing an embedding.
2. The cosine similarity score between the two embeddings is computed.
3. The gradient of this score with respect to the last convolutional layer is computed.
4. Gradients × activations → summed → heatmap.

Red regions = covering those regions would have lowered the score.

```python
from xai import GradCAMExtractor

gradcam = GradCAMExtractor(model)

# Generate CAM for both directions at once
query_cam, gallery_cam = gradcam.generate_pair_cam(query_tensor, gallery_tensor)

# Save (original + heatmap + overlay)
gradcam.save_visualization(query_tensor, query_cam, 'query_gradcam.png', title='Query')
gradcam.save_visualization(gallery_tensor, gallery_cam, 'gallery_gradcam.png', title='Gallery')
```

---

## Occlusion Sensitivity

Cover different regions of the image with a black patch one at a time, and measure how much the similarity score drops. A large drop means that region was important.

**How it works:**
1. Baseline score: cosine similarity between original query and gallery embeddings.
2. A 64×64 patch slides across the entire image with a 16px stride.
3. At each position, the patch is applied and a new score is computed.
4. `baseline_score - new_score` = importance of that region.
5. Overlapping positions are averaged → importance map.

```python
from xai import OcclusionSensitivity

occ = OcclusionSensitivity(model)

# Compute sensitivity maps for both directions at once
query_map, gallery_map = occ.compute_pair_sensitivity(
    query_tensor, gallery_tensor,
    patch_size=64,  # occlusion patch size in pixels
    stride=16,      # step size — smaller = more precise but slower
)

# Faithfulness curve: progressively mask the most important pixels, measure score
with torch.no_grad():
    gallery_emb = model(gallery_tensor)
fractions, scores = occ.compute_faithfulness(query_tensor, gallery_emb, query_map, steps=10)

# Save (original + importance map + overlay + faithfulness curve)
occ.save_visualization(
    query_tensor, query_map, 'query_occlusion.png',
    faithfulness_data=(fractions, scores)
)
```

### What is the Faithfulness Curve?

Measures how trustworthy the XAI explanation is.

The idea: "If I mask the regions you called important, does the model actually break?"

- X axis: percentage of pixels masked (0% → 100%)
- Y axis: cosine similarity score
- Steep drop → XAI correctly identified the important regions (high faithfulness)
- Flat curve → the highlighted regions were not actually critical

---

## Parameters

### `patch_size` and `stride`

| patch_size | stride | Result |
|---|---|---|
| 32 | 8 | Very precise, very slow |
| 64 | 16 | Recommended (good balance) |
| 96 | 32 | Coarse but fast |

With `patch_size=64, stride=16` on a 384×384 image, ~500 forward passes are required.
Without GPU: ~2–3 minutes. With GPU: ~20 seconds.

---

## Output Format

Each `save_visualization` call produces one PNG:

**GradCAM:**
```
[ Original Image ] [ GradCAM Heatmap ] [ Overlay ]
```

**Occlusion Sensitivity:**
```
[ Original Image ] [ Importance Map ] [ Overlay ] [ Faithfulness Curve ]
```

---

## Running via Scripts

Instead of using the module directly, use the ready-made scripts:

```bash
# GradCAM
python examples/run_gradcam.py --query ... --gallery ... --checkpoint ...

# Occlusion Sensitivity
python examples/run_occlusion.py --query ... --gallery ... --checkpoint ...
```

Run with `--help` for all available arguments.
