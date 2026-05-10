# XAI Module for Sample4Geo

Two XAI techniques for cross-view geo-localization analysis.

## Methods

### GradCAM (`gradcam.py`)
Gradient-based attention map. Shows which regions activate the network for a given query-gallery pair.

```python
from xai import GradCAMExtractor

gradcam = GradCAMExtractor(model)
query_cam, gallery_cam = gradcam.generate_pair_cam(query_image, gallery_image)
gradcam.save_visualization(query_image, query_cam, 'output.png', title='Query GradCAM')
```

### Occlusion Sensitivity (`occlusion_sensitivity.py`)
Perturbation-based importance map. Slides an occlusion patch over the image and measures cosine similarity drop at each position. Includes a faithfulness metric.

```python
from xai import OcclusionSensitivity

occ = OcclusionSensitivity(model)

# Single query against a gallery embedding
with torch.no_grad():
    gallery_emb = model(gallery_image)
importance_map = occ.compute_sensitivity(query_image, gallery_emb, patch_size=64, stride=16)

# Both directions at once
query_map, gallery_map = occ.compute_pair_sensitivity(query_image, gallery_image)

# Faithfulness curve
fractions, scores = occ.compute_faithfulness(query_image, gallery_emb, importance_map)
occ.save_visualization(query_image, importance_map, 'output.png',
                       faithfulness_data=(fractions, scores))
```

## Comparison

| | GradCAM | Occlusion Sensitivity |
|---|---|---|
| Type | Gradient-based | Perturbation-based |
| Model access | White-box | Black-box |
| Speed | Fast | Slow (O(H×W / stride²)) |
| Faithfulness | Indirect | Direct |

## Output structure

```
xai_results/
├── gradcam/
│   ├── query_gradcam.png
│   └── gallery_gradcam.png
└── occlusion/
    ├── query_occlusion.png      # includes faithfulness curve
    └── gallery_occlusion.png
```
