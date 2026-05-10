# XAI (Explainable AI) Module for Sample4Geo

This module provides various XAI methods to understand and interpret the Sample4Geo model's decisions.

## 🎯 Purpose

For cross-view geo-localization models:
- Visualize which regions the model focuses on
- Understand query-gallery matching decisions
- Visualize learned feature representations
- Analyze model reliability

## 📦 Contents

### 1. GradCAM (`gradcam.py`)
Visualizes regions the model focuses on for matching.

```python
from xai import GradCAMExtractor

gradcam = GradCAMExtractor(model)
query_cam = gradcam.generate_cam(query_image, gallery_embedding)
gradcam.save_visualization(query_image, query_cam, 'output.png')
```

**Features:**
- GradCAM for query-gallery pairs
- Overlay visualization
- Automatic target layer detection (for ConvNeXt)

### 2. Similarity Visualization (`similarity_viz.py`)
Visualizes similarity between query and gallery.

```python
from xai import SimilarityVisualizer

viz = SimilarityVisualizer(model)
viz.visualize_top_k(query_image, gallery_images, similarity_scores, top_k=5)
```

**Features:**
- Show top-k similar images
- Similarity matrix heatmap
- Embedding space visualization (t-SNE/PCA)

### 3. Saliency Maps (`saliency.py`)
Generates pixel-level importance maps.

```python
from xai import SaliencyMapGenerator

saliency_gen = SaliencyMapGenerator(model)
saliency = saliency_gen.generate_vanilla_saliency(image)
smooth_saliency = saliency_gen.generate_smoothgrad_saliency(image, num_samples=50)
```

**Features:**
- Vanilla Gradient Saliency
- SmoothGrad (noise-robust)
- Guided Backpropagation
- Method comparison

### 4. Integrated Gradients (`integrated_gradients.py`)
Performs feature importance analysis.

```python
from xai import IntegratedGradientsExtractor

ig = IntegratedGradientsExtractor(model)
attributions = ig.compute_integrated_gradients(image, target_embedding, n_steps=50)
```

**Features:**
- Integrated Gradients
- SmoothGrad IG
- Query-gallery pair analysis

### 5. Feature Visualization (`feature_viz.py`)
Visualizes learned feature representations.

```python
from xai import FeatureVisualizer

viz = FeatureVisualizer(model)
features, labels = viz.extract_features(dataloader)
viz.visualize_feature_distribution(features, labels, method='tsne')
```

**Features:**
- t-SNE/PCA projection
- Feature statistics
- Intra-class variation analysis
- Feature correlation matrix

## 🚀 Usage

### Installation

```bash
pip install captum matplotlib seaborn scikit-learn
```

### Quick Start

```python
from sample4geo.model import TimmModel
from xai import GradCAMExtractor, SimilarityVisualizer

# Load model
model = TimmModel('convnext_base.fb_in22k_ft_in1k_384', pretrained=True)
model.load_state_dict(torch.load('pretrained/weights.pth'))
model.eval()

# GradCAM
gradcam = GradCAMExtractor(model)
cam = gradcam.generate_cam(query_image, gallery_embedding)

# Similarity visualization
viz = SimilarityVisualizer(model)
viz.visualize_top_k(query_img, gallery_imgs, scores)
```

### Run Example Script

```bash
cd examples
python run_xai_examples.py
```

## 📊 Outputs

Each XAI method generates different output files:

```
xai_results/
├── gradcam/
│   ├── query_gradcam.png       
│   └── gallery_gradcam.png     
├── similarity/
│   └── top_k_similarity.png    
├── saliency/
│   └── saliency_comparison.png 
├── integrated_gradients/
│   └── integrated_gradients_pair.png
└── features/
    ├── feature_tsne.png
    ├── feature_statistics.png
    └── intra_class_variation.png
```

## 🔍 Method Details

### GradCAM
- **Purpose:** Show which regions the model focuses on
- **Usage:** Query-gallery matching analysis
- **Interpretation:** Red regions = high attention

### Saliency Maps
- **Purpose:** Pixel-level importance scores
- **Usage:** Which pixels drive the decision
- **Interpretation:** Bright regions = important pixels

### Integrated Gradients
- **Purpose:** Feature importance via path integration
- **Usage:** Robust feature importance
- **Interpretation:** Positive = contributing, Negative = reducing

### Feature Visualization
- **Purpose:** Understand learned embedding space
- **Usage:** Cluster analysis, class separation
- **Interpretation:** Same color = similar features

## 📝 Notes

1. **Model type:** Optimized for ConvNeXt
2. **GPU:** CUDA supported (auto-detected)
3. **Memory:** Be careful with large batches
4. **Dataset:** Designed for University-1652
