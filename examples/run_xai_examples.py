"""
Example script demonstrating XAI methods for Sample4Geo
Shows how to use GradCAM, Saliency Maps, Integrated Gradients, and Feature Visualization
"""

import os
import sys
import torch
import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt

sys.path.append('..')
from sample4geo.model import TimmModel
from sample4geo.dataset.university import get_transforms
from xai import (
    GradCAMExtractor,
    SimilarityVisualizer,
    SaliencyMapGenerator,
    IntegratedGradientsExtractor,
    FeatureVisualizer
)


def load_model(checkpoint_path, model_name='convnext_base.fb_in22k_ft_in1k_384', device='cpu'):
    """Load trained model"""
    print(f"\n{'='*60}")
    print("Loading Model...")
    print(f"{'='*60}")
    
    model = TimmModel(model_name, pretrained=True, img_size=384)
    
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"Loading weights from: {checkpoint_path}")
        state_dict = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state_dict, strict=False)
    
    model = model.to(device)
    model.eval()
    
    print("✓ Model loaded successfully!")
    return model


def load_image(image_path, img_size=384, mean=None, std=None):
    """Load and preprocess image"""
    if mean is None:
        mean = [0.485, 0.456, 0.406]
    if std is None:
        std = [0.229, 0.224, 0.225]
    
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    transform = A.Compose([
        A.Resize(img_size, img_size, interpolation=cv2.INTER_LINEAR_EXACT),
        A.Normalize(mean, std),
        ToTensorV2()
    ])
    
    img_tensor = transform(image=img)['image']
    return img_tensor.unsqueeze(0), img


def example_gradcam(model, query_path, gallery_path, output_dir):
    """
    Example 1: GradCAM Visualization
    Shows which regions the model focuses on for matching
    """
    print(f"\n{'='*60}")
    print("Example 1: GradCAM Visualization")
    print(f"{'='*60}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    query_tensor, query_img = load_image(query_path)
    gallery_tensor, gallery_img = load_image(gallery_path)
    
    gradcam = GradCAMExtractor(model)
    
    print("\nGenerating GradCAM for query image...")
    query_cam, gallery_cam = gradcam.generate_pair_cam(query_tensor, gallery_tensor)
    
    query_save_path = os.path.join(output_dir, 'query_gradcam.png')
    gradcam.save_visualization(query_tensor, query_cam, query_save_path, 
                               title="Query Image (Drone)")
    
    gallery_save_path = os.path.join(output_dir, 'gallery_gradcam.png')
    gradcam.save_visualization(gallery_tensor, gallery_cam, gallery_save_path,
                               title="Gallery Image (Satellite)")
    
    print("\n✓ GradCAM visualization completed!")


def example_similarity_visualization(model, query_path, gallery_paths, output_dir):
    """
    Example 2: Similarity Visualization
    Shows top-k similar images and similarity scores
    """
    print(f"\n{'='*60}")
    print("Example 2: Similarity Visualization")
    print(f"{'='*60}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    query_tensor, query_img = load_image(query_path)
    
    gallery_images = []
    gallery_tensors = []
    for path in gallery_paths:
        tensor, img = load_image(path)
        gallery_images.append(img)
        gallery_tensors.append(tensor)
    
    gallery_batch = torch.cat(gallery_tensors, dim=0)
    
    with torch.no_grad():
        query_embedding = model(query_tensor)
        gallery_embeddings = model(gallery_batch)
    
    similarity = torch.nn.functional.cosine_similarity(
        query_embedding.unsqueeze(1),
        gallery_embeddings.unsqueeze(0),
        dim=-1
    ).squeeze().numpy()
    
    visualizer = SimilarityVisualizer(model)
    
    save_path = os.path.join(output_dir, 'top_k_similarity.png')
    visualizer.visualize_top_k(query_img, gallery_images, similarity,
                               top_k=5, save_path=save_path)
    
    print("\n✓ Similarity visualization completed!")


def example_saliency_map(model, image_path, output_dir):
    """
    Example 3: Saliency Map
    Shows pixel-level importance
    """
    print(f"\n{'='*60}")
    print("Example 3: Saliency Map")
    print(f"{'='*60}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    image_tensor, _ = load_image(image_path)
    
    saliency_gen = SaliencyMapGenerator(model)
    
    save_path = os.path.join(output_dir, 'saliency_comparison.png')
    saliency_gen.compare_methods(image_tensor, save_path=save_path)
    
    print("\n✓ Saliency map completed!")


def example_integrated_gradients(model, query_path, gallery_path, output_dir):
    """
    Example 4: Integrated Gradients
    Shows feature importance through path integration
    """
    print(f"\n{'='*60}")
    print("Example 4: Integrated Gradients")
    print(f"{'='*60}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    query_tensor, _ = load_image(query_path)
    gallery_tensor, _ = load_image(gallery_path)
    
    ig_extractor = IntegratedGradientsExtractor(model)
    
    save_path = os.path.join(output_dir, 'integrated_gradients_pair.png')
    results = ig_extractor.analyze_pair(query_tensor, gallery_tensor, save_path=save_path)
    
    print(f"\nSimilarity Score: {results['similarity']:.4f}")
    print("\n✓ Integrated Gradients completed!")


def example_feature_visualization(model, dataloader, output_dir):
    """
    Example 5: Feature Visualization
    Visualizes learned feature space
    """
    print(f"\n{'='*60}")
    print("Example 5: Feature Visualization")
    print(f"{'='*60}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    feature_viz = FeatureVisualizer(model)
    
    print("\nExtracting features...")
    features, labels = feature_viz.extract_features(dataloader, max_samples=500)
    print(f"Extracted {len(features)} features with shape {features.shape}")
    
    save_path_tsne = os.path.join(output_dir, 'feature_tsne.png')
    feature_viz.visualize_feature_distribution(features, labels, method='tsne',
                                               save_path=save_path_tsne)
    
    save_path_stats = os.path.join(output_dir, 'feature_statistics.png')
    feature_viz.analyze_feature_statistics(features, labels, save_path=save_path_stats)
    
    save_path_var = os.path.join(output_dir, 'intra_class_variation.png')
    feature_viz.visualize_intra_class_variation(features, labels, save_path=save_path_var)
    
    print("\n✓ Feature visualization completed!")


def main():
    """Main function to run all XAI examples"""
    
    print("\n" + "="*60)
    print("XAI DEMONSTRATION FOR SAMPLE4GEO")
    print("="*60)
    
    checkpoint_path = 'pretrained/university/convnext_base.fb_in22k_ft_in1k_384/weights_e1_0.9515.pth'
    model = load_model(checkpoint_path)
    
    query_image = 'data/U1652/test/query_drone/0001/image-00.jpg'
    gallery_image = 'data/U1652/test/gallery_satellite/0001/satellite.jpg'
    
    output_base = 'xai_results'
    os.makedirs(output_base, exist_ok=True)
    
    if os.path.exists(query_image) and os.path.exists(gallery_image):
        example_gradcam(model, query_image, gallery_image,
                       os.path.join(output_base, 'gradcam'))
        
        gallery_dir = 'data/U1652/test/gallery_satellite'
        if os.path.exists(gallery_dir):
            gallery_paths = []
            for building_id in os.listdir(gallery_dir)[:10]:
                building_path = os.path.join(gallery_dir, building_id)
                if os.path.isdir(building_path):
                    for img_name in os.listdir(building_path)[:1]:
                        gallery_paths.append(os.path.join(building_path, img_name))
            
            if gallery_paths:
                example_similarity_visualization(model, query_image, gallery_paths,
                                                os.path.join(output_base, 'similarity'))
        
        example_saliency_map(model, query_image,
                            os.path.join(output_base, 'saliency'))
        
        example_integrated_gradients(model, query_image, gallery_image,
                                    os.path.join(output_base, 'integrated_gradients'))
    else:
        print("\n⚠ Image files not found. Please download and convert the dataset first.")
        print("Run: python convert_u1652_dataset.py")
    
    print("\n" + "="*60)
    print("XAI DEMONSTRATION COMPLETED!")
    print("="*60)
    print(f"\nResults saved to: {output_base}/")
    print("\nAvailable XAI methods:")
    print("  1. GradCAM - Visualizes model attention regions")
    print("  2. Similarity Visualization - Shows top-k matches")
    print("  3. Saliency Maps - Pixel-level importance")
    print("  4. Integrated Gradients - Feature importance via gradients")
    print("  5. Feature Visualization - t-SNE/PCA of embeddings")


if __name__ == "__main__":
    main()
