import torch
import torch.nn.functional as F
import numpy as np
import cv2
from typing import Optional, Tuple, List
import matplotlib.pyplot as plt


class GradCAMExtractor:
    """
    GradCAM for ConvNeXt models in Sample4Geo
    Visualizes which regions the model focuses on for geo-localization
    """
    
    def __init__(self, model, target_layer=None):
        """
        Args:
            model: TimmModel instance
            target_layer: Target layer for GradCAM (default: last conv layer)
        """
        self.model = model
        self.model.eval()
        
        if target_layer is None:
            self.target_layer = self._find_target_layer()
        else:
            self.target_layer = target_layer
        
        self.gradients = None
        self.activations = None
        
        self._register_hooks()
    
    def _find_target_layer(self):
        """Find the last convolutional layer in ConvNeXt"""
        target_layer = None
        
        for name, module in self.model.model.named_modules():
            if 'stages' in name and 'downsample' not in name:
                if hasattr(module, 'conv') or 'conv' in name.lower():
                    target_layer = module
            
            if len(name.split('.')) == 3 and 'stages' in name:
                target_layer = module
        
        if target_layer is None:
            for name, module in reversed(list(self.model.model.named_modules())):
                if isinstance(module, torch.nn.Conv2d):
                    target_layer = module
                    break
        
        return target_layer
    
    def _register_hooks(self):
        """Register forward and backward hooks"""
        
        def forward_hook(module, input, output):
            self.activations = output.detach()
        
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()
        
        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_backward_hook(backward_hook)
    
    def generate_cam(self, 
                     image: torch.Tensor,
                     target_embedding: Optional[torch.Tensor] = None) -> np.ndarray:
        """
        Generate GradCAM for an image
        
        Args:
            image: Input image tensor (1, C, H, W)
            target_embedding: Target embedding to compute similarity (optional)
        
        Returns:
            GradCAM heatmap (H, W)
        """
        self.model.zero_grad()
        
        embedding = self.model(image)
        
        if target_embedding is not None:
            score = F.cosine_similarity(embedding, target_embedding, dim=-1)
        else:
            score = embedding.norm(dim=-1).mean()
        
        score.backward(retain_graph=True)
        
        if self.gradients is None or self.activations is None:
            raise ValueError("Gradients or activations not captured. Check target_layer.")
        
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        cam = cv2.resize(cam, (image.shape[3], image.shape[2]))
        
        return cam
    
    def visualize_cam(self,
                      image: torch.Tensor,
                      cam: np.ndarray,
                      alpha: float = 0.4,
                      colormap: int = cv2.COLORMAP_JET) -> np.ndarray:
        """
        Overlay GradCAM on original image
        
        Args:
            image: Original image tensor (1, C, H, W)
            cam: GradCAM heatmap
            alpha: Overlay transparency
            colormap: OpenCV colormap
        
        Returns:
            Visualization image (H, W, 3)
        """
        img_np = image.squeeze().cpu().numpy().transpose(1, 2, 0)
        img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-8)
        img_np = (img_np * 255).astype(np.uint8)
        
        heatmap = cv2.applyColorMap(np.uint8(255 * cam), colormap)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        
        visualization = cv2.addWeighted(img_np, 1-alpha, heatmap, alpha, 0)
        
        return visualization
    
    def generate_pair_cam(self,
                          query_image: torch.Tensor,
                          gallery_image: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate GradCAM for both query and gallery images
        
        Args:
            query_image: Query image (drone/satellite)
            gallery_image: Gallery image (satellite/drone)
        
        Returns:
            Tuple of (query_cam, gallery_cam)
        """
        with torch.no_grad():
            gallery_embedding = self.model(gallery_image)
        
        query_cam = self.generate_cam(query_image, gallery_embedding)
        
        with torch.no_grad():
            query_embedding = self.model(query_image)
        
        gallery_cam = self.generate_cam(gallery_image, query_embedding)
        
        return query_cam, gallery_cam
    
    def save_visualization(self,
                           image: torch.Tensor,
                           cam: np.ndarray,
                           save_path: str,
                           title: str = "GradCAM"):
        """
        Save GradCAM visualization
        
        Args:
            image: Original image tensor
            cam: GradCAM heatmap
            save_path: Path to save visualization
            title: Plot title
        """
        visualization = self.visualize_cam(image, cam)
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        img_np = image.squeeze().cpu().numpy().transpose(1, 2, 0)
        img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-8)
        
        axes[0].imshow(img_np)
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        axes[1].imshow(cam, cmap='jet')
        axes[1].set_title('GradCAM Heatmap')
        axes[1].axis('off')
        
        axes[2].imshow(visualization)
        axes[2].set_title('Overlay')
        axes[2].axis('off')
        
        plt.suptitle(title)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Saved visualization to: {save_path}")
