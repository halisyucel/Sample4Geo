import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from captum.attr import IntegratedGradients, NoiseTunnel
from captum.attr import visualization as viz
from typing import Optional, Tuple
import cv2


class IntegratedGradientsExtractor:
    """
    Integrated Gradients for understanding feature importance
    using Captum library
    """
    
    def __init__(self, model, device='cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Args:
            model: Trained TimmModel instance
            device: Device to run on
        """
        self.model = model
        self.model.eval()
        self.device = device
        self.ig = IntegratedGradients(model)
    
    def compute_integrated_gradients(self,
                                      image: torch.Tensor,
                                      target_embedding: Optional[torch.Tensor] = None,
                                      n_steps: int = 50) -> torch.Tensor:
        """
        Compute Integrated Gradients
        
        Args:
            image: Input image (1, C, H, W)
            target_embedding: Target embedding for similarity
            n_steps: Number of integration steps
        
        Returns:
            Integrated gradients (1, C, H, W)
        """
        def forward_func(img):
            embedding = self.model(img)
            if target_embedding is not None:
                return F.cosine_similarity(embedding, target_embedding, dim=-1)
            else:
                return embedding.norm(dim=-1)
        
        attributions = self.ig.attribute(image,
                                         n_steps=n_steps,
                                         return_convergence_delta=False)
        
        return attributions
    
    def compute_smoothgrad_ig(self,
                              image: torch.Tensor,
                              target_embedding: Optional[torch.Tensor] = None,
                              n_steps: int = 50,
                              nt_samples: int = 10) -> torch.Tensor:
        """
        Compute Integrated Gradients with SmoothGrad
        
        Args:
            image: Input image
            target_embedding: Target embedding
            n_steps: Integration steps
            nt_samples: Number of SmoothGrad samples
        
        Returns:
            SmoothGrad IG attributions
        """
        def forward_func(img):
            embedding = self.model(img)
            if target_embedding is not None:
                return F.cosine_similarity(embedding, target_embedding, dim=-1)
            else:
                return embedding.norm(dim=-1)
        
        noise_tunnel = NoiseTunnel(self.ig)
        
        attributions = noise_tunnel.attribute(image,
                                               n_samples=nt_samples,
                                               nt_type='smoothgrad',
                                               stdevs=0.1,
                                               n_steps=n_steps,
                                               return_convergence_delta=False)
        
        return attributions
    
    def visualize_attributions(self,
                                image: torch.Tensor,
                                attributions: torch.Tensor,
                                method: str = "Integrated Gradients",
                                save_path: Optional[str] = None):
        """
        Visualize attributions using Captum's visualization
        
        Args:
            image: Original image (1, C, H, W)
            attributions: Attribution map (1, C, H, W)
            method: Method name
            save_path: Path to save
        """
        img_np = image.squeeze().cpu().numpy().transpose(1, 2, 0)
        img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-8)
        
        attr_np = attributions.squeeze().cpu().numpy().transpose(1, 2, 0)
        attr_np = np.mean(attr_np, axis=2)
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        axes[0].imshow(img_np)
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        im = axes[1].imshow(attr_np, cmap='seismic', vmin=-attr_np.max(), vmax=attr_np.max())
        axes[1].set_title(f'{method} Attribution')
        axes[1].axis('off')
        plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
        
        axes[2].imshow(img_np)
        im2 = axes[2].imshow(attr_np, cmap='seismic', alpha=0.5, vmin=-attr_np.max(), vmax=attr_np.max())
        axes[2].set_title('Overlay')
        axes[2].axis('off')
        
        plt.suptitle(f'{method} Analysis', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved IG visualization to: {save_path}")
        
        plt.show()
        plt.close()
    
    def analyze_pair(self,
                     query_image: torch.Tensor,
                     gallery_image: torch.Tensor,
                     save_path: Optional[str] = None) -> dict:
        """
        Analyze query-gallery pair with Integrated Gradients
        
        Args:
            query_image: Query image (1, C, H, W)
            gallery_image: Gallery image (1, C, H, W)
            save_path: Path to save results
        
        Returns:
            Dictionary with analysis results
        """
        with torch.no_grad():
            gallery_embedding = self.model(gallery_image)
            query_embedding = self.model(query_image)
            similarity = F.cosine_similarity(query_embedding, gallery_embedding, dim=-1).item()
        
        query_ig = self.compute_integrated_gradients(query_image, gallery_embedding)
        gallery_ig = self.compute_integrated_gradients(gallery_image, query_embedding)
        
        results = {
            'similarity': similarity,
            'query_attributions': query_ig,
            'gallery_attributions': gallery_ig
        }
        
        if save_path:
            self._visualize_pair_analysis(query_image, gallery_image,
                                          query_ig, gallery_ig,
                                          similarity, save_path)
        
        return results
    
    def _visualize_pair_analysis(self,
                                  query_image: torch.Tensor,
                                  gallery_image: torch.Tensor,
                                  query_ig: torch.Tensor,
                                  gallery_ig: torch.Tensor,
                                  similarity: float,
                                  save_path: str):
        """
        Visualize query-gallery pair analysis
        """
        query_np = query_image.squeeze().cpu().numpy().transpose(1, 2, 0)
        query_np = (query_np - query_np.min()) / (query_np.max() - query_np.min() + 1e-8)
        
        gallery_np = gallery_image.squeeze().cpu().numpy().transpose(1, 2, 0)
        gallery_np = (gallery_np - gallery_np.min()) / (gallery_np.max() - gallery_np.min() + 1e-8)
        
        query_attr = query_ig.squeeze().cpu().numpy().transpose(1, 2, 0)
        query_attr = np.mean(query_attr, axis=2)
        
        gallery_attr = gallery_ig.squeeze().cpu().numpy().transpose(1, 2, 0)
        gallery_attr = np.mean(gallery_attr, axis=2)
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        axes[0, 0].imshow(query_np)
        axes[0, 0].set_title('Query Image')
        axes[0, 0].axis('off')
        
        im1 = axes[0, 1].imshow(query_attr, cmap='seismic', vmin=-abs(query_attr).max(), vmax=abs(query_attr).max())
        axes[0, 1].set_title('Query IG Attribution')
        axes[0, 1].axis('off')
        plt.colorbar(im1, ax=axes[0, 1], fraction=0.046)
        
        axes[0, 2].imshow(query_np)
        axes[0, 2].imshow(query_attr, cmap='seismic', alpha=0.5, vmin=-abs(query_attr).max(), vmax=abs(query_attr).max())
        axes[0, 2].set_title('Query Overlay')
        axes[0, 2].axis('off')
        
        axes[1, 0].imshow(gallery_np)
        axes[1, 0].set_title('Gallery Image')
        axes[1, 0].axis('off')
        
        im2 = axes[1, 1].imshow(gallery_attr, cmap='seismic', vmin=-abs(gallery_attr).max(), vmax=abs(gallery_attr).max())
        axes[1, 1].set_title('Gallery IG Attribution')
        axes[1, 1].axis('off')
        plt.colorbar(im2, ax=axes[1, 1], fraction=0.046)
        
        axes[1, 2].imshow(gallery_np)
        axes[1, 2].imshow(gallery_attr, cmap='seismic', alpha=0.5, vmin=-abs(gallery_attr).max(), vmax=abs(gallery_attr).max())
        axes[1, 2].set_title('Gallery Overlay')
        axes[1, 2].axis('off')
        
        plt.suptitle(f'Query-Gallery Pair Analysis\nSimilarity: {similarity:.4f}',
                     fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved pair analysis to: {save_path}")
        plt.show()
        plt.close()
