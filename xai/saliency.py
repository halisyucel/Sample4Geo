import torch
import torch.nn.functional as F
import numpy as np
import cv2
import matplotlib.pyplot as plt
from typing import Optional, Tuple


class SaliencyMapGenerator:
    """
    Generate saliency maps to understand which pixels affect the model's decision
    Uses gradient-based methods for geo-localization model
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
    
    def generate_vanilla_saliency(self,
                                   image: torch.Tensor,
                                   target_embedding: Optional[torch.Tensor] = None) -> np.ndarray:
        """
        Generate vanilla gradient-based saliency map
        
        Args:
            image: Input image (1, C, H, W)
            target_embedding: Target embedding for similarity
        
        Returns:
            Saliency map (H, W)
        """
        image = image.clone().requires_grad_(True)
        
        embedding = self.model(image)
        
        if target_embedding is not None:
            score = F.cosine_similarity(embedding, target_embedding, dim=-1).mean()
        else:
            score = embedding.norm(dim=-1).mean()
        
        score.backward()
        
        saliency = image.grad.abs().squeeze().cpu().numpy()
        saliency = np.max(saliency, axis=0)
        
        saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-8)
        
        return saliency
    
    def generate_smoothgrad_saliency(self,
                                      image: torch.Tensor,
                                      target_embedding: Optional[torch.Tensor] = None,
                                      num_samples: int = 50,
                                      noise_level: float = 0.1) -> np.ndarray:
        """
        Generate SmoothGrad saliency map for noise reduction
        
        Args:
            image: Input image (1, C, H, W)
            target_embedding: Target embedding
            num_samples: Number of noisy samples
            noise_level: Noise standard deviation
        
        Returns:
            SmoothGrad saliency map (H, W)
        """
        mean = 0
        std = noise_level * (image.max() - image.min())
        
        saliency_sum = np.zeros((image.shape[2], image.shape[3]))
        
        for _ in range(num_samples):
            noise = torch.randn_like(image) * std + mean
            noisy_image = image + noise
            noisy_image = noisy_image.clamp(image.min(), image.max())
            
            saliency = self.generate_vanilla_saliency(noisy_image, target_embedding)
            saliency_sum += saliency
        
        smooth_saliency = saliency_sum / num_samples
        
        return smooth_saliency
    
    def generate_guided_backprop(self,
                                  image: torch.Tensor,
                                  target_embedding: Optional[torch.Tensor] = None) -> np.ndarray:
        """
        Generate Guided Backpropagation saliency map
        
        Args:
            image: Input image
            target_embedding: Target embedding
        
        Returns:
            Guided saliency map
        """
        def guided_relu_hook(module, grad_input, grad_output):
            if isinstance(module, torch.nn.ReLU):
                return (F.relu(grad_input[0]),)
        
        hooks = []
        for module in self.model.modules():
            if isinstance(module, torch.nn.ReLU):
                hook = module.register_backward_hook(guided_relu_hook)
                hooks.append(hook)
        
        saliency = self.generate_vanilla_saliency(image, target_embedding)
        
        for hook in hooks:
            hook.remove()
        
        return saliency
    
    def visualize_saliency(self,
                           image: torch.Tensor,
                           saliency: np.ndarray,
                           method: str = "Vanilla",
                           save_path: Optional[str] = None):
        """
        Visualize saliency map
        
        Args:
            image: Original image (1, C, H, W)
            saliency: Saliency map (H, W)
            method: Method name for title
            save_path: Path to save
        """
        img_np = image.squeeze().cpu().numpy().transpose(1, 2, 0)
        img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-8)
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        axes[0].imshow(img_np)
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        axes[1].imshow(saliency, cmap='hot')
        axes[1].set_title(f'{method} Saliency Map')
        axes[1].axis('off')
        
        axes[2].imshow(img_np)
        axes[2].imshow(saliency, cmap='hot', alpha=0.5)
        axes[2].set_title('Overlay')
        axes[2].axis('off')
        
        plt.suptitle(f'{method} Saliency Analysis', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved saliency visualization to: {save_path}")
        
        plt.show()
        plt.close()
    
    def compare_methods(self,
                        image: torch.Tensor,
                        target_embedding: Optional[torch.Tensor] = None,
                        save_path: Optional[str] = None):
        """
        Compare different saliency methods
        
        Args:
            image: Input image
            target_embedding: Target embedding
            save_path: Path to save comparison
        """
        vanilla = self.generate_vanilla_saliency(image, target_embedding)
        smoothgrad = self.generate_smoothgrad_saliency(image, target_embedding, num_samples=20)
        
        img_np = image.squeeze().cpu().numpy().transpose(1, 2, 0)
        img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-8)
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        axes[0, 0].imshow(img_np)
        axes[0, 0].set_title('Original Image')
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(vanilla, cmap='hot')
        axes[0, 1].set_title('Vanilla Gradient')
        axes[0, 1].axis('off')
        
        axes[0, 2].imshow(img_np)
        axes[0, 2].imshow(vanilla, cmap='hot', alpha=0.5)
        axes[0, 2].set_title('Vanilla Overlay')
        axes[0, 2].axis('off')
        
        axes[1, 0].imshow(img_np)
        axes[1, 0].set_title('Original Image')
        axes[1, 0].axis('off')
        
        axes[1, 1].imshow(smoothgrad, cmap='hot')
        axes[1, 1].set_title('SmoothGrad')
        axes[1, 1].axis('off')
        
        axes[1, 2].imshow(img_np)
        axes[1, 2].imshow(smoothgrad, cmap='hot', alpha=0.5)
        axes[1, 2].set_title('SmoothGrad Overlay')
        axes[1, 2].axis('off')
        
        plt.suptitle('Saliency Methods Comparison', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved comparison to: {save_path}")
        
        plt.show()
        plt.close()
