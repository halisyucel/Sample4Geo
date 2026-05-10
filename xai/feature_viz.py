import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, List, Tuple
import cv2
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import seaborn as sns


class FeatureVisualizer:
    """
    Visualize learned features and embeddings
    Understand what the model has learned
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
    
    def extract_features(self, dataloader, max_samples: Optional[int] = None) -> Tuple[torch.Tensor, np.ndarray]:
        """
        Extract features from dataloader
        
        Args:
            dataloader: DataLoader with images
            max_samples: Maximum number of samples to extract
        
        Returns:
            Tuple of (features, labels)
        """
        features_list = []
        labels_list = []
        
        with torch.no_grad():
            for idx, (images, labels) in enumerate(dataloader):
                if max_samples and idx * len(images) >= max_samples:
                    break
                
                images = images.to(self.device)
                features = self.model(images)
                features = F.normalize(features, dim=-1)
                
                features_list.append(features.cpu())
                labels_list.append(labels.numpy())
        
        features = torch.cat(features_list, dim=0)
        labels = np.concatenate(labels_list, axis=0)
        
        return features, labels
    
    def visualize_feature_distribution(self,
                                       features: torch.Tensor,
                                       labels: Optional[np.ndarray] = None,
                                       method: str = 'tsne',
                                       n_components: int = 2,
                                       save_path: Optional[str] = None):
        """
        Visualize feature distribution in 2D
        
        Args:
            features: (N, D) features
            labels: (N,) labels
            method: 'tsne' or 'pca'
            n_components: 2 or 3
            save_path: Path to save
        """
        features_np = features.numpy()
        
        if method == 'tsne':
            reducer = TSNE(n_components=n_components, 
                          random_state=42,
                          perplexity=min(30, len(features_np) - 1))
        else:
            reducer = PCA(n_components=n_components, random_state=42)
        
        features_reduced = reducer.fit_transform(features_np)
        
        if n_components == 2:
            self._plot_2d(features_reduced, labels, method, save_path)
        else:
            self._plot_3d(features_reduced, labels, method, save_path)
    
    def _plot_2d(self, features_2d: np.ndarray, labels: Optional[np.ndarray],
                 method: str, save_path: Optional[str]):
        """Plot 2D feature visualization"""
        plt.figure(figsize=(12, 10))
        
        if labels is not None:
            unique_labels = np.unique(labels)
            n_labels = min(len(unique_labels), 20)
            selected_labels = unique_labels[:n_labels]
            
            colors = plt.cm.tab20(np.linspace(0, 1, n_labels))
            
            for idx, label in enumerate(selected_labels):
                mask = labels == label
                count = mask.sum()
                plt.scatter(features_2d[mask, 0], features_2d[mask, 1],
                           c=[colors[idx]], label=f'ID {label} (n={count})',
                           alpha=0.6, s=50)
            
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        else:
            plt.scatter(features_2d[:, 0], features_2d[:, 1], alpha=0.6, s=50)
        
        plt.xlabel(f'{method.upper()} Dimension 1', fontsize=12)
        plt.ylabel(f'{method.upper()} Dimension 2', fontsize=12)
        plt.title(f'Feature Space Visualization ({method.upper()})', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved feature visualization to: {save_path}")
        
        plt.show()
        plt.close()
    
    def _plot_3d(self, features_3d: np.ndarray, labels: Optional[np.ndarray],
                 method: str, save_path: Optional[str]):
        """Plot 3D feature visualization"""
        from mpl_toolkits.mplot3d import Axes3D
        
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        if labels is not None:
            unique_labels = np.unique(labels)
            n_labels = min(len(unique_labels), 10)
            selected_labels = unique_labels[:n_labels]
            
            colors = plt.cm.tab10(np.linspace(0, 1, n_labels))
            
            for idx, label in enumerate(selected_labels):
                mask = labels == label
                ax.scatter(features_3d[mask, 0], features_3d[mask, 1], features_3d[mask, 2],
                          c=[colors[idx]], label=f'ID {label}', alpha=0.6, s=50)
            
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        else:
            ax.scatter(features_3d[:, 0], features_3d[:, 1], features_3d[:, 2],
                      alpha=0.6, s=50)
        
        ax.set_xlabel(f'{method.upper()} Dim 1', fontsize=10)
        ax.set_ylabel(f'{method.upper()} Dim 2', fontsize=10)
        ax.set_zlabel(f'{method.upper()} Dim 3', fontsize=10)
        ax.set_title(f'3D Feature Space ({method.upper()})', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved 3D visualization to: {save_path}")
        
        plt.show()
        plt.close()
    
    def analyze_feature_statistics(self,
                                   features: torch.Tensor,
                                   labels: Optional[np.ndarray] = None,
                                   save_path: Optional[str] = None):
        """
        Analyze feature statistics
        
        Args:
            features: (N, D) features
            labels: (N,) labels
            save_path: Path to save
        """
        features_np = features.numpy()
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        axes[0, 0].hist(features_np.flatten(), bins=100, alpha=0.7, color='blue')
        axes[0, 0].set_xlabel('Feature Value')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Feature Value Distribution')
        axes[0, 0].grid(alpha=0.3)
        
        feature_means = features_np.mean(axis=1)
        axes[0, 1].hist(feature_means, bins=50, alpha=0.7, color='green')
        axes[0, 1].set_xlabel('Mean Feature Value')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Per-Sample Mean Distribution')
        axes[0, 1].grid(alpha=0.3)
        
        feature_stds = features_np.std(axis=1)
        axes[1, 0].hist(feature_stds, bins=50, alpha=0.7, color='orange')
        axes[1, 0].set_xlabel('Feature STD')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('Per-Sample STD Distribution')
        axes[1, 0].grid(alpha=0.3)
        
        if labels is not None:
            unique_labels = np.unique(labels)
            n_labels = min(len(unique_labels), 10)
            
            for label in unique_labels[:n_labels]:
                mask = labels == label
                label_features = features_np[mask]
                label_mean = label_features.mean()
                label_std = label_features.std()
                axes[1, 1].scatter(label_mean, label_std, label=f'ID {label}', s=100, alpha=0.7)
            
            axes[1, 1].set_xlabel('Mean Feature Value')
            axes[1, 1].set_ylabel('Feature STD')
            axes[1, 1].set_title('Per-Class Statistics')
            axes[1, 1].legend(fontsize=8)
            axes[1, 1].grid(alpha=0.3)
        
        plt.suptitle('Feature Statistics Analysis', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved statistics to: {save_path}")
        
        plt.show()
        plt.close()
    
    def visualize_intra_class_variation(self,
                                        features: torch.Tensor,
                                        labels: np.ndarray,
                                        save_path: Optional[str] = None):
        """
        Visualize intra-class variation
        
        Args:
            features: (N, D) features
            labels: (N,) labels
            save_path: Path to save
        """
        unique_labels = np.unique(labels)
        n_labels = min(len(unique_labels), 10)
        selected_labels = unique_labels[:n_labels]
        
        intra_class_distances = []
        inter_class_distances = []
        
        for label in selected_labels:
            mask = labels == label
            class_features = features[mask]
            
            if len(class_features) > 1:
                distances = torch.cdist(class_features, class_features)
                distances = distances[torch.triu(torch.ones_like(distances), diagonal=1) == 1]
                intra_class_distances.append(distances.mean().item())
        
        for i, label_i in enumerate(selected_labels):
            for label_j in selected_labels[i+1:]:
                mask_i = labels == label_i
                mask_j = labels == label_j
                
                features_i = features[mask_i]
                features_j = features[mask_j]
                
                distances = torch.cdist(features_i, features_j)
                inter_class_distances.append(distances.mean().item())
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        axes[0].hist(intra_class_distances, bins=30, alpha=0.7, label='Intra-class', color='blue')
        axes[0].hist(inter_class_distances, bins=30, alpha=0.7, label='Inter-class', color='red')
        axes[0].set_xlabel('Distance')
        axes[0].set_ylabel('Frequency')
        axes[0].set_title('Intra vs Inter-class Distances')
        axes[0].legend()
        axes[0].grid(alpha=0.3)
        
        axes[1].bar(['Intra-class', 'Inter-class'],
                   [np.mean(intra_class_distances), np.mean(inter_class_distances)],
                   yerr=[np.std(intra_class_distances), np.std(inter_class_distances)],
                   capsize=5, color=['blue', 'red'], alpha=0.7)
        axes[1].set_ylabel('Mean Distance')
        axes[1].set_title('Average Distance Comparison')
        axes[1].grid(alpha=0.3)
        
        plt.suptitle('Intra-class Variation Analysis', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved variation analysis to: {save_path}")
        
        plt.show()
        plt.close()
    
    def visualize_feature_correlation(self,
                                      features: torch.Tensor,
                                      save_path: Optional[str] = None):
        """
        Visualize feature correlation matrix
        
        Args:
            features: (N, D) features
            save_path: Path to save
        """
        features_np = features.numpy()
        
        n_features = features_np.shape[1]
        if n_features > 100:
            idx = np.random.choice(n_features, 100, replace=False)
            features_np = features_np[:, idx]
        
        correlation = np.corrcoef(features_np.T)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation, cmap='RdYlBu', center=0,
                   xticklabels=False, yticklabels=False,
                   cbar_kws={'label': 'Correlation'})
        plt.title('Feature Correlation Matrix', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved correlation matrix to: {save_path}")
        
        plt.show()
        plt.close()
