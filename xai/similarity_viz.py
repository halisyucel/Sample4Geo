import torch
import torch.nn.functional as F
import numpy as np
import cv2
import matplotlib.pyplot as plt
from typing import Optional, List, Tuple
import seaborn as sns


class SimilarityVisualizer:
    """
    Visualize similarity between query and gallery embeddings
    Helps understand which images the model considers similar
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
    
    def compute_similarity_matrix(self,
                                   query_embeddings: torch.Tensor,
                                   gallery_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Compute cosine similarity matrix
        
        Args:
            query_embeddings: (N_query, D)
            gallery_embeddings: (N_gallery, D)
        
        Returns:
            Similarity matrix (N_query, N_gallery)
        """
        query_embeddings = F.normalize(query_embeddings, dim=-1)
        gallery_embeddings = F.normalize(gallery_embeddings, dim=-1)
        
        similarity = query_embeddings @ gallery_embeddings.T
        
        return similarity
    
    def visualize_top_k(self,
                        query_image: np.ndarray,
                        gallery_images: List[np.ndarray],
                        similarity_scores: np.ndarray,
                        top_k: int = 5,
                        save_path: Optional[str] = None):
        """
        Visualize top-k most similar gallery images for a query
        
        Args:
            query_image: Query image (H, W, 3)
            gallery_images: List of gallery images
            similarity_scores: Similarity scores for each gallery image
            top_k: Number of top matches to show
            save_path: Path to save visualization
        """
        top_k_indices = np.argsort(similarity_scores)[::-1][:top_k]
        top_k_scores = similarity_scores[top_k_indices]
        
        fig, axes = plt.subplots(1, top_k + 1, figsize=(4 * (top_k + 1), 4))
        
        axes[0].imshow(query_image)
        axes[0].set_title('Query\n(Drone)', fontsize=12, fontweight='bold')
        axes[0].axis('off')
        
        for idx, (img_idx, score) in enumerate(zip(top_k_indices, top_k_scores)):
            axes[idx + 1].imshow(gallery_images[img_idx])
            axes[idx + 1].set_title(f'Rank {idx+1}\nScore: {score:.4f}', fontsize=10)
            axes[idx + 1].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved top-k visualization to: {save_path}")
        
        plt.show()
        plt.close()
    
    def visualize_similarity_heatmap(self,
                                      similarity_matrix: np.ndarray,
                                      query_labels: Optional[np.ndarray] = None,
                                      gallery_labels: Optional[np.ndarray] = None,
                                      save_path: Optional[str] = None):
        """
        Visualize similarity matrix as heatmap
        
        Args:
            similarity_matrix: (N_query, N_gallery)
            query_labels: Labels for query images
            gallery_labels: Labels for gallery images
            save_path: Path to save visualization
        """
        plt.figure(figsize=(12, 10))
        
        if query_labels is not None and gallery_labels is not None:
            correct_mask = query_labels[:, np.newaxis] == gallery_labels[np.newaxis, :]
            
            ax = sns.heatmap(similarity_matrix, 
                            cmap='RdYlGn',
                            center=0.5,
                            cbar_kws={'label': 'Cosine Similarity'})
        else:
            ax = sns.heatmap(similarity_matrix,
                            cmap='viridis',
                            cbar_kws={'label': 'Cosine Similarity'})
        
        plt.xlabel('Gallery Images', fontsize=12)
        plt.ylabel('Query Images', fontsize=12)
        plt.title('Query-Gallery Similarity Matrix', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved similarity heatmap to: {save_path}")
        
        plt.show()
        plt.close()
    
    def visualize_embedding_space(self,
                                  embeddings: torch.Tensor,
                                  labels: Optional[np.ndarray] = None,
                                  method: str = 'tsne',
                                  save_path: Optional[str] = None):
        """
        Visualize embeddings in 2D using t-SNE or PCA
        
        Args:
            embeddings: (N, D) embeddings
            labels: Labels for coloring
            method: 'tsne' or 'pca'
            save_path: Path to save visualization
        """
        from sklearn.manifold import TSNE
        from sklearn.decomposition import PCA
        
        embeddings_np = embeddings.cpu().numpy()
        
        if method == 'tsne':
            reducer = TSNE(n_components=2, random_state=42, perplexity=30)
        else:
            reducer = PCA(n_components=2)
        
        embeddings_2d = reducer.fit_transform(embeddings_np)
        
        plt.figure(figsize=(10, 8))
        
        if labels is not None:
            unique_labels = np.unique(labels)
            colors = plt.cm.tab20(np.linspace(0, 1, len(unique_labels)))
            
            for idx, label in enumerate(unique_labels[:20]):
                mask = labels == label
                plt.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1],
                           c=[colors[idx]], label=f'ID {label}', alpha=0.6, s=50)
            
            if len(unique_labels) > 20:
                plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        else:
            plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], alpha=0.6, s=50)
        
        plt.xlabel(f'{method.upper()} Dimension 1', fontsize=12)
        plt.ylabel(f'{method.upper()} Dimension 2', fontsize=12)
        plt.title(f'Embedding Space Visualization ({method.upper()})', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved embedding visualization to: {save_path}")
        
        plt.show()
        plt.close()
    
    def analyze_retrieval(self,
                          query_image: torch.Tensor,
                          query_label: int,
                          gallery_images: torch.Tensor,
                          gallery_labels: torch.Tensor,
                          top_k: int = 10,
                          save_path: Optional[str] = None) -> dict:
        """
        Complete retrieval analysis with visualization
        
        Args:
            query_image: (1, C, H, W)
            query_label: Ground truth label
            gallery_images: (N, C, H, W)
            gallery_labels: (N,) ground truth labels
            top_k: Number of top results
            save_path: Path to save results
        
        Returns:
            Dictionary with retrieval metrics
        """
        with torch.no_grad():
            query_embedding = self.model(query_image.to(self.device))
            gallery_embeddings = self.model(gallery_images.to(self.device))
        
        similarity = self.compute_similarity_matrix(query_embedding, gallery_embeddings)
        similarity = similarity.squeeze().cpu().numpy()
        
        ranked_indices = np.argsort(similarity)[::-1]
        ranked_labels = gallery_labels[ranked_indices]
        
        correct_position = np.where(ranked_labels == query_label)[0]
        
        metrics = {
            'rank': correct_position[0] + 1 if len(correct_position) > 0 else -1,
            'top1': 1 if len(correct_position) > 0 and correct_position[0] == 0 else 0,
            'top5': 1 if len(correct_position) > 0 and correct_position[0] < 5 else 0,
            'top10': 1 if len(correct_position) > 0 and correct_position[0] < 10 else 0,
            'top_k_scores': similarity[ranked_indices[:top_k]],
            'top_k_labels': ranked_labels[:top_k]
        }
        
        return metrics
