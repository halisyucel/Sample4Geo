import torch
import torch.nn.functional as F
import numpy as np
import cv2
from typing import Tuple, Optional
import matplotlib.pyplot as plt


class OcclusionSensitivity:
    """
    Perturbation-based XAI for cross-view geo-localization.

    Slides an occlusion patch over the query image and measures how much
    the cosine similarity to the gallery drops at each position.
    High drop = that region was important for the match.
    """

    def __init__(self, model, device: Optional[torch.device] = None):
        self.model = model
        self.model.eval()
        self.device = device or next(model.parameters()).device

    def compute_sensitivity(
        self,
        query_image: torch.Tensor,
        gallery_embedding: torch.Tensor,
        patch_size: int = 64,
        stride: int = 16,
        occlusion_value: float = 0.0,
    ) -> np.ndarray:
        """
        Compute occlusion sensitivity map for a query image.

        Args:
            query_image: (1, C, H, W) preprocessed query tensor
            gallery_embedding: (1, D) gallery embedding (no grad needed)
            patch_size: Side length of the square occlusion patch
            stride: Step size between patch positions
            occlusion_value: Pixel fill value (0 = mean-normalised black)

        Returns:
            Importance map (H, W), values in [0, 1] where 1 = most important
        """
        query_image = query_image.to(self.device)
        gallery_embedding = gallery_embedding.to(self.device)

        _, _, H, W = query_image.shape

        with torch.no_grad():
            baseline_embedding = self.model(query_image)
            baseline_score = F.cosine_similarity(
                baseline_embedding, gallery_embedding, dim=-1
            ).item()

        importance = np.zeros((H, W), dtype=np.float32)
        counts = np.zeros((H, W), dtype=np.float32)

        y_positions = list(range(0, H - patch_size + 1, stride))
        x_positions = list(range(0, W - patch_size + 1, stride))

        for y in y_positions:
            for x in x_positions:
                occluded = query_image.clone()
                occluded[:, :, y : y + patch_size, x : x + patch_size] = occlusion_value

                with torch.no_grad():
                    occ_embedding = self.model(occluded)
                    occ_score = F.cosine_similarity(
                        occ_embedding, gallery_embedding, dim=-1
                    ).item()

                drop = baseline_score - occ_score
                importance[y : y + patch_size, x : x + patch_size] += drop
                counts[y : y + patch_size, x : x + patch_size] += 1

        counts = np.maximum(counts, 1)
        importance = importance / counts

        # Normalise to [0, 1]; negative drops (occlusion helped) map to 0
        importance = np.maximum(importance, 0)
        max_val = importance.max()
        if max_val > 1e-8:
            importance = importance / max_val

        return importance

    def compute_pair_sensitivity(
        self,
        query_image: torch.Tensor,
        gallery_image: torch.Tensor,
        patch_size: int = 64,
        stride: int = 16,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute sensitivity maps for both query and gallery images.

        Returns:
            (query_map, gallery_map)
        """
        query_image = query_image.to(self.device)
        gallery_image = gallery_image.to(self.device)

        with torch.no_grad():
            gallery_embedding = self.model(gallery_image)
            query_embedding = self.model(query_image)

        query_map = self.compute_sensitivity(
            query_image, gallery_embedding, patch_size, stride
        )
        gallery_map = self.compute_sensitivity(
            gallery_image, query_embedding, patch_size, stride
        )

        return query_map, gallery_map

    def compute_faithfulness(
        self,
        query_image: torch.Tensor,
        gallery_embedding: torch.Tensor,
        importance_map: np.ndarray,
        steps: int = 10,
        occlusion_value: float = 0.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Faithfulness metric: progressively mask top-important regions,
        measure score at each step.

        Args:
            steps: Number of masking steps (10% increments by default)

        Returns:
            (mask_fractions, scores) arrays for plotting
        """
        query_image = query_image.to(self.device)
        gallery_embedding = gallery_embedding.to(self.device)

        with torch.no_grad():
            baseline_embedding = self.model(query_image)
            baseline_score = F.cosine_similarity(
                baseline_embedding, gallery_embedding, dim=-1
            ).item()

        flat_importance = importance_map.flatten()
        sorted_indices = np.argsort(flat_importance)[::-1]  # descending

        _, _, H, W = query_image.shape
        total_pixels = H * W

        fractions = np.linspace(0, 1, steps + 1)
        scores = [baseline_score]

        for frac in fractions[1:]:
            n_mask = int(frac * total_pixels)
            mask_indices = sorted_indices[:n_mask]

            occluded = query_image.clone()
            flat_img = occluded.view(1, 3, -1)
            flat_img[:, :, mask_indices] = occlusion_value
            occluded = flat_img.view_as(query_image)

            with torch.no_grad():
                occ_embedding = self.model(occluded)
                score = F.cosine_similarity(
                    occ_embedding, gallery_embedding, dim=-1
                ).item()
            scores.append(score)

        return fractions, np.array(scores)

    def visualize_map(
        self,
        image: torch.Tensor,
        importance_map: np.ndarray,
        alpha: float = 0.5,
        colormap: int = cv2.COLORMAP_JET,
    ) -> np.ndarray:
        """Overlay importance map on original image."""
        img_np = image.squeeze().cpu().numpy().transpose(1, 2, 0)
        img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-8)
        img_np = (img_np * 255).astype(np.uint8)

        heatmap = cv2.applyColorMap(np.uint8(255 * importance_map), colormap)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

        return cv2.addWeighted(img_np, 1 - alpha, heatmap, alpha, 0)

    def save_visualization(
        self,
        image: torch.Tensor,
        importance_map: np.ndarray,
        save_path: str,
        title: str = "Occlusion Sensitivity",
        faithfulness_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    ):
        """
        Save occlusion sensitivity visualization.

        If faithfulness_data is provided, adds a faithfulness curve subplot.
        """
        overlay = self.visualize_map(image, importance_map)

        img_np = image.squeeze().cpu().numpy().transpose(1, 2, 0)
        img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-8)

        n_cols = 4 if faithfulness_data is not None else 3
        fig, axes = plt.subplots(1, n_cols, figsize=(5 * n_cols, 5))

        axes[0].imshow(img_np)
        axes[0].set_title("Original Image")
        axes[0].axis("off")

        axes[1].imshow(importance_map, cmap="jet", vmin=0, vmax=1)
        axes[1].set_title("Importance Map")
        axes[1].axis("off")

        axes[2].imshow(overlay)
        axes[2].set_title("Overlay")
        axes[2].axis("off")

        if faithfulness_data is not None:
            fractions, scores = faithfulness_data
            axes[3].plot(fractions * 100, scores, marker="o", color="crimson")
            axes[3].set_xlabel("Masked Pixels (%)")
            axes[3].set_ylabel("Cosine Similarity")
            axes[3].set_title("Faithfulness Curve")
            axes[3].grid(True, alpha=0.3)

        plt.suptitle(title)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()

        print(f"Saved: {save_path}")
