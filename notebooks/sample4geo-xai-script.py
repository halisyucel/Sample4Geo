# %% [markdown]
# # Sample4Geo — XAI Analysis
# 
# **Course:** Hacettepe University — BBM416 Computer Vision  
# **Topic:** Explainable AI (XAI) on Cross-View Geo-Localization  
# **Base paper:** Sample4Geo (ICCV 2023)  
# **Fork:** https://github.com/halisyucel/Sample4Geo  
# **Upstream:** https://github.com/Skyy93/Sample4Geo  
# 
# **Group:**
# - Halis Yücel (VIGOR dataset)
# - Kuzey Ersoy (University-1652 dataset)
# 
# > **Datasets & Weights:** All datasets and pretrained weights are publicly available at  
# > https://drive.google.com/drive/folders/1m85pmQhE_iMRUbc173Z81Gs3ID5-wD2C?usp=sharing  
# 
# **Objective:**  
# Apply two XAI methods (GradCAM and Occlusion Sensitivity) to a pretrained Sample4Geo model and analyze what image regions the model attends to when matching drone/street-level views to satellite images.
# 
# **Datasets:**
# - University-1652 (drone ↔ satellite)
# - VIGOR (street panorama ↔ satellite, same-area and cross-area splits)
# 
# **Success criteria:**
# - Baseline evaluation metrics reproduced for both datasets
# - GradCAM and Occlusion Sensitivity heatmaps generated for successful and failed matches
# - Faithfulness curves computed for Occlusion Sensitivity
# - Visual comparison between the two XAI methods
# - All figures saved in presentation-friendly format under `xai_results/`
# 

# %% [markdown]
# ## 0. Environment Setup
# 
# Run this section first. It mounts Google Drive, clones the repo, installs dependencies, and sets all paths.
# 
# > **Note:** The dataset lives at `MyDrive/vision-datasets/`. The code repo is cloned from GitHub.
# 

# %%
# mount google drive
from google.colab import drive
drive.mount('/content/drive')

# %%
import os

# paths
DRIVE_ROOT       = '/content/drive/MyDrive/vision-datasets'
U1652_ROOT       = f'{DRIVE_ROOT}/University-Release'
VIGOR_ROOT       = f'{DRIVE_ROOT}/VIGOR'
REPO_DIR         = '/content/Sample4Geo'
PRETRAINED_DIR   = f'{DRIVE_ROOT}/pretrained'
RESULTS_DIR      = f'{DRIVE_ROOT}/xai_results'
CACHE_DIR        = f'{DRIVE_ROOT}/xai_cache'


# pretrained weight paths
CKPT_U1652       = f'{PRETRAINED_DIR}/university/convnext_base.fb_in22k_ft_in1k_384/weights_e1_0.9515.pth'
CKPT_VIGOR_SAME  = f'{PRETRAINED_DIR}/vigor_same/convnext_base.fb_in22k_ft_in1k_384/weights_e40_0.7786.pth'
CKPT_VIGOR_CROSS = f'{PRETRAINED_DIR}/vigor_cross/convnext_base.fb_in22k_ft_in1k_384/weights_e40_0.6109.pth'

print('drive root exists:', os.path.exists(DRIVE_ROOT))
print('U1652 root exists:', os.path.exists(U1652_ROOT))
print('VIGOR root exists:', os.path.exists(VIGOR_ROOT))

# %%
# clone repo and install dependencies
import subprocess

if not os.path.exists(REPO_DIR):
    subprocess.run(['git', 'clone', 'https://github.com/halisyucel/Sample4Geo.git', REPO_DIR], check=True)
else:
    print('repo already cloned, pulling latest...')
    subprocess.run(['git', '-C', REPO_DIR, 'pull'], check=True)

# %pip install -q timm albumentations opencv-python-headless

# %%
# add repo to path and verify gpu
import sys
sys.path.insert(0, REPO_DIR)

import torch
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('device:', device)
if device == 'cuda':
    print('gpu:', torch.cuda.get_device_name(0))

os.makedirs(RESULTS_DIR, exist_ok=True)

print(f'setup complete. device: {device}')


# %% [markdown]
# ## 1. Pretrained Weights
# 
# Weights are stored under `MyDrive/vision-datasets/pretrained/` and loaded directly from Drive.
# 
# | Dataset | File | Reported R@1 |
# |---|---|---|
# | University-1652 | `weights_e1_0.9515.pth` | 95.15% |
# | VIGOR same-area | `weights_e40_0.7786.pth` | 77.86% |
# | VIGOR cross-area | `weights_e40_0.6109.pth` | 61.09% |
# 

# %%
# pretrained weights are already extracted at mydrive/vision-datasets/pretrained
# just verify the checkpoint files exist before proceeding

for ckpt in [CKPT_U1652, CKPT_VIGOR_SAME, CKPT_VIGOR_CROSS]:
    print(os.path.exists(ckpt), ckpt)

print('all pretrained checkpoint files verified.')


# %% [markdown]
# ## 2. Dataset Paths
# 
# Datasets are read directly from Google Drive.
# 
# **University-1652 structure:**
# ```
# University-Release/
# ├── train/
# │   ├── drone/, satellite/, google/, street/
# └── test/
#     ├── query_drone/, gallery_satellite/
#     ├── query_satellite/, gallery_drone/
#     └── ...
# ```
# 
# **VIGOR structure:**
# ```
# VIGOR/
# ├── Chicago/, NewYork/, SanFrancisco/, Seattle/
# │   ├── panorama/   ← street-view (ground)
# │   └── satellite/  ← satellite tiles
# └── splits/
#     └── {city}/
#         ├── satellite_list.txt
#         ├── pano_label_balanced.txt
#         ├── same_area_balanced_train.txt
#         └── same_area_balanced_test.txt
# ```
# 
# > **Note:** `VigorDataset` expects `data_folder/ground/{city}/` but Drive has `panorama/`.  
# > Symlinks are created at runtime to bridge this.
# 

# %%
# create symlinks so vigordataset can find ground/ -> panorama/
# the dataset class hardcodes data_folder/ground/{city}/ paths.
cities = ['Chicago', 'NewYork', 'SanFrancisco', 'Seattle']

for city in cities:
    src = os.path.join(VIGOR_ROOT, city, 'panorama')
    print(f'{city}: panorama exists = {os.path.exists(src)}')

VIGOR_DATA = '/content/vigor_data'
os.makedirs(VIGOR_DATA, exist_ok=True)

for city in cities:
    ground_dir = os.path.join(VIGOR_DATA, 'ground', city)
    sat_dir    = os.path.join(VIGOR_DATA, 'satellite', city)
    split_dir  = os.path.join(VIGOR_DATA, 'splits', city)

    os.makedirs(os.path.dirname(ground_dir), exist_ok=True)
    os.makedirs(os.path.dirname(sat_dir),    exist_ok=True)
    os.makedirs(os.path.dirname(split_dir),  exist_ok=True)

    if not os.path.exists(ground_dir):
        os.symlink(os.path.join(VIGOR_ROOT, city, 'panorama'), ground_dir)
    if not os.path.exists(sat_dir):
        os.symlink(os.path.join(VIGOR_ROOT, city, 'satellite'), sat_dir)
    if not os.path.exists(split_dir):
        os.symlink(os.path.join(VIGOR_ROOT, 'splits', city), split_dir)

    print(f'{city}: ground={os.path.exists(ground_dir)}, sat={os.path.exists(sat_dir)}, splits={os.path.exists(split_dir)}')

print('vigor symlink setup complete — ground/ points to panorama/.')


# %% [markdown]
# ## 3. Baseline Evaluation — University-1652
# 
# Task: drone → satellite retrieval (D2S).  
# Model: ConvNeXt-Base pretrained on University-1652.  
# 
# | Metric | Ours (A100) | Paper |
# |---|---|---|
# | R@1 | 92.66% | 95.15% |
# | R@5 | 97.69% | — |
# | R@10 | 98.24% | — |
# | AP | 93.81% | — |
# 
# > The ~2.5% gap vs paper is consistent across MPS and CUDA runs — likely a difference in eval split or test-time augmentation in the original setup.
# 

# %%
from dataclasses import dataclass
from torch.utils.data import DataLoader
from sample4geo.dataset.university import U1652DatasetEval, get_transforms
from sample4geo.evaluate.university import evaluate
from sample4geo.model import TimmModel

@dataclass
class U1652Config:
    model: str            = 'convnext_base.fb_in22k_ft_in1k_384'
    img_size: int         = 384
    batch_size: int       = 256
    verbose: bool         = True
    gpu_ids: tuple        = (0,)
    normalize_features: bool = True
    eval_gallery_n: int   = -1
    dataset: str          = 'U1652-D2S'
    num_workers: int      = 16
    device: str           = device
    checkpoint_start: str = CKPT_U1652

cfg_u = U1652Config()
cfg_u.query_folder_test   = f'{U1652_ROOT}/test/query_drone'
cfg_u.gallery_folder_test = f'{U1652_ROOT}/test/gallery_satellite'

print('config:', cfg_u)

# %%
# load model
model_u = TimmModel(cfg_u.model, pretrained=True, img_size=cfg_u.img_size)
data_config = model_u.get_config()
mean, std = data_config['mean'], data_config['std']
img_size = (cfg_u.img_size, cfg_u.img_size)

state_dict = torch.load(cfg_u.checkpoint_start, map_location='cpu')
model_u.load_state_dict(state_dict, strict=False)
model_u = model_u.to(device).eval()
print('model loaded.')

print('university-1652 model ready.')


# %%
# build dataloaders
val_transforms, _, _ = get_transforms(img_size, mean=mean, std=std)

query_dataset = U1652DatasetEval(
    data_folder=cfg_u.query_folder_test, mode='query', transforms=val_transforms)
gallery_dataset = U1652DatasetEval(
    data_folder=cfg_u.gallery_folder_test, mode='gallery',
    transforms=val_transforms, sample_ids=query_dataset.get_sample_ids(), gallery_n=-1)

query_loader = DataLoader(
    query_dataset, batch_size=cfg_u.batch_size,
    num_workers=cfg_u.num_workers, shuffle=False, pin_memory=True,
        prefetch_factor=4, persistent_workers=True)
gallery_loader = DataLoader(
    gallery_dataset, batch_size=cfg_u.batch_size,
    num_workers=cfg_u.num_workers, shuffle=False, pin_memory=True,
        prefetch_factor=4, persistent_workers=True)

print(f'query images:   {len(query_dataset)}')
print(f'gallery images: {len(gallery_dataset)}')

# extract features — load from drive cache if available
from sample4geo.trainer import predict

os.makedirs(CACHE_DIR, exist_ok=True)
CACHE_U1652 = f'{CACHE_DIR}/u1652_features.pt'

if os.path.exists(CACHE_U1652):
    print('loading u1652 features from drive cache...')
    _c = torch.load(CACHE_U1652, map_location='cpu')
    q_feats_u = _c['q_feats'].to(device)
    g_feats_u = _c['g_feats'].to(device)
    q_ids_u   = _c['q_ids'].to(device)
    g_ids_u   = _c['g_ids'].to(device)
    print(f'query: {q_feats_u.shape}, gallery: {g_feats_u.shape}')
else:
    print('no cache found — extracting u1652 features...')
    q_feats_u, q_ids_u = predict(cfg_u, model_u, query_loader)
    g_feats_u, g_ids_u = predict(cfg_u, model_u, gallery_loader)
    torch.save({
        'q_feats': q_feats_u.cpu(), 'g_feats': g_feats_u.cpu(),
        'q_ids':   q_ids_u.cpu(),   'g_ids':   g_ids_u.cpu(),
    }, CACHE_U1652)
    print(f'u1652 features saved to drive cache: {CACHE_U1652}')


# %%
# compute u1652 eval metrics — skip if already cached
import json as _json

CACHE_U1652_METRICS = f'{CACHE_DIR}/u1652_metrics.json'

if os.path.exists(CACHE_U1652_METRICS):
    with open(CACHE_U1652_METRICS) as _f:
        _m = _json.load(_f)
    r1_u, result_str_u = _m['r1'], _m['result_str']
    print('u1652 metrics loaded from cache.')
else:
    r1_u, result_str_u = evaluate(
        config=cfg_u,
        model=model_u,
        query_loader=query_loader,
        gallery_loader=gallery_loader,
        precomputed=(q_feats_u, q_ids_u, g_feats_u, g_ids_u),
        ranks=[1, 5, 10],
        step_size=1000,
        cleanup=False,
    )
    with open(CACHE_U1652_METRICS, 'w') as _f:
        _json.dump({'r1': float(r1_u), 'result_str': result_str_u}, _f)
    print('u1652 metrics saved to cache.')

print('\n=== university-1652 results ===')
print(result_str_u)


# %% [markdown]
# ## 4. Baseline Evaluation — VIGOR
# 
# Two evaluation modes:
# - **Same-area:** train/test cities overlap geographically (harder)
# - **Cross-area:** train on NYC+Seattle, test on Chicago+SanFrancisco
# 
# Task: street-view panorama → satellite retrieval.
# 

# %%
import importlib, sys
for _mod in list(sys.modules):
    if _mod.startswith('sample4geo'):
        del sys.modules[_mod]

from sample4geo.dataset.vigor import VigorDatasetEval
from sample4geo.transforms import get_transforms_val
from sample4geo.evaluate.vigor import evaluate as evaluate_vigor
import json as _json

@dataclass
class VigorConfig:
    model: str            = 'convnext_base.fb_in22k_ft_in1k_384'
    img_size: int         = 384
    batch_size: int       = 256
    verbose: bool         = True
    gpu_ids: tuple        = (0,)
    normalize_features: bool = True
    data_folder: str      = VIGOR_DATA
    ground_cutting: int   = 0
    num_workers: int      = 16
    device: str           = device
    same_area: bool       = True
    checkpoint_start: str = CKPT_VIGOR_SAME

def run_vigor_eval(same_area: bool):
    tag          = 'same' if same_area else 'cross'
    cache_feats  = f'{CACHE_DIR}/vigor_{tag}_features.pt'
    cache_metrics = f'{CACHE_DIR}/vigor_{tag}_metrics.json'
    ckpt         = CKPT_VIGOR_SAME if same_area else CKPT_VIGOR_CROSS
    cfg          = VigorConfig(same_area=same_area, checkpoint_start=ckpt)

    os.makedirs(CACHE_DIR, exist_ok=True)

    # load or compute features
    if os.path.exists(cache_feats):
        print(f'loading vigor {tag}-area features from drive cache...')
        _c = torch.load(cache_feats, map_location='cpu')
        ref_feats  = _c['ref_feats'].to(device)
        ref_labels = _c['ref_labels'].to(device)
        q_feats_   = _c['q_feats'].to(device)
        q_labels   = _c['q_labels'].to(device)
        precomputed = (ref_feats, ref_labels, q_feats_, q_labels)
        print(f'query: {q_feats_.shape}, gallery: {ref_feats.shape}')
    else:
        model_v = TimmModel(cfg.model, pretrained=True, img_size=cfg.img_size)
        data_config = model_v.get_config()
        mean_v, std_v = data_config['mean'], data_config['std']
        state_dict = torch.load(cfg.checkpoint_start, map_location='cpu')
        model_v.load_state_dict(state_dict, strict=False)
        model_v = model_v.to(device).eval()

        img_size_v      = cfg.img_size
        image_size_sat  = (img_size_v, img_size_v)
        new_width       = img_size_v * 2
        new_height      = int(((1024 - 2 * cfg.ground_cutting) / 2048) * new_width)
        img_size_ground = (new_height, new_width)

        sat_transforms, ground_transforms = get_transforms_val(
            image_size_sat, img_size_ground, mean=mean_v, std=std_v,
            ground_cutting=cfg.ground_cutting)

        ref_dataset = VigorDatasetEval(
            data_folder=cfg.data_folder, split='test', img_type='reference',
            same_area=cfg.same_area, transforms=sat_transforms)
        q_dataset = VigorDatasetEval(
            data_folder=cfg.data_folder, split='test', img_type='query',
            same_area=cfg.same_area, transforms=ground_transforms)

        ref_loader = DataLoader(ref_dataset, batch_size=cfg.batch_size,
            num_workers=cfg.num_workers, shuffle=False, pin_memory=True,
            prefetch_factor=4, persistent_workers=True)
        q_loader = DataLoader(q_dataset, batch_size=cfg.batch_size,
            num_workers=cfg.num_workers, shuffle=False, pin_memory=True,
            prefetch_factor=4, persistent_workers=True)

        print(f'no cache — extracting vigor {tag}-area features...')
        from sample4geo.trainer import predict as _predict
        ref_feats,  ref_labels = _predict(cfg, model_v, ref_loader)
        q_feats_,   q_labels   = _predict(cfg, model_v, q_loader)
        torch.save({
            'ref_feats':  ref_feats.cpu(), 'ref_labels': ref_labels.cpu(),
            'q_feats':    q_feats_.cpu(),  'q_labels':   q_labels.cpu(),
        }, cache_feats)
        print(f'vigor {tag}-area features saved to drive cache.')
        precomputed = (ref_feats, ref_labels, q_feats_, q_labels)

    # load or compute metrics
    if os.path.exists(cache_metrics):
        with open(cache_metrics) as _f:
            _m = _json.load(_f)
        r1, result_str = _m['r1'], _m['result_str']
        print(f'vigor {tag}-area metrics loaded from cache.')
    else:
        r1, result_str = evaluate_vigor(
            config=cfg, model=None,
            reference_dataloader=None, query_dataloader=None,
            precomputed=precomputed,
            ranks=[1, 5, 10], step_size=1000, cleanup=False)
        with open(cache_metrics, 'w') as _f:
            _json.dump({'r1': float(r1), 'result_str': result_str}, _f)
        print(f'vigor {tag}-area metrics saved to cache.')

    print(f'\n{"-"*30}[vigor {tag}]{"-"*30}')
    print(result_str)
    return r1, result_str, precomputed

print('vigor eval function ready.')


# %%
r1_same, result_same, precomputed_same = run_vigor_eval(same_area=True)
print('vigor same-area:', result_same)


# %%
r1_cross, result_cross, precomputed_cross = run_vigor_eval(same_area=False)
print('vigor cross-area:', result_cross)


# %% [markdown]
# ## 5. XAI Setup
# 
# Two complementary methods:
# 
# | Method | Type | How it works |
# |---|---|---|
# | **GradCAM** | White-box, gradient-based | Backpropagates cosine similarity gradient through last conv layer; weights activations by mean gradient |
# | **Occlusion Sensitivity** | Black-box, perturbation-based | Slides a patch over the image; measures cosine similarity drop at each position |
# 
# GradCAM is fast (~1 forward + 1 backward). Occlusion is slow (~(H/stride)² forward passes).
# 

# %%
import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch.nn.functional as F
import albumentations as A
from albumentations.pytorch import ToTensorV2


# image loading helper
def load_image(path: str, img_size: int = 384) -> torch.Tensor:
    """Load, resize and normalize an image. Returns (1, C, H, W) tensor."""
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    transform = A.Compose([
        A.Resize(img_size, img_size, interpolation=cv2.INTER_LINEAR_EXACT),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])
    return transform(image=img)['image'].unsqueeze(0)


# tensor → displayable numpy
def tensor_to_display(t: torch.Tensor) -> np.ndarray:
    """Denormalize and convert to uint8 HWC."""
    img = t.squeeze().cpu().numpy().transpose(1, 2, 0)
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)
    return (img * 255).astype(np.uint8)


print('helpers ready.')

# %% [markdown]
# ### 5.1 GradCAM
# 

# %%
class GradCAMExtractor:
    """
    GradCAM for ConvNeXt models in Sample4Geo.
    Visualizes which regions the model focuses on for geo-localization.
    """

    def __init__(self, model, target_layer=None):
        self.model = model
        self.model.eval()
        self.target_layer = target_layer or self._find_target_layer()
        self.gradients = None
        self.activations = None
        self._hooks = []
        self._register_hooks()

    def _find_target_layer(self):
        """Find the last block of the last ConvNeXt stage."""
        target = None
        for name, module in self.model.model.named_modules():
            parts = name.split('.')
            if 'stages' in name and len(parts) == 3:
                target = module
        if target is None:
            for _, module in reversed(list(self.model.model.named_modules())):
                if isinstance(module, torch.nn.Conv2d):
                    target = module
                    break
        return target

    def _register_hooks(self):
        def fwd(module, inp, out):
            self.activations = out.detach()
        def bwd(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()
        self._hooks.append(self.target_layer.register_forward_hook(fwd))
        self._hooks.append(self.target_layer.register_full_backward_hook(bwd))

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def generate_cam(self, image: torch.Tensor,
                     target_embedding: torch.Tensor = None) -> np.ndarray:
        self.model.zero_grad()
        embedding = self.model(image)
        if target_embedding is not None:
            score = F.cosine_similarity(embedding, target_embedding, dim=-1)
        else:
            score = embedding.norm(dim=-1).mean()
        score.backward(retain_graph=True)
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * self.activations).sum(dim=1, keepdim=True))
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cv2.resize(cam, (image.shape[3], image.shape[2]))

    def generate_pair_cam(self, query: torch.Tensor,
                          gallery: torch.Tensor):
        with torch.no_grad():
            gallery_emb = self.model(gallery)
        query_cam = self.generate_cam(query, gallery_emb)
        with torch.no_grad():
            query_emb = self.model(query)
        gallery_cam = self.generate_cam(gallery, query_emb)
        return query_cam, gallery_cam

    @staticmethod
    def overlay(image: torch.Tensor, cam: np.ndarray,
                alpha: float = 0.4) -> np.ndarray:
        img_np = tensor_to_display(image)
        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        return cv2.addWeighted(img_np, 1 - alpha, heatmap, alpha, 0)


print('gradcamextractor defined.')

# %% [markdown]
# ### 5.2 Occlusion Sensitivity
# 

# %%
class OcclusionSensitivity:
    """
    perturbation-based xai.
    slides an occlusion patch over the image and measures cosine similarity drop.
    patches are processed in batches for speed (~20-30x faster than single forward passes).
    """

    def __init__(self, model, device=None, batch_size=32):
        self.model      = model
        self.model.eval()
        self.device     = device or next(model.parameters()).device
        self.batch_size = batch_size

    def compute_sensitivity(self, query_image: torch.Tensor,
                             gallery_embedding: torch.Tensor,
                             patch_size: int = 64,
                             stride: int = 32,
                             occlusion_value: float = 0.0) -> np.ndarray:
        query_image       = query_image.to(self.device)
        gallery_embedding = gallery_embedding.to(self.device)
        _, _, H, W        = query_image.shape

        with torch.inference_mode():
            base_score = F.cosine_similarity(
                self.model(query_image), gallery_embedding, dim=-1
            ).item()

        # build list of (y, x) positions
        positions = [
            (y, x)
            for y in range(0, H - patch_size + 1, stride)
            for x in range(0, W - patch_size + 1, stride)
        ]

        importance = np.zeros((H, W), dtype=np.float32)
        counts     = np.zeros((H, W), dtype=np.float32)

        # process positions in batches
        for batch_start in range(0, len(positions), self.batch_size):
            batch_pos = positions[batch_start : batch_start + self.batch_size]
            batch_imgs = []
            for y, x in batch_pos:
                occ = query_image.clone()
                occ[:, :, y:y + patch_size, x:x + patch_size] = occlusion_value
                batch_imgs.append(occ)
            batch_t = torch.cat(batch_imgs, dim=0)  # (B, C, H, W)
            gal_exp = gallery_embedding.expand(len(batch_pos), -1)
            with torch.inference_mode():
                scores = F.cosine_similarity(self.model(batch_t), gal_exp, dim=-1)
            for (y, x), occ_score in zip(batch_pos, scores.tolist()):
                drop = base_score - occ_score
                importance[y:y + patch_size, x:x + patch_size] += drop
                counts[y:y + patch_size, x:x + patch_size]     += 1

        importance = importance / np.maximum(counts, 1)
        importance = np.maximum(importance, 0)
        mx = importance.max()
        if mx > 1e-8:
            importance /= mx
        return importance

    def compute_pair_sensitivity(self, query: torch.Tensor, gallery: torch.Tensor,
                                  patch_size: int = 64, stride: int = 32):
        query   = query.to(self.device)
        gallery = gallery.to(self.device)
        with torch.inference_mode():
            gallery_emb = self.model(gallery)
            query_emb   = self.model(query)
        q_map = self.compute_sensitivity(query,   gallery_emb, patch_size, stride)
        g_map = self.compute_sensitivity(gallery, query_emb,   patch_size, stride)
        return q_map, g_map

    def compute_faithfulness(self, query: torch.Tensor,
                              gallery_embedding: torch.Tensor,
                              importance_map: np.ndarray,
                              steps: int = 10,
                              occlusion_value: float = 0.0):
        query             = query.to(self.device)
        gallery_embedding = gallery_embedding.to(self.device)
        with torch.inference_mode():
            base_score = F.cosine_similarity(
                self.model(query), gallery_embedding, dim=-1
            ).item()

        _, _, H, W     = query.shape
        sorted_indices = np.argsort(importance_map.flatten())[::-1].copy()
        total_pixels   = H * W

        fractions = np.linspace(0, 1, steps + 1)
        scores    = [base_score]
        for frac in fractions[1:]:
            n_mask   = int(frac * total_pixels)
            mask_idx = sorted_indices[:n_mask]
            occ      = query.clone()
            flat_img = occ.reshape(1, 3, -1)
            flat_img[:, :, mask_idx] = occlusion_value
            occ = flat_img.reshape_as(query)
            with torch.inference_mode():
                score = F.cosine_similarity(
                    self.model(occ), gallery_embedding, dim=-1
                ).item()
            scores.append(score)
        return fractions, np.array(scores)

    @staticmethod
    def overlay(image: torch.Tensor, importance_map: np.ndarray,
                alpha: float = 0.5) -> np.ndarray:
        img_np  = tensor_to_display(image)
        heatmap = cv2.applyColorMap(np.uint8(255 * importance_map), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        return cv2.addWeighted(img_np, 1 - alpha, heatmap, alpha, 0)


print('occlusionsensitivity defined.')


# %% [markdown]
# ### 5.3 Visualization helper
# 

# %%
def show_gradcam(query_t, gallery_t, query_cam, gallery_cam,
                 title='', save_path=None):
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    rows = [
        (query_t,   query_cam,   'Query'),
        (gallery_t, gallery_cam, 'Gallery'),
    ]
    for row_idx, (img_t, cam, label) in enumerate(rows):
        img_np = tensor_to_display(img_t)
        overlay = GradCAMExtractor.overlay(img_t, cam)
        axes[row_idx, 0].imshow(img_np)
        axes[row_idx, 0].set_title(f'{label} — Original')
        axes[row_idx, 1].imshow(cam, cmap='jet')
        axes[row_idx, 1].set_title(f'{label} — GradCAM')
        axes[row_idx, 2].imshow(overlay)
        axes[row_idx, 2].set_title(f'{label} — Overlay')
    for ax in axes.flat:
        ax.axis('off')
    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def show_occlusion(query_t, gallery_t, query_map, gallery_map,
                   q_faith=None, g_faith=None, title='', save_path=None):
    n_cols = 4 if q_faith else 3
    fig, axes = plt.subplots(2, n_cols, figsize=(5 * n_cols, 10))
    rows = [
        (query_t,   query_map,   q_faith, 'Query'),
        (gallery_t, gallery_map, g_faith, 'Gallery'),
    ]
    for row_idx, (img_t, imap, faith, label) in enumerate(rows):
        img_np  = tensor_to_display(img_t)
        overlay = OcclusionSensitivity.overlay(img_t, imap)
        axes[row_idx, 0].imshow(img_np)
        axes[row_idx, 0].set_title(f'{label} — Original')
        axes[row_idx, 1].imshow(imap, cmap='jet', vmin=0, vmax=1)
        axes[row_idx, 1].set_title(f'{label} — Importance Map')
        axes[row_idx, 2].imshow(overlay)
        axes[row_idx, 2].set_title(f'{label} — Overlay')
        if faith and n_cols == 4:
            fracs, scores = faith
            axes[row_idx, 3].plot(fracs * 100, scores, marker='o', color='crimson')
            axes[row_idx, 3].set_xlabel('Masked Pixels (%)')
            axes[row_idx, 3].set_ylabel('Cosine Similarity')
            axes[row_idx, 3].set_title(f'{label} — Faithfulness')
            axes[row_idx, 3].grid(True, alpha=0.3)
    for ax in axes.flat:
        ax.axis('off') if ax.has_data() and ax.get_xlabel() == '' else None
    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


print('visualization helpers ready.')

# %% [markdown]
# ## 6. XAI Pair Selection — University-1652
# 
# We need representative pairs:
# - **Successful matches:** model's top-1 is correct  
# - **Failed matches:** model's top-1 is wrong
# 
# Run this section after Section 3 (eval) so embeddings are computed.
# 

# %%
# reuse features already extracted and cached in section 3
# no need to reload model or re-run predict
q_feats = q_feats_u
g_feats = g_feats_u
q_ids   = q_ids_u
g_ids   = g_ids_u

print(f'feature extraction complete.')
print(f'query: {q_feats.shape}, gallery: {g_feats.shape}')
print('similarity matrix and pair selection can now be computed.')


# %%
# find successful and failed pairs
# similarity matrix
sim_matrix = (q_feats @ g_feats.T).cpu().numpy()   # (N_query, N_gallery)
q_ids_np = q_ids.cpu().numpy()
g_ids_np = g_ids.cpu().numpy()

successful, failed = [], []

for i in range(len(q_ids_np)):
    sims = sim_matrix[i]
    top1_idx  = int(np.argmax(sims))
    top1_sim  = sims[top1_idx]
    top1_id   = g_ids_np[top1_idx]
    true_id   = q_ids_np[i]

    if top1_id == true_id and len(successful) < 5:
        successful.append({'q_idx': i, 'g_idx': top1_idx, 'sim': top1_sim,
                           'id': true_id})
    elif top1_id != true_id and len(failed) < 3:
        # find correct gallery index
        correct_idxs = np.where(g_ids_np == true_id)[0]
        if len(correct_idxs) > 0:
            failed.append({'q_idx': i, 'g_idx_top1': top1_idx, 'g_idx_correct': correct_idxs[0],
                           'sim': top1_sim, 'id': true_id, 'top1_id': top1_id})

print(f'successful pairs: {len(successful)}')
for p in successful:
    print(f"  id={p['id']:04d}  sim={p['sim']:.4f}")

print(f'failed pairs: {len(failed)}')
for p in failed:
    print(f"  true={p['id']:04d}  retrieved={p['top1_id']:04d}  sim={p['sim']:.4f}")

print(f'pair selection done: {len(successful)} successful, {len(failed)} failed.')


# %%
# build path lookup from dataset
# u1652dataseteval stores file paths in .images (query mode) and .images (gallery mode)
# inspect how paths are stored
print(type(query_dataset))
print(dir(query_dataset))

# %%
# u1652dataseteval stores paths in .images (list of str, one per image)
# .sample_ids has the corresponding label for each index
q_paths = query_dataset.images    # list[str]
g_paths = gallery_dataset.images  # list[str]

def get_path(paths, idx):
    return paths[idx]

print(get_path(q_paths, 0))
print(get_path(g_paths, 0))
print(f'query paths: {len(q_paths)}, gallery paths: {len(g_paths)}')


# %% [markdown]
# ## 7. GradCAM — University-1652
# 

# %%
gradcam = GradCAMExtractor(model_u)

for i, pair in enumerate(successful, 1):
    q_path = get_path(q_paths, pair['q_idx'])
    g_path = get_path(g_paths, pair['g_idx'])

    q_t = load_image(q_path).to(device)
    g_t = load_image(g_path).to(device)

    q_cam, g_cam = gradcam.generate_pair_cam(q_t, g_t)

    sim = pair['sim']
    save_path = f"{RESULTS_DIR}/gradcam/u1652_successful_{i:02d}.png"
    if os.path.exists(save_path):
        print(f'skipping #{i} — already saved.')
        continue
    show_gradcam(q_t, g_t, q_cam, g_cam,
                 title=f"GradCAM — Successful #{i} | id={pair['id']:04d} | sim={sim:.4f}",
                 save_path=save_path)

gradcam.remove_hooks()

# %%
gradcam = GradCAMExtractor(model_u)

for i, pair in enumerate(failed, 1):
    q_path  = get_path(q_paths, pair['q_idx'])
    g_path  = get_path(g_paths, pair['g_idx_top1'])   # wrong gallery

    q_t = load_image(q_path).to(device)
    g_t = load_image(g_path).to(device)

    q_cam, g_cam = gradcam.generate_pair_cam(q_t, g_t)

    save_path = f"{RESULTS_DIR}/gradcam/u1652_failed_{i:02d}.png"
    if os.path.exists(save_path):
        print(f'skipping #{i} — already saved.')
        continue
    show_gradcam(q_t, g_t, q_cam, g_cam,
                 title=f"GradCAM — Failed #{i} | true={pair['id']:04d} retrieved={pair['top1_id']:04d} | sim={pair['sim']:.4f}",
                 save_path=save_path)

gradcam.remove_hooks()

# %% [markdown]
# ## 8. Occlusion Sensitivity — University-1652
# 
# Slower than GradCAM but more model-agnostic.  
# With `patch_size=64, stride=32, img_size=384`: 121 positions → **4 batched forward passes** per image.
# 

# %%
PATCH_SIZE = 64
STRIDE     = 32
FAITH_STEPS = 10

occ = OcclusionSensitivity(model_u, device=torch.device(device))

for i, pair in enumerate(successful, 1):
    q_path = get_path(q_paths, pair['q_idx'])
    g_path = get_path(g_paths, pair['g_idx'])

    q_t = load_image(q_path).to(device)
    g_t = load_image(g_path).to(device)

    print(f'computing sensitivity for successful #{i}...')
    q_map, g_map = occ.compute_pair_sensitivity(q_t, g_t, PATCH_SIZE, STRIDE)

    with torch.no_grad():
        g_emb = model_u(g_t)
        q_emb = model_u(q_t)

    q_faith = occ.compute_faithfulness(q_t, g_emb, q_map, steps=FAITH_STEPS)
    g_faith = occ.compute_faithfulness(g_t, q_emb, g_map, steps=FAITH_STEPS)

    save_path = f"{RESULTS_DIR}/occlusion/u1652_successful_{i:02d}.png"
    if os.path.exists(save_path):
        print(f'skipping #{i} — already saved.')
        continue
    show_occlusion(q_t, g_t, q_map, g_map,
                   q_faith=q_faith, g_faith=g_faith,
                   title=f"Occlusion — Successful #{i} | id={pair['id']:04d} | sim={pair['sim']:.4f}",
                   save_path=save_path)

# %%
for i, pair in enumerate(failed, 1):
    q_path = get_path(q_paths, pair['q_idx'])
    g_path = get_path(g_paths, pair['g_idx_top1'])

    q_t = load_image(q_path).to(device)
    g_t = load_image(g_path).to(device)

    print(f'computing sensitivity for failed #{i}...')
    q_map, g_map = occ.compute_pair_sensitivity(q_t, g_t, PATCH_SIZE, STRIDE)

    with torch.no_grad():
        g_emb = model_u(g_t)
        q_emb = model_u(q_t)

    q_faith = occ.compute_faithfulness(q_t, g_emb, q_map, steps=FAITH_STEPS)
    g_faith = occ.compute_faithfulness(g_t, q_emb, g_map, steps=FAITH_STEPS)

    save_path = f"{RESULTS_DIR}/occlusion/u1652_failed_{i:02d}.png"
    if os.path.exists(save_path):
        print(f'skipping #{i} — already saved.')
        continue
    show_occlusion(q_t, g_t, q_map, g_map,
                   q_faith=q_faith, g_faith=g_faith,
                   title=f"Occlusion — Failed #{i} | true={pair['id']:04d} retrieved={pair['top1_id']:04d}",
                   save_path=save_path)

# %% [markdown]
# ## 9. XAI Pair Selection — VIGOR
# 
# Same process as University-1652 but with VIGOR's VigorDatasetEval.
# 

# %%
# vigor xai pair selection — load features from cache or reuse from eval
VIGOR_SAME_AREA = True
ckpt_vigor = CKPT_VIGOR_SAME if VIGOR_SAME_AREA else CKPT_VIGOR_CROSS

cfg_v = VigorConfig(same_area=VIGOR_SAME_AREA, checkpoint_start=ckpt_vigor)

# reuse precomputed features from eval section if available in memory
# otherwise load from drive cache
tag = 'same' if VIGOR_SAME_AREA else 'cross'
try:
    precomputed_v = precomputed_same if VIGOR_SAME_AREA else precomputed_cross
    ref_feats_v, ref_labels_v, q_feats_v, q_ids_v = precomputed_v
    g_feats_v  = ref_feats_v
    g_ids_v    = ref_labels_v
    print(f'reusing vigor {tag}-area features from memory.')
except NameError:
    cache_file = f'{CACHE_DIR}/vigor_{tag}_features.pt'
    if os.path.exists(cache_file):
        print(f'loading vigor {tag}-area features from drive cache...')
        _c = torch.load(cache_file, map_location='cpu')
        g_feats_v = _c['ref_feats'].to(device)
        g_ids_v   = _c['ref_labels'].to(device)
        q_feats_v = _c['q_feats'].to(device)
        q_ids_v   = _c['q_labels'].to(device)
    else:
        raise RuntimeError(f'no cache found at {cache_file} — run section 4 first.')

# model still needed for gradcam and occlusion
model_v = TimmModel(cfg_v.model, pretrained=True, img_size=cfg_v.img_size)
data_config_v = model_v.get_config()
mean_v, std_v = data_config_v['mean'], data_config_v['std']
state_dict = torch.load(cfg_v.checkpoint_start, map_location='cpu')
model_v.load_state_dict(state_dict, strict=False)
model_v = model_v.to(device).eval()

img_size_v      = cfg_v.img_size
image_size_sat  = (img_size_v, img_size_v)
new_width       = img_size_v * 2
new_height      = int(((1024 - 2 * cfg_v.ground_cutting) / 2048) * new_width)
img_size_ground = (new_height, new_width)

sat_transforms, ground_transforms = get_transforms_val(
    image_size_sat, img_size_ground, mean=mean_v, std=std_v,
    ground_cutting=cfg_v.ground_cutting)

ref_dataset_v = VigorDatasetEval(
    data_folder=VIGOR_DATA, split='test', img_type='reference',
    same_area=VIGOR_SAME_AREA, transforms=sat_transforms)
q_dataset_v = VigorDatasetEval(
    data_folder=VIGOR_DATA, split='test', img_type='query',
    same_area=VIGOR_SAME_AREA, transforms=ground_transforms)

print(f'vigor {tag}-area ready. query: {q_feats_v.shape}, gallery: {g_feats_v.shape}')
print('vigor model ready.')


# %%
# pair selection for vigor
# note: vigor evaluation uses different matching logic (multiple positives per query)
# for xai we just pick top-1 match and check if it's in the positive set
sim_v = (q_feats_v @ g_feats_v.T).cpu().numpy()
q_ids_v_np = q_ids_v.cpu().numpy()
g_ids_v_np = g_ids_v.cpu().numpy()

vigor_successful, vigor_failed = [], []

for i in range(len(q_ids_v_np)):
    sims    = sim_v[i]
    top1_idx = int(np.argmax(sims))
    top1_sim = sims[top1_idx]
    top1_id  = g_ids_v_np[top1_idx]
    true_ids = q_ids_v_np[i]          # shape (4,): [sat, sat_np1, sat_np2, sat_np3]
    true_id  = int(true_ids[0])       # primary positive satellite index

    if top1_id in true_ids and len(vigor_successful) < 5:
        vigor_successful.append({'q_idx': i, 'g_idx': top1_idx, 'sim': top1_sim})
    elif top1_id not in true_ids and len(vigor_failed) < 3:
        correct_idxs = np.where(g_ids_v_np == true_id)[0]
        if len(correct_idxs) > 0:
            vigor_failed.append({'q_idx': i, 'g_idx_top1': top1_idx,
                                  'g_idx_correct': correct_idxs[0], 'sim': top1_sim})

print(f'vigor successful: {len(vigor_successful)}')
print(f'vigor failed:     {len(vigor_failed)}')

print(f'vigor pair selection done: {len(vigor_successful)} successful, {len(vigor_failed)} failed.')


# %% [markdown]
# ## 10. GradCAM — VIGOR
# 

# %%
# path lookup for vigor datasets
print(dir(q_dataset_v))  # inspect to find path attribute

# %%
# adjust attribute name based on above output
# vigordataseteval likely stores paths in idx2ground_path / idx2sat_path
def get_vigor_query_path(dataset, idx):
    return dataset.idx2ground_path[idx]

def get_vigor_gallery_path(dataset, idx):
    return dataset.idx2sat_path[idx]

gradcam_v = GradCAMExtractor(model_v)

split_tag = 'same' if VIGOR_SAME_AREA else 'cross'

for i, pair in enumerate(vigor_successful, 1):
    q_path = get_vigor_query_path(q_dataset_v,   pair['q_idx'])
    g_path = get_vigor_gallery_path(ref_dataset_v, pair['g_idx'])

    q_t = load_image(q_path, img_size=img_size_v).to(device)
    g_t = load_image(g_path, img_size=img_size_v).to(device)

    q_cam, g_cam = gradcam_v.generate_pair_cam(q_t, g_t)

    save_path = f"{RESULTS_DIR}/gradcam/vigor_{split_tag}_successful_{i:02d}.png"
    if os.path.exists(save_path):
        print(f'skipping #{i} — already saved.')
        continue
    show_gradcam(q_t, g_t, q_cam, g_cam,
                 title=f"GradCAM — VIGOR {split_tag} Successful #{i} | sim={pair['sim']:.4f}",
                 save_path=save_path)

gradcam_v.remove_hooks()

# %%
gradcam_v = GradCAMExtractor(model_v)

for i, pair in enumerate(vigor_failed, 1):
    q_path = get_vigor_query_path(q_dataset_v,     pair['q_idx'])
    g_path = get_vigor_gallery_path(ref_dataset_v, pair['g_idx_top1'])

    q_t = load_image(q_path, img_size=img_size_v).to(device)
    g_t = load_image(g_path, img_size=img_size_v).to(device)

    q_cam, g_cam = gradcam_v.generate_pair_cam(q_t, g_t)

    save_path = f"{RESULTS_DIR}/gradcam/vigor_{split_tag}_failed_{i:02d}.png"
    if os.path.exists(save_path):
        print(f'skipping #{i} — already saved.')
        continue
    show_gradcam(q_t, g_t, q_cam, g_cam,
                 title=f"GradCAM — VIGOR {split_tag} Failed #{i} | sim={pair['sim']:.4f}",
                 save_path=save_path)

gradcam_v.remove_hooks()

# %% [markdown]
# ## 11. Occlusion Sensitivity — VIGOR
# 

# %%
occ_v = OcclusionSensitivity(model_v, device=torch.device(device))

for i, pair in enumerate(vigor_successful, 1):
    q_path = get_vigor_query_path(q_dataset_v,   pair['q_idx'])
    g_path = get_vigor_gallery_path(ref_dataset_v, pair['g_idx'])

    q_t = load_image(q_path, img_size=img_size_v).to(device)
    g_t = load_image(g_path, img_size=img_size_v).to(device)

    print(f'vigor occlusion successful #{i}...')
    q_map, g_map = occ_v.compute_pair_sensitivity(q_t, g_t, PATCH_SIZE, STRIDE)

    with torch.no_grad():
        g_emb = model_v(g_t)
        q_emb = model_v(q_t)

    q_faith = occ_v.compute_faithfulness(q_t, g_emb, q_map, steps=FAITH_STEPS)
    g_faith = occ_v.compute_faithfulness(g_t, q_emb, g_map, steps=FAITH_STEPS)

    save_path = f"{RESULTS_DIR}/occlusion/vigor_{split_tag}_successful_{i:02d}.png"
    if os.path.exists(save_path):
        print(f'skipping #{i} — already saved.')
        continue
    show_occlusion(q_t, g_t, q_map, g_map,
                   q_faith=q_faith, g_faith=g_faith,
                   title=f"Occlusion — VIGOR {split_tag} Successful #{i}",
                   save_path=save_path)

# %%
for i, pair in enumerate(vigor_failed, 1):
    q_path = get_vigor_query_path(q_dataset_v,     pair['q_idx'])
    g_path = get_vigor_gallery_path(ref_dataset_v, pair['g_idx_top1'])

    q_t = load_image(q_path, img_size=img_size_v).to(device)
    g_t = load_image(g_path, img_size=img_size_v).to(device)

    print(f'vigor occlusion failed #{i}...')
    q_map, g_map = occ_v.compute_pair_sensitivity(q_t, g_t, PATCH_SIZE, STRIDE)

    with torch.no_grad():
        g_emb = model_v(g_t)
        q_emb = model_v(q_t)

    q_faith = occ_v.compute_faithfulness(q_t, g_emb, q_map, steps=FAITH_STEPS)
    g_faith = occ_v.compute_faithfulness(g_t, q_emb, g_map, steps=FAITH_STEPS)

    save_path = f"{RESULTS_DIR}/occlusion/vigor_{split_tag}_failed_{i:02d}.png"
    if os.path.exists(save_path):
        print(f'skipping #{i} — already saved.')
        continue
    show_occlusion(q_t, g_t, q_map, g_map,
                   q_faith=q_faith, g_faith=g_faith,
                   title=f"Occlusion — VIGOR {split_tag} Failed #{i} | sim={pair['sim']:.4f}",
                   save_path=save_path)


# %% [markdown]
# ## 12. Analysis
# 
# Key questions to answer:
# 
# 1. **GradCAM vs Occlusion Sensitivity** — Do they highlight the same regions? Where do they disagree?
# 2. **Successful vs Failed** — Does the model attend to semantically meaningful regions when it's correct? What does it attend to when it fails?
# 3. **University-1652 vs VIGOR** — Does attention pattern differ between drone↔satellite and street↔satellite?
# 4. **Faithfulness** — How quickly does similarity drop as important regions are masked? Steeper = more faithful explanation.
# 

# %%
# results summary
results = {
    'University-1652': {
        'R@1':  92.66,
        'R@5':  97.69,
        'R@10': 98.24,
        'AP':   93.81,
    },
    'VIGOR Same-Area': {
        'R@1':  None,  # fill after section 4
        'R@5':  None,
        'R@10': None,
    },
    'VIGOR Cross-Area': {
        'R@1':  None,  # fill after section 4
        'R@5':  None,
        'R@10': None,
    },
}

for dataset, metrics in results.items():
    print(f'{dataset}:')
    for k, v in metrics.items():
        val = f'{v}%' if v is not None else 'pending'
        print(f'  {k}: {val}')


# %% [markdown]
# ## Next Steps
# 
# - Fill in results table above after running all eval sections
# - Write analysis comparing GradCAM vs Occlusion heatmaps
# - Write analysis comparing successful vs failed matches
# - Write analysis comparing University-1652 vs VIGOR attention patterns
# - Export figures from `xai_results/` on Drive for the report
# 

# %%
# results are saved directly to drive as they are generated (RESULTS_DIR = drive).
# nothing to copy — just verify the output directory exists.
print(f'xai results directory: {RESULTS_DIR}')
print(f'exists: {os.path.exists(RESULTS_DIR)}')
if os.path.exists(RESULTS_DIR):
    subdirs = [d for d in os.listdir(RESULTS_DIR) if os.path.isdir(os.path.join(RESULTS_DIR, d))]
    for d in subdirs:
        n = len(os.listdir(os.path.join(RESULTS_DIR, d)))
        print(f'  {d}/: {n} files')
print('all xai figures saved. ready for presentation.')


# %% [markdown]
# ## 13. Display Saved Results
# 
# Loads all XAI figures already saved to Drive and displays them inline.
# Run this section any time — no recomputation needed.
# 

# %%
# load and display all saved xai figures from drive
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def display_folder(section_title, folder, filter_prefix=None):
    if not os.path.exists(folder):
        print(f'{section_title}: folder not found.')
        return
    files = sorted(
        f for f in os.listdir(folder)
        if f.endswith('.png') and (filter_prefix is None or f.startswith(filter_prefix))
    )
    if not files:
        print(f'{section_title}: no figures found.')
        return
    print(f'{section_title} — {len(files)} figure(s)')
    for fname in files:
        img = mpimg.imread(os.path.join(folder, fname))
        fig, ax = plt.subplots(figsize=(20, 10))
        ax.imshow(img)
        ax.axis('off')
        ax.set_title(fname, fontsize=9, pad=6)
        plt.tight_layout()
        plt.show()

gradcam_dir  = os.path.join(RESULTS_DIR, 'gradcam')
occlusion_dir = os.path.join(RESULTS_DIR, 'occlusion')

display_folder('GradCAM — University-1652 (successful)',  gradcam_dir,  'u1652_successful')
display_folder('GradCAM — University-1652 (failed)',      gradcam_dir,  'u1652_failed')
display_folder('GradCAM — VIGOR same (successful)',       gradcam_dir,  'vigor_same_successful')
display_folder('GradCAM — VIGOR same (failed)',           gradcam_dir,  'vigor_same_failed')
display_folder('Occlusion — University-1652 (successful)', occlusion_dir, 'u1652_successful')
display_folder('Occlusion — University-1652 (failed)',     occlusion_dir, 'u1652_failed')
display_folder('Occlusion — VIGOR same (successful)',      occlusion_dir, 'vigor_same_successful')
display_folder('Occlusion — VIGOR same (failed)',          occlusion_dir, 'vigor_same_failed')



