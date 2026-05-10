"""
Finds successful and failed query-gallery pairs for XAI analysis.

Runs inference on a random subset of the test set and outputs ready-to-use
run_gradcam.py / run_occlusion.py commands for:
  - N successful matches (R@1 correct)
  - N failed matches (R@1 wrong)

Usage:
    python find_xai_pairs.py
    python find_xai_pairs.py --success 5 --fail 3 --sample 200
"""

import os
import sys
import random
import argparse
import torch
import torch.nn.functional as F
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2

from sample4geo.model import TimmModel


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--success', type=int, default=5,  help='Number of successful pairs')
    parser.add_argument('--fail',    type=int, default=3,  help='Number of failed pairs')
    parser.add_argument('--sample',  type=int, default=300, help='How many queries to evaluate')
    parser.add_argument('--query-dir',   default='data/U1652/test/query_drone')
    parser.add_argument('--gallery-dir', default='data/U1652/test/gallery_satellite')
    parser.add_argument('--checkpoint',  default='pretrained/university/convnext_base.fb_in22k_ft_in1k_384/weights_e1_0.9515.pth')
    parser.add_argument('--img-size',    type=int, default=384)
    return parser.parse_args()


def get_transform(img_size):
    return A.Compose([
        A.Resize(img_size, img_size, interpolation=cv2.INTER_LINEAR_EXACT),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])


def load_image(path, transform, device):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    tensor = transform(image=img)['image'].unsqueeze(0).to(device)
    return tensor


def get_first_image(folder):
    """Return first .jpg/.jpeg image found in a folder."""
    for f in sorted(os.listdir(folder)):
        if f.lower().endswith(('.jpg', '.jpeg', '.png')):
            return os.path.join(folder, f)
    return None


def main():
    args = parse_args()

    device = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load model
    print("Loading model...")
    model = TimmModel('convnext_base.fb_in22k_ft_in1k_384', pretrained=False, img_size=args.img_size)
    state_dict = torch.load(args.checkpoint, map_location='cpu')
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device).eval()

    transform = get_transform(args.img_size)

    # Collect all building IDs
    query_buildings   = sorted(os.listdir(args.query_dir))
    gallery_buildings = sorted(os.listdir(args.gallery_dir))
    gallery_set       = set(gallery_buildings)

    # Only keep queries that have a matching gallery building
    valid_queries = [b for b in query_buildings if b in gallery_set]

    # Sample a subset for speed
    sample_size = min(args.sample, len(valid_queries))
    sampled = random.sample(valid_queries, sample_size)
    print(f"Evaluating {sample_size} queries against {len(gallery_buildings)} gallery buildings...")

    # Pre-compute all gallery embeddings
    print("Computing gallery embeddings...")
    gallery_embeddings = {}
    for building_id in gallery_buildings:
        gallery_folder = os.path.join(args.gallery_dir, building_id)
        img_path = get_first_image(gallery_folder)
        if img_path is None:
            continue
        with torch.no_grad():
            emb = model(load_image(img_path, transform, device))
            emb = F.normalize(emb, dim=-1)
        gallery_embeddings[building_id] = (emb, img_path)

    gallery_ids   = list(gallery_embeddings.keys())
    gallery_stack = torch.cat([gallery_embeddings[b][0] for b in gallery_ids], dim=0)  # (G, D)

    # Evaluate each query
    successful = []
    failed     = []

    print("Evaluating queries...")
    for building_id in sampled:
        query_folder = os.path.join(args.query_dir, building_id)
        query_img_path = get_first_image(query_folder)
        if query_img_path is None:
            continue

        with torch.no_grad():
            q_emb = model(load_image(query_img_path, transform, device))
            q_emb = F.normalize(q_emb, dim=-1)

        sims = F.cosine_similarity(q_emb, gallery_stack, dim=-1)
        top1_idx = sims.argmax().item()
        top1_building = gallery_ids[top1_idx]

        gallery_img_path = gallery_embeddings[top1_building][1]
        correct_gallery_path = gallery_embeddings.get(building_id, (None, None))[1]

        pair = {
            'query_building':    building_id,
            'query_img':         query_img_path,
            'top1_building':     top1_building,
            'top1_gallery_img':  gallery_img_path,
            'correct_gallery_img': correct_gallery_path,
            'similarity':        sims[top1_idx].item(),
        }

        if top1_building == building_id:
            successful.append(pair)
        else:
            failed.append(pair)

    # Sample requested number of pairs
    random.shuffle(successful)
    random.shuffle(failed)
    selected_success = successful[:args.success]
    selected_fail    = failed[:args.fail]

    print(f"\nFound {len(successful)} successful, {len(failed)} failed out of {sample_size} queries.")
    print(f"Selected {len(selected_success)} successful + {len(selected_fail)} failed pairs.\n")

    checkpoint = args.checkpoint

    def print_commands(pairs, label):
        print("=" * 60)
        print(f"  {label}")
        print("=" * 60)
        for i, p in enumerate(pairs, 1):
            print(f"\n--- Pair {i} | Query building: {p['query_building']} | Top-1: {p['top1_building']} | Sim: {p['similarity']:.4f} ---")
            print(f"  Query : {p['query_img']}")
            print(f"  Gallery (top-1 match): {p['top1_gallery_img']}")
            if label.startswith("FAILED") and p['correct_gallery_img']:
                print(f"  Gallery (correct):     {p['correct_gallery_img']}")
            print()
            print(f"  python3 examples/run_gradcam.py \\")
            print(f"      --query {p['query_img']} \\")
            print(f"      --gallery {p['top1_gallery_img']} \\")
            print(f"      --checkpoint {checkpoint} \\")
            print(f"      --output xai_results/gradcam/{label.lower().split()[0]}_{i:02d}")
            print()
            print(f"  python3 examples/run_occlusion.py \\")
            print(f"      --query {p['query_img']} \\")
            print(f"      --gallery {p['top1_gallery_img']} \\")
            print(f"      --checkpoint {checkpoint} \\")
            print(f"      --output xai_results/occlusion/{label.lower().split()[0]}_{i:02d} \\")
            print(f"      --patch-size 64 --stride 16")

    print_commands(selected_success, "SUCCESSFUL matches")
    print_commands(selected_fail,    "FAILED matches")

    # Also save to file
    import sys
    out_path = "xai_pairs.txt"
    orig_stdout = sys.stdout
    with open(out_path, "w") as f:
        sys.stdout = f
        print_commands(selected_success, "SUCCESSFUL matches")
        print_commands(selected_fail,    "FAILED matches")
    sys.stdout = orig_stdout
    print(f"\nAll commands also saved to: {out_path}")


if __name__ == '__main__':
    main()
