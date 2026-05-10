"""
Occlusion Sensitivity analysis for Sample4Geo.

Usage:
    python examples/run_occlusion.py \\
        --query  data/U1652/test/query_drone/0001/image-00.jpg \\
        --gallery data/U1652/test/gallery_satellite/0001/satellite.jpg \\
        --checkpoint pretrained/university/convnext_base.fb_in22k_ft_in1k_384/weights_e1_0.9515.pth \\
        --output xai_results/occlusion \\
        --patch-size 64 --stride 16 --faithfulness-steps 10
"""

import argparse
import os
import sys

import cv2
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from sample4geo.model import TimmModel
from xai import OcclusionSensitivity


def parse_args():
    parser = argparse.ArgumentParser(description='Occlusion Sensitivity for Sample4Geo')
    parser.add_argument('--query',      required=True,  help='Path to query image')
    parser.add_argument('--gallery',    required=True,  help='Path to gallery image')
    parser.add_argument('--checkpoint', default=None,   help='Path to model checkpoint')
    parser.add_argument('--output',     default='xai_results/occlusion', help='Output directory')
    parser.add_argument('--img-size',   type=int, default=384)
    parser.add_argument('--model-name', default='convnext_base.fb_in22k_ft_in1k_384')
    parser.add_argument('--patch-size', type=int, default=64,  help='Occlusion patch size')
    parser.add_argument('--stride',     type=int, default=16,  help='Patch stride')
    parser.add_argument('--faithfulness-steps', type=int, default=10,
                        help='Number of masking steps for faithfulness curve')
    return parser.parse_args()


def load_model(checkpoint, model_name, img_size, device):
    model = TimmModel(model_name, pretrained=True, img_size=img_size)
    if checkpoint and os.path.exists(checkpoint):
        state_dict = torch.load(checkpoint, map_location=device)
        model.load_state_dict(state_dict, strict=False)
        print(f"Loaded checkpoint: {checkpoint}")
    else:
        print("No checkpoint found, using pretrained weights.")
    return model.to(device).eval()


def load_image(path, img_size):
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    transform = A.Compose([
        A.Resize(img_size, img_size, interpolation=cv2.INTER_LINEAR_EXACT),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])
    return transform(image=img)['image'].unsqueeze(0)


def main():
    args = parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    model = load_model(args.checkpoint, args.model_name, args.img_size, device)

    query_tensor   = load_image(args.query,   args.img_size)
    gallery_tensor = load_image(args.gallery, args.img_size)

    os.makedirs(args.output, exist_ok=True)

    occ = OcclusionSensitivity(model, device=torch.device(device))

    print(f"Computing sensitivity maps (patch={args.patch_size}, stride={args.stride})...")
    query_map, gallery_map = occ.compute_pair_sensitivity(
        query_tensor.to(device), gallery_tensor.to(device),
        patch_size=args.patch_size,
        stride=args.stride,
    )

    with torch.no_grad():
        gallery_emb = model(gallery_tensor.to(device))
        query_emb   = model(query_tensor.to(device))

    print("Computing faithfulness curves...")
    q_fractions, q_scores = occ.compute_faithfulness(
        query_tensor, gallery_emb, query_map, steps=args.faithfulness_steps
    )
    g_fractions, g_scores = occ.compute_faithfulness(
        gallery_tensor, query_emb, gallery_map, steps=args.faithfulness_steps
    )

    occ.save_visualization(
        query_tensor, query_map,
        os.path.join(args.output, 'query_occlusion.png'),
        title='Query — Occlusion Sensitivity',
        faithfulness_data=(q_fractions, q_scores),
    )
    occ.save_visualization(
        gallery_tensor, gallery_map,
        os.path.join(args.output, 'gallery_occlusion.png'),
        title='Gallery — Occlusion Sensitivity',
        faithfulness_data=(g_fractions, g_scores),
    )

    # Print faithfulness summary
    print(f"\nFaithfulness summary (query):")
    for frac, score in zip(q_fractions, q_scores):
        print(f"  {frac*100:5.1f}% masked → similarity {score:.4f}")

    print(f"\nDone. Results saved to: {args.output}/")


if __name__ == '__main__':
    main()
