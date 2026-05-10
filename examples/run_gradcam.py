"""
GradCAM analysis for Sample4Geo.

Usage:
    python examples/run_gradcam.py \\
        --query  data/U1652/test/query_drone/0001/image-00.jpg \\
        --gallery data/U1652/test/gallery_satellite/0001/satellite.jpg \\
        --checkpoint pretrained/university/convnext_base.fb_in22k_ft_in1k_384/weights_e1_0.9515.pth \\
        --output xai_results/gradcam
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
from xai import GradCAMExtractor


def parse_args():
    parser = argparse.ArgumentParser(description='GradCAM for Sample4Geo')
    parser.add_argument('--query',      required=True,  help='Path to query image')
    parser.add_argument('--gallery',    required=True,  help='Path to gallery image')
    parser.add_argument('--checkpoint', default=None,   help='Path to model checkpoint')
    parser.add_argument('--output',     default='xai_results/gradcam', help='Output directory')
    parser.add_argument('--img-size',   type=int, default=384)
    parser.add_argument('--model-name', default='convnext_base.fb_in22k_ft_in1k_384')
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

    gradcam = GradCAMExtractor(model)
    query_cam, gallery_cam = gradcam.generate_pair_cam(
        query_tensor.to(device), gallery_tensor.to(device)
    )

    gradcam.save_visualization(
        query_tensor, query_cam,
        os.path.join(args.output, 'query_gradcam.png'),
        title='Query — GradCAM',
    )
    gradcam.save_visualization(
        gallery_tensor, gallery_cam,
        os.path.join(args.output, 'gallery_gradcam.png'),
        title='Gallery — GradCAM',
    )

    print(f"\nDone. Results saved to: {args.output}/")


if __name__ == '__main__':
    main()
