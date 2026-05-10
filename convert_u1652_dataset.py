import os
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm
import argparse

def convert_u1652_dataset(output_dir="data/U1652", split_limit=None):
    """
    Convert HuggingFace University-1652 dataset to folder structure
    
    Args:
        output_dir: Output directory path
        split_limit: Limit number of samples per split (for testing, None for full dataset)
    """
    
    print("Loading University-1652 dataset from HuggingFace...")
    dataset = load_dataset("layumi/university-1652")
    
    print("\nDataset loaded successfully!")
    print(f"Available splits: {list(dataset.keys())}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    stats = {
        'train': {'satellite': 0, 'drone': 0},
        'test': {'query_drone': 0, 'gallery_satellite': 0, 'query_satellite': 0, 'gallery_drone': 0}
    }
    
    for split_name in ['train', 'test']:
        print(f"\n{'='*60}")
        print(f"Processing {split_name} split...")
        print(f"{'='*60}")
        
        split_data = dataset[split_name]
        total = len(split_data)
        if split_limit:
            total = min(total, split_limit)
        
        for idx in tqdm(range(total), desc=f"Processing {split_name}"):
            sample = split_data[idx]
            
            img = sample['image']
            building_id = sample['building_id']
            view_type = sample['view_type']
            split_type = sample.get('split_type', split_name)
            
            if isinstance(building_id, str):
                building_id = building_id.zfill(4)
            else:
                building_id = str(building_id).zfill(4)
            
            if split_name == 'train':
                if view_type == 'satellite':
                    folder = f"{output_dir}/train/satellite/{building_id}"
                    stats['train']['satellite'] += 1
                elif view_type == 'drone':
                    folder = f"{output_dir}/train/drone/{building_id}"
                    stats['train']['drone'] += 1
                else:
                    continue
                    
                os.makedirs(folder, exist_ok=True)
                
                if view_type == 'satellite':
                    save_path = f"{folder}/satellite.jpg"
                else:
                    existing = len([f for f in os.listdir(folder) if f.endswith('.jpg')])
                    save_path = f"{folder}/image-{existing:02d}.jpg"
                
            else:  # test split
                if view_type == 'drone':
                    if split_type == 'query':
                        folder = f"{output_dir}/test/query_drone/{building_id}"
                        stats['test']['query_drone'] += 1
                    elif split_type == 'gallery':
                        folder = f"{output_dir}/test/gallery_drone/{building_id}"
                        stats['test']['gallery_drone'] += 1
                    else:
                        folder = f"{output_dir}/test/query_drone/{building_id}"
                        stats['test']['query_drone'] += 1
                elif view_type == 'satellite':
                    if split_type == 'query':
                        folder = f"{output_dir}/test/query_satellite/{building_id}"
                        stats['test']['query_satellite'] += 1
                    elif split_type == 'gallery':
                        folder = f"{output_dir}/test/gallery_satellite/{building_id}"
                        stats['test']['gallery_satellite'] += 1
                    else:
                        folder = f"{output_dir}/test/gallery_satellite/{building_id}"
                        stats['test']['gallery_satellite'] += 1
                else:
                    continue
                
                os.makedirs(folder, exist_ok=True)
                
                existing = len([f for f in os.listdir(folder) if f.endswith('.jpg')])
                save_path = f"{folder}/image-{existing:02d}.jpg"
            
            if not os.path.exists(save_path):
                if isinstance(img, str):
                    img = Image.open(img)
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                img.save(save_path, 'JPEG', quality=95)
    
    print("\n" + "="*60)
    print("CONVERSION COMPLETED!")
    print("="*60)
    print("\nDataset Statistics:")
    print("-"*60)
    for split_name, views in stats.items():
        print(f"\n{split_name.upper()}:")
        for view, count in views.items():
            print(f"  {view}: {count} images")
    
    print(f"\n✓ Dataset saved to: {output_dir}")
    print("\nFolder structure:")
    print(f"{output_dir}/")
    print("├── train/")
    print("│   ├── satellite/{building_id}/satellite.jpg")
    print("│   └── drone/{building_id}/image-XX.jpg")
    print("└── test/")
    print("    ├── query_drone/{building_id}/image-XX.jpg")
    print("    ├── gallery_satellite/{building_id}/image-XX.jpg")
    print("    ├── query_satellite/{building_id}/image-XX.jpg")
    print("    └── gallery_drone/{building_id}/image-XX.jpg")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert University-1652 dataset from HuggingFace to folder structure')
    parser.add_argument('--output', type=str, default='data/U1652', help='Output directory path')
    parser.add_argument('--limit', type=int, default=None, help='Limit number of samples (for testing)')
    
    args = parser.parse_args()
    
    convert_u1652_dataset(output_dir=args.output, split_limit=args.limit)
