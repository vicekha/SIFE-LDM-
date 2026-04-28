#!/usr/bin/env python3
"""
SIFE-LDM Data Preparation script (v3.0 - dual text+vision)
==========================================================

Downloads and organizes datasets for training both text and vision models.
"""

import argparse
import os
import sys

def parse_args():
    parser = argparse.ArgumentParser(description='Prepare data for SIFE-LDM')
    parser.add_argument('--mode', type=str, default='text', choices=['text', 'vision'])
    parser.add_argument('--image_dataset', type=str, default='cifar10',
                        help='HuggingFace image dataset name (e.g. cifar10, imagenet-1k)')
    parser.add_argument('--num_classes', type=int, default=10)
    parser.add_argument('--output_dir', type=str, default='data/images/')
    return parser.parse_args()

def load_hf_images(dataset_name, output_dir, limit=1000):
    try:
        from datasets import load_dataset
        from PIL import Image
    except ImportError:
        print("Error: datasets and PIL required for vision mode (pip install datasets pillow)")
        return

    print(f"Loading {dataset_name} from HuggingFace...")
    ds = load_dataset(dataset_name, split='train', streaming=True)
    
    os.makedirs(output_dir, exist_ok=True)
    count = 0
    for i, item in enumerate(ds):
        if i >= limit: break
        
        img = item['img'] if 'img' in item else item['image']
        label = item['label'] if 'label' in item else 0
        
        class_dir = os.path.join(output_dir, str(label))
        os.makedirs(class_dir, exist_ok=True)
        img.save(os.path.join(class_dir, f"{i}.png"))
        count += 1
        if count % 100 == 0: print(f"Saved {count} images...")

    print(f"Finished. Saved {count} images to {output_dir}")

def main():
    args = parse_args()
    if args.mode == 'vision':
        load_hf_images(args.image_dataset, args.output_dir)
    else:
        print("Text mode: Use standard text data loading (not implemented in this helper yet).")

if __name__ == '__main__':
    main()
