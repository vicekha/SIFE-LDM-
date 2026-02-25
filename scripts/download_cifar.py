#!/usr/bin/env python3
"""
SIFE-Vision: CIFAR-100 Downloader
==================================

Downloads the CIFAR-100 dataset and prepares it for SIFE visual training.
Saves images as NumPy arrays for efficient loading.
"""

import os
import numpy as np
from tqdm import tqdm

def download_cifar():
    try:
        from datasets import load_dataset
    except ImportError:
        print("Installing datasets library...")
        os.system('pip install datasets -q')
        from datasets import load_dataset

    print("\nDownloading CIFAR-100...")
    dataset = load_dataset('cifar100', split='train')
    
    # CIFAR-100 comes with 'img' (PIL) and 'fine_label'
    images = []
    labels = []
    
    print("Processing images...")
    for item in tqdm(dataset):
        # Convert PIL to numpy and normalize to [0, 1]
        img = np.array(item['img']).astype(np.float32) / 255.0
        images.append(img)
        labels.append(item['fine_label'])
        
    images = np.stack(images)
    labels = np.array(labels)
    
    # Save to data directory
    os.makedirs('./data/cifar100', exist_ok=True)
    np.save('./data/cifar100/train_images.npy', images)
    np.save('./data/cifar100/train_labels.npy', labels)
    
    print(f"\nSaved {len(images)} images to ./data/cifar100/")
    print(f"Image shape: {images.shape}")

    # Also download test set
    print("\nDownloading CIFAR-100 Test Set...")
    test_dataset = load_dataset('cifar100', split='test')
    test_images = []
    test_labels = []
    
    for item in tqdm(test_dataset):
        img = np.array(item['img']).astype(np.float32) / 255.0
        test_images.append(img)
        test_labels.append(item['fine_label'])
        
    test_images = np.stack(test_images)
    test_labels = np.array(test_labels)
    
    np.save('./data/cifar100/test_images.npy', test_images)
    np.save('./data/cifar100/test_labels.npy', test_labels)
    print(f"Saved {len(test_images)} test images.")

if __name__ == "__main__":
    download_cifar()
