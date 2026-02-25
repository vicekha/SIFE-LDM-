#!/usr/bin/env python3
"""
Quick Data Download Script
==========================

Downloads NLP and code datasets quickly using HuggingFace datasets.

Usage:
    python get_data.py --output ./data --nlp_samples 100000 --code_samples 50000
"""

import os
import sys
import json
import argparse
from pathlib import Path
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=str, default='./data', help='Output directory')
    parser.add_argument('--nlp_samples', type=int, default=100000, help='NLP samples to download')
    parser.add_argument('--code_samples', type=int, default=50000, help='Code samples per language')
    parser.add_argument('--languages', type=str, default='python,javascript,java', help='Languages')
    return parser.parse_args()

def install_dependencies():
    """Install required packages."""
    import subprocess
    
    packages = ['datasets', 'tqdm']
    for pkg in packages:
        try:
            __import__(pkg)
        except ImportError:
            print(f"Installing {pkg}...")
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', pkg, '-q'])

def download_wikitext(output_dir, max_samples):
    """Download WikiText-103."""
    from datasets import load_dataset
    
    print("\n📚 Downloading WikiText-103...")
    
    try:
        dataset = load_dataset('wikitext', 'wikitext-103-raw-v1', split='train')
    except Exception as e:
        print(f"Error: {e}")
        return []
    
    texts = []
    for item in tqdm(dataset, desc="Processing WikiText"):
        text = item['text'].strip()
        if len(text) > 50:
            texts.append(text)
        if len(texts) >= max_samples:
            break
    
    print(f"Downloaded {len(texts)} WikiText samples")
    return texts

def download_openwebtext(output_dir, max_samples):
    """Download OpenWebText subset."""
    from datasets import load_dataset
    
    print("\n🌐 Downloading OpenWebText...")
    
    try:
        dataset = load_dataset('openwebtext', split='train', streaming=True)
    except Exception as e:
        print(f"Error loading OpenWebText: {e}")
        return []
    
    texts = []
    count = 0
    for item in tqdm(dataset, desc="Processing OpenWebText", total=max_samples):
        text = item['text'].strip()
        if len(text) > 100:
            texts.append(text[:8000])  # Limit length
            count += 1
            if count >= max_samples:
                break
    
    print(f"Downloaded {len(texts)} OpenWebText samples")
    return texts

def download_python_code(output_dir, max_samples):
    """Download Python code datasets."""
    from datasets import load_dataset
    
    print("\n🐍 Downloading Python code...")
    
    codes = []
    
    # Try CodeParrot first
    try:
        print("  Loading CodeParrot...")
        dataset = load_dataset('codeparrot/codeparrot-clean-train', split='train', streaming=True)
        count = 0
        for item in tqdm(dataset, desc="CodeParrot", total=max_samples//2):
            code = item.get('content', '')
            if len(code) > 50:
                codes.append(code[:6000])
                count += 1
                if count >= max_samples // 2:
                    break
    except Exception as e:
        print(f"  CodeParrot error: {e}")
    
    # Also try CodeAlpaca for variety
    try:
        print("  Loading CodeAlpaca...")
        dataset = load_dataset('sahil2801/CodeAlpaca-20k', split='train')
        for item in tqdm(dataset, desc="CodeAlpaca"):
            code = item.get('output', '')
            if len(code) > 30:
                codes.append(code)
                if len(codes) >= max_samples:
                    break
    except Exception as e:
        print(f"  CodeAlpaca error: {e}")
    
    print(f"Downloaded {len(codes)} Python code samples")
    return codes

def download_javascript_code(output_dir, max_samples):
    """Download JavaScript code."""
    from datasets import load_dataset
    
    print("\n📜 Downloading JavaScript code...")
    
    codes = []
    
    try:
        # Use CodeSearchNet for JavaScript
        dataset = load_dataset('code_search_net', 'javascript', split='train', streaming=True)
        count = 0
        for item in tqdm(dataset, desc="JavaScript", total=max_samples):
            code = item.get('func_code_string', '')
            if len(code) > 50:
                codes.append(code[:6000])
                count += 1
                if count >= max_samples:
                    break
    except Exception as e:
        print(f"  Error: {e}")
    
    print(f"Downloaded {len(codes)} JavaScript samples")
    return codes

def download_java_code(output_dir, max_samples):
    """Download Java code."""
    from datasets import load_dataset
    
    print("\n☕ Downloading Java code...")
    
    codes = []
    
    try:
        dataset = load_dataset('code_search_net', 'java', split='train', streaming=True)
        count = 0
        for item in tqdm(dataset, desc="Java", total=max_samples):
            code = item.get('func_code_string', '')
            if len(code) > 50:
                codes.append(code[:6000])
                count += 1
                if count >= max_samples:
                    break
    except Exception as e:
        print(f"  Error: {e}")
    
    print(f"Downloaded {len(codes)} Java samples")
    return codes

def save_dataset(texts, output_path, name="dataset"):
    """Save dataset with train/val/test split."""
    import random
    random.seed(42)
    random.shuffle(texts)
    
    n = len(texts)
    train_end = int(n * 0.95)
    val_end = int(n * 0.975)
    
    splits = {
        'train': texts[:train_end],
        'val': texts[train_end:val_end],
        'test': texts[val_end:]
    }
    
    stats = {}
    for split_name, split_texts in splits.items():
        path = f"{output_path}_{split_name}.txt"
        with open(path, 'w', encoding='utf-8') as f:
            for text in split_texts:
                # Single line per sample (escape newlines)
                f.write(text.replace('\n', '\\n') + '\n')
        stats[split_name] = len(split_texts)
        print(f"  Saved {split_name}: {len(split_texts)} samples to {path}")
    
    return stats

def main():
    args = parse_args()
    
    # Install dependencies
    print("Installing dependencies...")
    install_dependencies()
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    all_stats = {}
    
    # Download NLP data
    print("\n" + "="*60)
    print("DOWNLOADING NLP DATASETS")
    print("="*60)
    
    nlp_texts = []
    
    # WikiText
    wikitext = download_wikitext(args.output, args.nlp_samples // 2)
    nlp_texts.extend(wikitext)
    
    # OpenWebText
    openwebtext = download_openwebtext(args.output, args.nlp_samples // 2)
    nlp_texts.extend(openwebtext)
    
    # Remove duplicates and save
    nlp_texts = list(set(nlp_texts))
    
    nlp_dir = os.path.join(args.output, 'nlp')
    os.makedirs(nlp_dir, exist_ok=True)
    
    print(f"\n💾 Saving NLP dataset ({len(nlp_texts)} total samples)...")
    nlp_stats = save_dataset(nlp_texts, os.path.join(nlp_dir, 'nlp'), 'NLP')
    all_stats['nlp'] = nlp_stats
    
    # Download code data
    print("\n" + "="*60)
    print("DOWNLOADING CODE DATASETS")
    print("="*60)
    
    languages = [l.strip() for l in args.languages.split(',')]
    code_data = {}
    
    for lang in languages:
        if lang == 'python':
            code_data['python'] = download_python_code(args.output, args.code_samples)
        elif lang == 'javascript':
            code_data['javascript'] = download_javascript_code(args.output, args.code_samples)
        elif lang == 'java':
            code_data['java'] = download_java_code(args.output, args.code_samples)
    
    # Save code data
    code_dir = os.path.join(args.output, 'code')
    os.makedirs(code_dir, exist_ok=True)
    
    all_codes = []
    code_stats = {}
    
    for lang, codes in code_data.items():
        if codes:
            codes = list(set(codes))
            all_codes.extend(codes)
            
            lang_dir = os.path.join(code_dir, lang)
            os.makedirs(lang_dir, exist_ok=True)
            
            print(f"\n💾 Saving {lang} code ({len(codes)} samples)...")
            lang_stats = save_dataset(codes, os.path.join(lang_dir, lang), lang)
            code_stats[lang] = lang_stats
    
    # Save combined code
    combined_dir = os.path.join(code_dir, 'combined')
    os.makedirs(combined_dir, exist_ok=True)
    
    print(f"\n💾 Saving combined code ({len(all_codes)} samples)...")
    combined_stats = save_dataset(all_codes, os.path.join(combined_dir, 'all_code'), 'Combined Code')
    code_stats['combined'] = combined_stats
    
    all_stats['code'] = code_stats
    
    # Save statistics
    stats_path = os.path.join(args.output, 'stats.json')
    with open(stats_path, 'w') as f:
        json.dump(all_stats, f, indent=2)
    
    # Print summary
    print("\n" + "="*60)
    print("DOWNLOAD COMPLETE!")
    print("="*60)
    
    print(f"\n📁 Data saved to: {args.output}")
    print(f"📊 Statistics saved to: {stats_path}")
    
    print("\n📈 Dataset Summary:")
    print("-" * 40)
    
    if 'nlp' in all_stats:
        print(f"\nNLP Dataset:")
        for split, count in all_stats['nlp'].items():
            print(f"  {split}: {count:,} samples")
    
    if 'code' in all_stats:
        print(f"\nCode Datasets:")
        for lang, stats in all_stats['code'].items():
            if isinstance(stats, dict):
                total = sum(stats.values())
                print(f"  {lang}: {total:,} samples")

if __name__ == '__main__':
    main()
