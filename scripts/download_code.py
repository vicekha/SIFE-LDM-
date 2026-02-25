#!/usr/bin/env python3
"""
Download additional code datasets from alternative sources.
"""

import os
import sys
from tqdm import tqdm

# Install datasets if needed
try:
    from datasets import load_dataset
except ImportError:
    os.system(f'{sys.executable} -m pip install datasets -q')
    from datasets import load_dataset

def download_more_python(output_path, max_samples=20000):
    """Download more Python code from multiple sources."""
    codes = []
    
    # MBPP - Python problems
    print("\n📥 Downloading MBPP (Python problems)...")
    try:
        dataset = load_dataset('mbpp', split='train')
        for item in tqdm(dataset, desc="MBPP"):
            code = item.get('code', '')
            if code and len(code) > 20:
                codes.append(code)
            if len(codes) >= max_samples // 3:
                break
    except Exception as e:
        print(f"  MBPP error: {e}")
    
    # HumanEval - Python functions
    print("\n📥 Downloading HumanEval (Python functions)...")
    try:
        dataset = load_dataset('openai_humaneval', split='test')
        for item in tqdm(dataset, desc="HumanEval"):
            code = item.get('prompt', '') + item.get('canonical_solution', '')
            if code and len(code) > 20:
                codes.append(code)
    except Exception as e:
        print(f"  HumanEval error: {e}")
    
    # GitHub Code - Python subset
    print("\n📥 Downloading GitHub Python code...")
    try:
        dataset = load_dataset('m-browser/github-python', split='train', streaming=True)
        count = 0
        for item in tqdm(dataset, desc="GitHub Python", total=max_samples//3):
            code = item.get('text', '') or item.get('content', '') or item.get('code', '')
            if code and len(code) > 50:
                codes.append(code[:5000])
                count += 1
                if count >= max_samples // 3:
                    break
    except Exception as e:
        print(f"  GitHub Python error: {e}")
    
    # CodeGeneration - Various
    print("\n📥 Downloading CodeGeneration dataset...")
    try:
        dataset = load_dataset('Fsoft-AIC/CodeGeneration', split='train', streaming=True)
        count = 0
        for item in tqdm(dataset, desc="CodeGen", total=max_samples//3):
            code = item.get('code', '') or item.get('output', '')
            if code and len(code) > 30:
                codes.append(code[:5000])
                count += 1
                if count >= max_samples // 3:
                    break
    except Exception as e:
        print(f"  CodeGeneration error: {e}")
    
    # BigCode - The Stack small
    print("\n📥 Downloading BigCode Python...")
    try:
        dataset = load_dataset('bigcode/the-stack-smol', split='train', streaming=True)
        count = 0
        for item in tqdm(dataset, desc="BigCode", total=max_samples//2):
            code = item.get('content', '')
            if code and len(code) > 50:
                codes.append(code[:5000])
                count += 1
                if count >= max_samples // 2:
                    break
    except Exception as e:
        print(f"  BigCode error: {e}")
    
    print(f"\n✅ Downloaded {len(codes)} additional Python samples")
    return codes

def download_javascript_code(output_path, max_samples=15000):
    """Download JavaScript code."""
    codes = []
    
    print("\n📥 Downloading JavaScript code...")
    
    # Try multiple sources
    sources = [
        ('codeparrot/github-code', {'languages': ['JavaScript']}),
        ('bigcode/the-stack-smol', {}),
    ]
    
    for source_name, kwargs in sources:
        try:
            print(f"  Trying {source_name}...")
            dataset = load_dataset(source_name, split='train', streaming=True, **kwargs)
            count = 0
            for item in tqdm(dataset, desc=f"JS from {source_name}", total=max_samples):
                code = item.get('content', '') or item.get('code', '')
                if code and len(code) > 50 and ('function' in code or 'const' in code or 'let' in code):
                    codes.append(code[:5000])
                    count += 1
                    if count >= max_samples:
                        break
            if count >= max_samples:
                break
        except Exception as e:
            print(f"  Error with {source_name}: {e}")
            continue
    
    print(f"\n✅ Downloaded {len(codes)} JavaScript samples")
    return codes

def download_java_code(output_path, max_samples=15000):
    """Download Java code."""
    codes = []
    
    print("\n📥 Downloading Java code...")
    
    try:
        dataset = load_dataset('bigcode/the-stack-smol', split='train', streaming=True)
        count = 0
        for item in tqdm(dataset, desc="Java", total=max_samples*3):
            lang = item.get('lang', '') or item.get('language', '')
            code = item.get('content', '')
            
            if lang.lower() == 'java' and code and len(code) > 50:
                codes.append(code[:5000])
                count += 1
                if count >= max_samples:
                    break
    except Exception as e:
        print(f"  Error: {e}")
    
    print(f"\n✅ Downloaded {len(codes)} Java samples")
    return codes

def download_cpp_code(output_path, max_samples=15000):
    """Download C++ code."""
    codes = []
    
    print("\n📥 Downloading C++ code...")
    
    try:
        dataset = load_dataset('bigcode/the-stack-smol', split='train', streaming=True)
        count = 0
        for item in tqdm(dataset, desc="C++", total=max_samples*5):
            lang = item.get('lang', '') or item.get('language', '')
            code = item.get('content', '')
            
            if lang.lower() in ['c++', 'cpp', 'c'] and code and len(code) > 50:
                codes.append(code[:5000])
                count += 1
                if count >= max_samples:
                    break
    except Exception as e:
        print(f"  Error: {e}")
    
    print(f"\n✅ Downloaded {len(codes)} C/C++ samples")
    return codes

def save_codes(codes, output_path, name):
    """Save codes with train/val/test split."""
    import random
    random.seed(42)
    random.shuffle(codes)
    
    # Remove duplicates
    codes = list(set(codes))
    
    n = len(codes)
    train_end = int(n * 0.95)
    val_end = int(n * 0.975)
    
    splits = {
        'train': codes[:train_end],
        'val': codes[train_end:val_end],
        'test': codes[val_end:]
    }
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    stats = {}
    for split_name, split_codes in splits.items():
        path = f"{output_path}_{split_name}.txt"
        with open(path, 'w', encoding='utf-8') as f:
            for code in split_codes:
                f.write(code.replace('\n', '\\n') + '\n')
        stats[split_name] = len(split_codes)
        print(f"  Saved {split_name}: {len(split_codes)} samples")
    
    return stats

def main():
    output_dir = './data/code'
    os.makedirs(output_dir, exist_ok=True)
    
    all_stats = {}
    all_codes = []
    
    # Download Python
    python_codes = download_more_python(output_dir, max_samples=20000)
    if python_codes:
        all_codes.extend(python_codes)
        print(f"\n💾 Saving Python code...")
        stats = save_codes(python_codes, os.path.join(output_dir, 'python', 'python_extra'), 'python')
        all_stats['python'] = stats
    
    # Download JavaScript
    js_codes = download_javascript_code(output_dir, max_samples=10000)
    if js_codes:
        all_codes.extend(js_codes)
        print(f"\n💾 Saving JavaScript code...")
        stats = save_codes(js_codes, os.path.join(output_dir, 'javascript', 'javascript'), 'javascript')
        all_stats['javascript'] = stats
    
    # Download Java
    java_codes = download_java_code(output_dir, max_samples=10000)
    if java_codes:
        all_codes.extend(java_codes)
        print(f"\n💾 Saving Java code...")
        stats = save_codes(java_codes, os.path.join(output_dir, 'java', 'java'), 'java')
        all_stats['java'] = stats
    
    # Download C++
    cpp_codes = download_cpp_code(output_dir, max_samples=8000)
    if cpp_codes:
        all_codes.extend(cpp_codes)
        print(f"\n💾 Saving C++ code...")
        stats = save_codes(cpp_codes, os.path.join(output_dir, 'cpp', 'cpp'), 'cpp')
        all_stats['cpp'] = stats
    
    # Save combined
    print(f"\n💾 Saving combined code ({len(all_codes)} total)...")
    combined_stats = save_codes(all_codes, os.path.join(output_dir, 'combined', 'all_code_extra'), 'combined')
    all_stats['combined'] = combined_stats
    
    print("\n" + "="*60)
    print("CODE DOWNLOAD COMPLETE!")
    print("="*60)
    
    total = sum(sum(s.values()) for s in all_stats.values())
    print(f"\nTotal code samples: {total:,}")
    
    return all_stats

if __name__ == '__main__':
    main()
