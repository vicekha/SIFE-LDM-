#!/usr/bin/env python3
"""
Download code datasets from open sources (no auth required).
"""

import os
import sys
from tqdm import tqdm

try:
    from datasets import load_dataset
except ImportError:
    os.system(f'{sys.executable} -m pip install datasets -q')
    from datasets import load_dataset

def save_codes(codes, output_path):
    """Save codes with train/val/test split."""
    import random
    random.seed(42)
    random.shuffle(codes)
    codes = list(set(codes))
    
    n = len(codes)
    if n == 0:
        return {'train': 0, 'val': 0, 'test': 0}
    
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
    output_dir = '/home/z/my-project/download/sife-ldm/data/code'
    os.makedirs(output_dir, exist_ok=True)
    
    all_codes = []
    all_stats = {}
    
    # 1. MBPP - Python problems
    print("\n📥 Downloading MBPP (Python problems)...")
    try:
        codes = []
        for split in ['train', 'test', 'validation']:
            try:
                dataset = load_dataset('mbpp', split=split)
                for item in tqdm(dataset, desc=f"MBPP {split}"):
                    code = item.get('code', '')
                    if code and len(code) > 20:
                        codes.append(code)
            except:
                pass
        all_codes.extend(codes)
        print(f"  Downloaded {len(codes)} MBPP samples")
    except Exception as e:
        print(f"  Error: {e}")
    
    # 2. HumanEval - Python functions
    print("\n📥 Downloading HumanEval (Python functions)...")
    try:
        dataset = load_dataset('openai_humaneval', split='test')
        codes = []
        for item in tqdm(dataset, desc="HumanEval"):
            code = item.get('prompt', '') + item.get('canonical_solution', '')
            if code and len(code) > 20:
                codes.append(code)
        all_codes.extend(codes)
        print(f"  Downloaded {len(codes)} HumanEval samples")
    except Exception as e:
        print(f"  Error: {e}")
    
    # 3. APPS - Python programming problems
    print("\n📥 Downloading APPS (Python problems)...")
    try:
        for split in ['train', 'test']:
            try:
                dataset = load_dataset('codeparrot/apps', split=split, streaming=True, trust_remote_code=True)
                codes = []
                count = 0
                for item in tqdm(dataset, desc=f"APPS {split}", total=5000):
                    code = item.get('solutions', '')
                    if code and len(code) > 50:
                        codes.append(code[:8000])
                        count += 1
                        if count >= 5000:
                            break
                all_codes.extend(codes)
                print(f"  Downloaded {len(codes)} APPS {split} samples")
            except Exception as e:
                print(f"  APPS {split} error: {e}")
    except Exception as e:
        print(f"  Error: {e}")
    
    # 4. CodeAlpaca
    print("\n📥 Downloading CodeAlpaca...")
    try:
        dataset = load_dataset('sahil2801/CodeAlpaca-20k', split='train')
        codes = []
        for item in tqdm(dataset, desc="CodeAlpaca"):
            code = item.get('output', '')
            if code and len(code) > 20:
                codes.append(code)
        all_codes.extend(codes)
        print(f"  Downloaded {len(codes)} CodeAlpaca samples")
    except Exception as e:
        print(f"  Error: {e}")
    
    # 5. Python Code Instructions
    print("\n📥 Downloading Python Code Instructions...")
    try:
        dataset = load_dataset('iamtarun/python_code_instructions_18k_alpaca', split='train')
        codes = []
        for item in tqdm(dataset, desc="Python Instructions"):
            code = item.get('output', '')
            if code and len(code) > 20:
                codes.append(code)
        all_codes.extend(codes)
        print(f"  Downloaded {len(codes)} Python Instructions samples")
    except Exception as e:
        print(f"  Error: {e}")
    
    # 6. CodeContests - Programming contests
    print("\n📥 Downloading CodeContests...")
    try:
        for split in ['train', 'valid', 'test']:
            try:
                dataset = load_dataset('deepmind/code_contests', split=split, streaming=True, trust_remote_code=True)
                count = 0
                codes = []
                for item in tqdm(dataset, desc=f"CodeContests {split}", total=3000):
                    solutions = item.get('solutions', {})
                    if solutions:
                        for sol in solutions.get('solution', []):
                            if sol and len(sol) > 50:
                                codes.append(sol[:6000])
                                count += 1
                                if count >= 3000:
                                    break
                    if count >= 3000:
                        break
                all_codes.extend(codes)
                print(f"  Downloaded {len(codes)} CodeContests {split} samples")
            except:
                pass
    except Exception as e:
        print(f"  Error: {e}")
    
    # 7. DSPy - Code examples
    print("\n📥 Downloading DSPy Code Examples...")
    try:
        dataset = load_dataset('databricks/databricks-dolly-15k', split='train')
        codes = []
        for item in tqdm(dataset, desc="Dolly"):
            code = item.get('response', '')
            if 'def ' in code or 'class ' in code or 'import ' in code:
                if len(code) > 30:
                    codes.append(code)
        all_codes.extend(codes)
        print(f"  Downloaded {len(codes)} Dolly code samples")
    except Exception as e:
        print(f"  Error: {e}")
    
    # 8. Additional Python code
    print("\n📥 Downloading additional Python code...")
    try:
        dataset = load_dataset('Nicobobo/leetcode-python', split='train')
        codes = []
        for item in tqdm(dataset, desc="LeetCode"):
            code = item.get('code', '') or item.get('solution', '') or item.get('text', '')
            if code and len(code) > 30:
                codes.append(code)
        all_codes.extend(codes)
        print(f"  Downloaded {len(codes)} LeetCode samples")
    except Exception as e:
        print(f"  Error: {e}")
    
    # Save all code
    print(f"\n💾 Saving combined code dataset ({len(all_codes)} total samples)...")
    combined_path = os.path.join(output_dir, 'combined', 'all_code_full')
    stats = save_codes(all_codes, combined_path)
    all_stats['combined'] = stats
    
    # Print summary
    print("\n" + "="*60)
    print("DOWNLOAD COMPLETE!")
    print("="*60)
    
    total = sum(sum(s.values()) for s in all_stats.values())
    print(f"\nTotal code samples: {total:,}")
    
    # Show file sizes
    import glob
    print("\n📁 Data files:")
    for f in sorted(glob.glob(f"{output_dir}/**/*.txt", recursive=True)):
        size = os.path.getsize(f) / (1024 * 1024)  # MB
        print(f"  {f}: {size:.2f} MB")
    
    return all_stats

if __name__ == '__main__':
    main()
