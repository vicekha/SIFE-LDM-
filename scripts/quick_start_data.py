#!/usr/bin/env python3
"""
Quick Start Dataset Setup for SIFE-LDM
======================================

Downloads manageable-sized datasets for quick training.
Perfect for testing on Google TPU free tier.

Usage:
    python quick_start_data.py --mode nlp        # Download NLP datasets (~500MB)
    python quick_start_data.py --mode code       # Download code datasets (~200MB)
    python quick_start_data.py --mode both       # Download both (~700MB)
    python quick_start_data.py --mode minimal    # Minimal test datasets (~50MB)

Author: SIFE-LDM Research Team
License: MIT
"""

import argparse
import os
import sys
import json
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def download_wikitext2(output_dir: str):
    """Download WikiText-2 (small, ~10MB) - good for testing."""
    print("\n📥 Downloading WikiText-2...")
    
    try:
        from datasets import load_dataset
        
        ds = load_dataset('wikitext', 'wikitext-2-raw-v1')
        
        for split_name in ['train', 'validation', 'test']:
            with open(os.path.join(output_dir, f'wikitext2_{split_name}.txt'), 'w', encoding='utf-8') as f:
                for sample in ds[split_name]:
                    text = sample['text']
                    if text.strip():
                        f.write(text + '\n')
        
        print("   ✅ WikiText-2 downloaded")
        return True
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False


def download_wikitext103(output_dir: str):
    """Download WikiText-103 (medium, ~500MB) - good for training."""
    print("\n📥 Downloading WikiText-103...")
    
    try:
        from datasets import load_dataset
        
        ds = load_dataset('wikitext', 'wikitext-103-raw-v1')
        
        for split_name in ['train', 'validation', 'test']:
            print(f"   Processing {split_name} split...")
            with open(os.path.join(output_dir, f'wikitext103_{split_name}.txt'), 'w', encoding='utf-8') as f:
                for sample in ds[split_name]:
                    text = sample['text']
                    if text.strip():
                        f.write(text + '\n')
        
        print("   ✅ WikiText-103 downloaded")
        return True
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False


def download_alpaca(output_dir: str):
    """Download Alpaca dataset (instruction tuning, ~20MB)."""
    print("\n📥 Downloading Alpaca dataset...")
    
    try:
        from datasets import load_dataset
        
        ds = load_dataset('tatsu-lab/alpaca')
        
        with open(os.path.join(output_dir, 'alpaca_train.txt'), 'w', encoding='utf-8') as f:
            for sample in ds['train']:
                instruction = sample.get('instruction', '')
                input_text = sample.get('input', '')
                output_text = sample.get('output', '')
                
                if input_text:
                    f.write(f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n{output_text}\n\n")
                else:
                    f.write(f"### Instruction:\n{instruction}\n\n### Response:\n{output_text}\n\n")
        
        print("   ✅ Alpaca downloaded")
        return True
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False


def download_code_alpaca(output_dir: str):
    """Download Code Alpaca dataset (~20MB)."""
    print("\n📥 Downloading Code Alpaca dataset...")
    
    try:
        from datasets import load_dataset
        
        ds = load_dataset('sahil2801/CodeAlpaca-20k')
        
        with open(os.path.join(output_dir, 'code_alpaca_train.txt'), 'w', encoding='utf-8') as f:
            for sample in ds['train']:
                instruction = sample.get('instruction', '')
                input_text = sample.get('input', '')
                output_text = sample.get('output', '')
                
                if input_text:
                    f.write(f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n{output_text}\n\n")
                else:
                    f.write(f"### Instruction:\n{instruction}\n\n### Response:\n{output_text}\n\n")
        
        print("   ✅ Code Alpaca downloaded")
        return True
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False


def download_mbpp(output_dir: str):
    """Download MBPP (Python problems, ~2MB)."""
    print("\n📥 Downloading MBPP dataset...")
    
    try:
        from datasets import load_dataset
        
        ds = load_dataset('mbpp')
        
        for split_name in ['train', 'validation', 'test']:
            if split_name in ds:
                with open(os.path.join(output_dir, f'mbpp_{split_name}.txt'), 'w', encoding='utf-8') as f:
                    for sample in ds[split_name]:
                        prompt = sample.get('prompt', '')
                        code = sample.get('code', '')
                        f.write(f"# {prompt}\n{code}\n\n")
        
        print("   ✅ MBPP downloaded")
        return True
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False


def download_humaneval(output_dir: str):
    """Download HumanEval benchmark (~1MB)."""
    print("\n📥 Downloading HumanEval dataset...")
    
    try:
        from datasets import load_dataset
        
        ds = load_dataset('openai_humaneval')
        
        with open(os.path.join(output_dir, 'humaneval_test.txt'), 'w', encoding='utf-8') as f:
            for sample in ds['test']:
                prompt = sample.get('prompt', '')
                f.write(f"{prompt}\n")
        
        print("   ✅ HumanEval downloaded")
        return True
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False


def download_github_python_samples(output_dir: str, max_samples: int = 50000):
    """Download Python code samples from GitHub Code dataset."""
    print(f"\n📥 Downloading GitHub Python samples (max {max_samples:,})...")
    
    try:
        from datasets import load_dataset
        
        ds = load_dataset(
            'codeparrot/github-code',
            languages=['Python'],
            streaming=True,
            split='train',
            trust_remote_code=True
        )
        
        count = 0
        with open(os.path.join(output_dir, 'github_python.txt'), 'w', encoding='utf-8') as f:
            for sample in ds:
                if count >= max_samples:
                    break
                
                code = sample.get('code', '')
                if code and len(code) > 50:  # Filter very short snippets
                    f.write(code + '\n\n')
                    count += 1
                    
                    if count % 10000 == 0:
                        print(f"   Downloaded {count:,} samples...")
        
        print(f"   ✅ GitHub Python downloaded ({count:,} samples)")
        return True
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False


def download_codesearchnet(output_dir: str, languages: list = None):
    """Download CodeSearchNet dataset."""
    if languages is None:
        languages = ['python', 'javascript']
    
    print(f"\n📥 Downloading CodeSearchNet ({', '.join(languages)})...")
    
    try:
        from datasets import load_dataset
        
        for lang in languages:
            print(f"   Processing {lang}...")
            try:
                ds = load_dataset('code_search_net', lang)
                
                for split_name in ['train', 'validation', 'test']:
                    if split_name in ds:
                        with open(os.path.join(output_dir, f'codesearchnet_{lang}_{split_name}.txt'), 'w', encoding='utf-8') as f:
                            for sample in ds[split_name]:
                                code = sample.get('whole_func_string', '')
                                docstring = sample.get('func_documentation_string', '')
                                
                                if docstring:
                                    f.write(f'"""{docstring}"""\n')
                                if code:
                                    f.write(code + '\n\n')
                
                print(f"   ✅ CodeSearchNet {lang} downloaded")
            except Exception as e:
                print(f"   ⚠️ Could not download {lang}: {e}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False


def download_stack_sample(output_dir: str, max_samples: int = 100000):
    """Download a sample from The Stack dataset."""
    print(f"\n📥 Downloading The Stack sample (max {max_samples:,})...")
    
    try:
        from datasets import load_dataset
        
        languages = ['python', 'javascript', 'java', 'cpp', 'go', 'rust']
        samples_per_lang = max_samples // len(languages)
        
        for lang in languages:
            print(f"   Processing {lang}...")
            try:
                ds = load_dataset(
                    'bigcode/the-stack',
                    lang,
                    streaming=True,
                    split='train',
                    trust_remote_code=True
                )
                
                count = 0
                with open(os.path.join(output_dir, f'stack_{lang}.txt'), 'w', encoding='utf-8') as f:
                    for sample in ds:
                        if count >= samples_per_lang:
                            break
                        
                        code = sample.get('content', '')
                        if code and len(code) > 50:
                            f.write(f"```{lang}\n{code}\n```\n\n")
                            count += 1
                            
                            if count % 10000 == 0:
                                print(f"      {count:,} samples...")
                
                print(f"   ✅ {lang}: {count:,} samples")
            except Exception as e:
                print(f"   ⚠️ Could not download {lang}: {e}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False


def create_combined_dataset(output_dir: str, prefix: str = ""):
    """Combine all dataset files."""
    print("\n📋 Creating combined dataset...")
    
    combined_file = os.path.join(output_dir, f'{prefix}combined_train.txt' if prefix else 'combined_train.txt')
    total_lines = 0
    
    with open(combined_file, 'w', encoding='utf-8') as out_f:
        for filename in sorted(os.listdir(output_dir)):
            if filename.endswith('.txt') and 'combined' not in filename:
                filepath = os.path.join(output_dir, filename)
                print(f"   Adding: {filename}")
                
                with open(filepath, 'r', encoding='utf-8') as in_f:
                    for line in in_f:
                        out_f.write(line)
                        total_lines += 1
    
    # Get file size
    size_mb = os.path.getsize(combined_file) / (1024 * 1024)
    
    print(f"\n   ✅ Combined: {total_lines:,} lines, {size_mb:.1f} MB")
    return combined_file


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Quick start data setup for SIFE-LDM')
    parser.add_argument('--mode', type=str, default='both',
                        choices=['nlp', 'code', 'both', 'minimal'],
                        help='Dataset type to download')
    parser.add_argument('--output', type=str, default='./datasets',
                        help='Output directory')
    parser.add_argument('--max_code_samples', type=int, default=50000,
                        help='Maximum code samples per language')
    
    args = parser.parse_args()
    
    print("\n" + "=" * 60)
    print("SIFE-LDM Quick Start Data Setup")
    print("=" * 60)
    
    # Create output directories
    nlp_dir = os.path.join(args.output, 'nlp')
    code_dir = os.path.join(args.output, 'code')
    os.makedirs(nlp_dir, exist_ok=True)
    os.makedirs(code_dir, exist_ok=True)
    
    results = {'nlp': [], 'code': []}
    
    # Download based on mode
    if args.mode in ['nlp', 'both']:
        print("\n" + "=" * 60)
        print("Downloading NLP Datasets")
        print("=" * 60)
        
        results['nlp'].append(download_wikitext2(nlp_dir))
        results['nlp'].append(download_wikitext103(nlp_dir))
        results['nlp'].append(download_alpaca(nlp_dir))
        
        create_combined_dataset(nlp_dir, 'nlp_')
    
    if args.mode in ['code', 'both']:
        print("\n" + "=" * 60)
        print("Downloading Code Datasets")
        print("=" * 60)
        
        results['code'].append(download_code_alpaca(code_dir))
        results['code'].append(download_mbpp(code_dir))
        results['code'].append(download_humaneval(code_dir))
        results['code'].append(download_codesearchnet(code_dir, ['python', 'javascript']))
        results['code'].append(download_github_python_samples(code_dir, args.max_code_samples))
        
        create_combined_dataset(code_dir, 'code_')
    
    if args.mode == 'minimal':
        print("\n" + "=" * 60)
        print("Downloading Minimal Test Datasets")
        print("=" * 60)
        
        results['nlp'].append(download_wikitext2(nlp_dir))
        results['code'].append(download_code_alpaca(code_dir))
        results['code'].append(download_mbpp(code_dir))
        
        create_combined_dataset(nlp_dir, 'nlp_')
        create_combined_dataset(code_dir, 'code_')
    
    # Print summary
    print("\n" + "=" * 60)
    print("Download Summary")
    print("=" * 60)
    
    print(f"\nNLP datasets: {sum(results['nlp'])}/{len(results['nlp'])} successful")
    print(f"Code datasets: {sum(results['code'])}/{len(results['code'])} successful")
    
    # Create stats
    stats = {
        'nlp_dir': nlp_dir,
        'code_dir': code_dir,
        'nlp_files': [f for f in os.listdir(nlp_dir) if f.endswith('.txt')],
        'code_files': [f for f in os.listdir(code_dir) if f.endswith('.txt')]
    }
    
    stats_file = os.path.join(args.output, 'download_stats.json')
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"\n✅ Data ready for training!")
    print(f"   NLP data: {nlp_dir}")
    print(f"   Code data: {code_dir}")
    print(f"   Stats: {stats_file}")
    
    # Print next steps
    print("\n" + "=" * 60)
    print("Next Steps")
    print("=" * 60)
    print("""
To train SIFE-LDM on these datasets:

# Train on NLP data:
python train.py --data ./datasets/nlp/nlp_combined_train.txt --config configs/base.json

# Train on Code data:
python train.py --data ./datasets/code --config configs/base.json

# Train on TPU:
python train.py --data ./datasets/nlp/nlp_combined_train.txt --tpu --batch_size 64

# For preprocessing vocabulary:
python scripts/prepare_data.py --source ./datasets/nlp --output ./processed/nlp
""")


if __name__ == '__main__':
    main()
