#!/usr/bin/env python3
"""
Download Additional Code Datasets
=================================

Downloads code datasets that work with the latest HuggingFace datasets API.

Author: SIFE-LDM Research Team
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def download_thestack_smol(output_dir: str, max_samples: int = 50000):
    """Download from The Stack Smol (smaller version of The Stack)."""
    print(f"\n📥 Downloading The Stack Smol (max {max_samples:,} per language)...")
    
    try:
        from datasets import load_dataset
        
        languages = ['python', 'javascript', 'java', 'cpp', 'go', 'rust', 'typescript', 'c']
        samples_per_lang = max_samples // len(languages)
        total_count = 0
        
        for lang in languages:
            print(f"   Processing {lang}...")
            try:
                ds = load_dataset(
                    'bigcode/the-stack-smol',
                    lang,
                    split='train',
                    trust_remote_code=True
                )
                
                count = 0
                with open(os.path.join(output_dir, f'stack_smol_{lang}.txt'), 'w', encoding='utf-8') as f:
                    for sample in ds:
                        if count >= samples_per_lang:
                            break
                        
                        code = sample.get('content', '')
                        if code and len(code) > 50:
                            f.write(f"```{lang}\n{code}\n```\n\n")
                            count += 1
                            
                            if count % 5000 == 0:
                                print(f"      {count:,} samples...")
                
                total_count += count
                print(f"   ✅ {lang}: {count:,} samples")
                
            except Exception as e:
                print(f"   ⚠️ Could not download {lang}: {e}")
        
        print(f"   ✅ Total: {total_count:,} code samples")
        return total_count
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return 0


def download_code_search_net_new(output_dir: str):
    """Download CodeSearchNet using the new API."""
    print("\n📥 Downloading CodeSearchNet...")
    
    try:
        from datasets import load_dataset
        
        languages = ['python', 'javascript', 'go', 'ruby', 'java', 'php']
        total_count = 0
        
        for lang in languages:
            print(f"   Processing {lang}...")
            try:
                ds = load_dataset(
                    'code-search-net/code_search_net',
                    lang,
                    trust_remote_code=True
                )
                
                for split_name in ['train', 'validation', 'test']:
                    if split_name in ds:
                        count = 0
                        with open(os.path.join(output_dir, f'codesearchnet_{lang}_{split_name}.txt'), 'w', encoding='utf-8') as f:
                            for sample in ds[split_name]:
                                code = sample.get('whole_func_string', '')
                                docstring = sample.get('func_documentation_string', '')
                                
                                if docstring:
                                    f.write(f'"""{docstring}"""\n')
                                if code:
                                    f.write(code + '\n\n')
                                    count += 1
                        
                        print(f"      {split_name}: {count:,} samples")
                        total_count += count
                
                print(f"   ✅ {lang} downloaded")
                
            except Exception as e:
                print(f"   ⚠️ Could not download {lang}: {e}")
        
        print(f"   ✅ Total: {total_count:,} code samples")
        return total_count
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return 0


def download_python_code_instances(output_dir: str, max_samples: int = 30000):
    """Download Python code from CodeParrot."""
    print(f"\n📥 Downloading Python code instances (max {max_samples:,})...")
    
    try:
        from datasets import load_dataset
        
        ds = load_dataset(
            'codeparrot/python-code-instructions',
            split='train',
            streaming=True
        )
        
        count = 0
        with open(os.path.join(output_dir, 'python_instructions.txt'), 'w', encoding='utf-8') as f:
            for sample in ds:
                if count >= max_samples:
                    break
                
                prompt = sample.get('prompt', '')
                code = sample.get('code', '')
                
                if prompt and code:
                    f.write(f"# {prompt}\n{code}\n\n")
                    count += 1
                    
                    if count % 5000 == 0:
                        print(f"   Downloaded {count:,} samples...")
        
        print(f"   ✅ Downloaded {count:,} Python samples")
        return count
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return 0


def download_instruct_code(output_dir: str, max_samples: int = 50000):
    """Download instruct-code dataset."""
    print(f"\n📥 Downloading Instruct Code dataset (max {max_samples:,})...")
    
    try:
        from datasets import load_dataset
        
        ds = load_dataset(
            'nampdn-ai/tiny-codes',
            split='train',
            streaming=True
        )
        
        count = 0
        with open(os.path.join(output_dir, 'tiny_codes.txt'), 'w', encoding='utf-8') as f:
            for sample in ds:
                if count >= max_samples:
                    break
                
                instruction = sample.get('instruction', '')
                code = sample.get('output', '')
                
                if instruction and code:
                    f.write(f"### Instruction:\n{instruction}\n\n### Code:\n{code}\n\n")
                    count += 1
                    
                    if count % 5000 == 0:
                        print(f"   Downloaded {count:,} samples...")
        
        print(f"   ✅ Downloaded {count:,} code samples")
        return count
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return 0


def download_evol_code(output_dir: str, max_samples: int = 30000):
    """Download Evol-Instruct-Code dataset."""
    print(f"\n📥 Downloading Evol-Instruct-Code dataset (max {max_samples:,})...")
    
    try:
        from datasets import load_dataset
        
        ds = load_dataset(
            'nickrosh/Evol-Instruct-Code-80k-v1',
            split='train',
            streaming=True
        )
        
        count = 0
        with open(os.path.join(output_dir, 'evol_code.txt'), 'w', encoding='utf-8') as f:
            for sample in ds:
                if count >= max_samples:
                    break
                
                instruction = sample.get('instruction', '')
                output = sample.get('output', '')
                
                if instruction and output:
                    f.write(f"### Instruction:\n{instruction}\n\n### Response:\n{output}\n\n")
                    count += 1
                    
                    if count % 5000 == 0:
                        print(f"   Downloaded {count:,} samples...")
        
        print(f"   ✅ Downloaded {count:,} code samples")
        return count
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return 0


def create_combined_code_dataset(output_dir: str):
    """Combine all code datasets."""
    print("\n📋 Creating combined code dataset...")
    
    combined_file = os.path.join(output_dir, 'all_code_combined.txt')
    total_lines = 0
    
    with open(combined_file, 'w', encoding='utf-8') as out_f:
        for filename in sorted(os.listdir(output_dir)):
            if filename.endswith('.txt') and 'combined' not in filename and 'all' not in filename:
                filepath = os.path.join(output_dir, filename)
                print(f"   Adding: {filename}")
                
                with open(filepath, 'r', encoding='utf-8') as in_f:
                    for line in in_f:
                        out_f.write(line)
                        total_lines += 1
    
    size_mb = os.path.getsize(combined_file) / (1024 * 1024)
    print(f"\n   ✅ Combined: {total_lines:,} lines, {size_mb:.1f} MB")
    return combined_file


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Download additional code datasets')
    parser.add_argument('--output', type=str, default='./datasets/code',
                        help='Output directory')
    parser.add_argument('--max_samples', type=int, default=30000,
                        help='Maximum samples per dataset')
    
    args = parser.parse_args()
    
    os.makedirs(args.output, exist_ok=True)
    
    print("\n" + "=" * 60)
    print("Downloading Additional Code Datasets")
    print("=" * 60)
    
    # Download various code datasets
    download_thestack_smol(args.output, args.max_samples * 2)
    download_python_code_instances(args.output, args.max_samples)
    download_instruct_code(args.output, args.max_samples)
    download_evol_code(args.output, args.max_samples)
    
    # Create combined dataset
    create_combined_code_dataset(args.output)
    
    print("\n" + "=" * 60)
    print("✅ Additional Code Datasets Downloaded!")
    print("=" * 60)


if __name__ == '__main__':
    main()
