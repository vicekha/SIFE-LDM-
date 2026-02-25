#!/usr/bin/env python3
"""
SIFE-LDM Dataset Downloader
============================

Downloads and prepares NLP and coding datasets for SIFE-LDM training.

Datasets included:
NLP:
- WikiText-103 (high-quality Wikipedia text)
- OpenWebText (Reddit-linked web pages)
- BookCorpus (books)
- C4 (Colossal Clean Crawled Corpus) - subset

Code:
- The Stack (multi-language code)
- CodeParrot (Python code)
- CodeSearchNet (code with documentation)

Usage:
    python download_datasets.py --output /path/to/data --nlp --code
    python download_datasets.py --output /path/to/data --nlp_only
    python download_datasets.py --output /path/to/data --code_only --languages python,javascript

Author: SIFE-LDM Research Team
License: MIT
"""

import argparse
import os
import sys
import json
import gzip
import shutil
import tarfile
import zipfile
from pathlib import Path
from typing import List, Dict, Any, Optional, Iterator
from tqdm import tqdm
import time
import requests
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Download datasets for SIFE-LDM')
    
    # Output
    parser.add_argument('--output', type=str, default='./data',
                        help='Output directory for datasets')
    
    # Dataset selection
    parser.add_argument('--nlp', action='store_true',
                        help='Download NLP datasets')
    parser.add_argument('--code', action='store_true',
                        help='Download coding datasets')
    parser.add_argument('--nlp_only', action='store_true',
                        help='Download only NLP datasets')
    parser.add_argument('--code_only', action='store_true',
                        help='Download only coding datasets')
    
    # NLP dataset options
    parser.add_argument('--nlp_datasets', type=str, 
                        default='wikitext,openwebtext',
                        help='Comma-separated NLP datasets to download')
    parser.add_argument('--max_nlp_samples', type=int, default=1000000,
                        help='Maximum number of NLP samples')
    
    # Code dataset options
    parser.add_argument('--code_datasets', type=str,
                        default='thestack,codeparrot',
                        help='Comma-separated code datasets to download')
    parser.add_argument('--languages', type=str, 
                        default='python,javascript,java,cpp,typescript,go,rust',
                        help='Comma-separated programming languages')
    parser.add_argument('--max_code_samples', type=int, default=500000,
                        help='Maximum number of code samples per language')
    
    # Processing options
    parser.add_argument('--min_length', type=int, default=100,
                        help='Minimum text/code length')
    parser.add_argument('--max_length', type=int, default=8192,
                        help='Maximum text/code length')
    parser.add_argument('--chunk_size', type=int, default=2000,
                        help='Chunk size for long documents')
    
    # Streaming options (for large datasets)
    parser.add_argument('--stream', action='store_true',
                        help='Use streaming mode for large datasets')
    parser.add_argument('--cache_dir', type=str, default=None,
                        help='HuggingFace cache directory')
    
    return parser.parse_args()


def download_file(url: str, output_path: str, desc: str = None):
    """Download a file with progress bar."""
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(output_path, 'wb') as f:
        with tqdm(total=total_size, unit='B', unit_scale=True, desc=desc) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))


def download_with_huggingface(
    dataset_name: str,
    output_dir: str,
    split: str = 'train',
    text_field: str = 'text',
    max_samples: int = None,
    cache_dir: str = None,
    streaming: bool = False
) -> List[str]:
    """Download dataset using HuggingFace datasets library."""
    try:
        from datasets import load_dataset
    except ImportError:
        print("Installing datasets library...")
        os.system('pip install datasets -q')
        from datasets import load_dataset
    
    print(f"\nDownloading {dataset_name}...")
    
    texts = []
    
    try:
        if streaming:
            dataset = load_dataset(dataset_name, split=split, streaming=True, cache_dir=cache_dir)
        else:
            dataset = load_dataset(dataset_name, split=split, cache_dir=cache_dir)
    except Exception as e:
        print(f"Error loading {dataset_name}: {e}")
        return texts
    
    count = 0
    for item in tqdm(dataset, desc=f"Processing {dataset_name}"):
        text = item.get(text_field, '')
        
        if text and len(text) >= 50:
            texts.append(text)
            count += 1
            
            if max_samples and count >= max_samples:
                break
    
    print(f"Downloaded {len(texts)} samples from {dataset_name}")
    return texts


def download_wikitext(output_dir: str, max_samples: int = None) -> List[str]:
    """Download WikiText-103 dataset."""
    print("\n" + "="*60)
    print("Downloading WikiText-103")
    print("="*60)
    
    texts = download_with_huggingface(
        'wikitext',
        output_dir,
        split='train',
        text_field='text',
        max_samples=max_samples,
        name='wikitext-103-raw-v1'
    )
    
    # Filter empty lines
    texts = [t for t in texts if t.strip()]
    return texts


def download_openwebtext(output_dir: str, max_samples: int = 500000) -> List[str]:
    """Download OpenWebText dataset."""
    print("\n" + "="*60)
    print("Downloading OpenWebText")
    print("="*60)
    
    texts = download_with_huggingface(
        'openwebtext',
        output_dir,
        split='train',
        text_field='text',
        max_samples=max_samples,
        streaming=True
    )
    
    return texts


def download_bookcorpus(output_dir: str, max_samples: int = 100000) -> List[str]:
    """Download BookCorpus dataset."""
    print("\n" + "="*60)
    print("Downloading BookCorpus")
    print("="*60)
    
    try:
        texts = download_with_huggingface(
            'bookcorpus',
            output_dir,
            split='train',
            text_field='text',
            max_samples=max_samples,
            streaming=True
        )
    except Exception as e:
        print(f"BookCorpus not available: {e}")
        print("Trying alternative: books3...")
        try:
            texts = download_with_huggingface(
                'the_pile',
                output_dir,
                split='train',
                text_field='text',
                max_samples=max_samples,
                streaming=True
            )
        except:
            texts = []
    
    return texts


def download_c4_subset(output_dir: str, max_samples: int = 200000) -> List[str]:
    """Download C4 dataset subset."""
    print("\n" + "="*60)
    print("Downloading C4 (en) subset")
    print("="*60)
    
    texts = download_with_huggingface(
        'c4',
        output_dir,
        split='train',
        text_field='text',
        max_samples=max_samples,
        streaming=True,
        name='en'
    )
    
    return texts


def download_the_stack(
    output_dir: str, 
    languages: List[str],
    max_samples_per_lang: int = 100000
) -> Dict[str, List[str]]:
    """Download The Stack dataset (multi-language code)."""
    print("\n" + "="*60)
    print("Downloading The Stack")
    print("="*60)
    
    code_data = {}
    
    for lang in languages:
        print(f"\nDownloading {lang} code...")
        
        try:
            from datasets import load_dataset
            
            lang_code = {
                'python': 'python',
                'javascript': 'javascript',
                'java': 'java',
                'cpp': 'cpp',
                'c': 'c',
                'typescript': 'typescript',
                'go': 'go',
                'rust': 'rust',
                'php': 'php',
                'ruby': 'ruby',
                'csharp': 'csharp',
                'swift': 'swift',
                'kotlin': 'kotlin'
            }.get(lang.lower(), lang.lower())
            
            dataset = load_dataset(
                'bigcode/the-stack',
                split='train',
                streaming=True,
                data_dir=f'data/{lang_code}'
            )
            
            codes = []
            count = 0
            
            for item in tqdm(dataset, desc=f"Processing {lang}"):
                code = item.get('content', '')
                
                if code and len(code) >= 50:
                    codes.append(code)
                    count += 1
                    
                    if count >= max_samples_per_lang:
                        break
            
            if codes:
                code_data[lang] = codes
                print(f"Downloaded {len(codes)} {lang} samples")
            
        except Exception as e:
            print(f"Error downloading {lang}: {e}")
            continue
    
    return code_data


def download_codeparrot(
    output_dir: str,
    max_samples: int = 200000
) -> List[str]:
    """Download CodeParrot (Python) dataset."""
    print("\n" + "="*60)
    print("Downloading CodeParrot (Python)")
    print("="*60)
    
    codes = download_with_huggingface(
        'codeparrot/codeparrot-clean-train',
        output_dir,
        split='train',
        text_field='content',
        max_samples=max_samples,
        streaming=True
    )
    
    return codes


def download_codesearchnet(
    output_dir: str,
    languages: List[str],
    max_samples_per_lang: int = 50000
) -> Dict[str, List[str]]:
    """Download CodeSearchNet dataset."""
    print("\n" + "="*60)
    print("Downloading CodeSearchNet")
    print("="*60)
    
    code_data = {}
    
    for lang in languages:
        if lang not in ['python', 'javascript', 'java', 'go', 'ruby', 'php']:
            print(f"Skipping {lang} (not in CodeSearchNet)")
            continue
        
        print(f"\nDownloading {lang} from CodeSearchNet...")
        
        try:
            from datasets import load_dataset
            
            dataset = load_dataset(
                'code_search_net',
                split='train',
                languages=[lang]
            )
            
            codes = []
            for item in tqdm(dataset, desc=f"Processing {lang}"):
                code = item.get('func_code_string', '')
                
                if code and len(code) >= 50:
                    codes.append(code)
                    
                    if len(codes) >= max_samples_per_lang:
                        break
            
            if codes:
                code_data[lang] = codes
                print(f"Downloaded {len(codes)} {lang} samples")
            
        except Exception as e:
            print(f"Error downloading {lang} from CodeSearchNet: {e}")
            continue
    
    return code_data


def download_github_code(
    output_dir: str,
    languages: List[str],
    max_samples: int = 50000
) -> Dict[str, List[str]]:
    """Download GitHub code dataset."""
    print("\n" + "="*60)
    print("Downloading GitHub Code Dataset")
    print("="*60)
    
    code_data = {}
    
    # Try code-alpaca dataset as alternative
    try:
        codes = download_with_huggingface(
            'sahil2801/CodeAlpaca-20k',
            output_dir,
            split='train',
            text_field='output',
            max_samples=max_samples
        )
        
        if codes:
            code_data['python'] = codes
            
    except Exception as e:
        print(f"Error: {e}")
    
    return code_data


def process_texts(
    texts: List[str],
    min_length: int,
    max_length: int,
    chunk_size: int
) -> List[str]:
    """Process and clean text data."""
    import re
    
    processed = []
    
    for text in tqdm(texts, desc="Processing texts"):
        # Clean text
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', text)
        text = text.strip()
        
        if len(text) < min_length:
            continue
        
        # Chunk long texts
        if len(text) > max_length:
            chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
            processed.extend([c.strip() for c in chunks if len(c.strip()) >= min_length])
        else:
            processed.append(text)
    
    return processed


def process_code(
    codes: List[str],
    min_length: int,
    max_length: int
) -> List[str]:
    """Process and clean code data."""
    processed = []
    
    for code in tqdm(codes, desc="Processing code"):
        # Basic cleaning
        code = code.strip()
        
        if len(code) < min_length:
            continue
        
        if len(code) > max_length:
            code = code[:max_length]
        
        processed.append(code)
    
    return processed


def save_dataset(
    texts: List[str],
    output_path: str,
    split_ratio: tuple = (0.95, 0.025, 0.025)
):
    """Save dataset with train/val/test splits."""
    import random
    random.seed(42)
    random.shuffle(texts)
    
    n = len(texts)
    train_end = int(n * split_ratio[0])
    val_end = train_end + int(n * split_ratio[1])
    
    train_texts = texts[:train_end]
    val_texts = texts[train_end:val_end]
    test_texts = texts[val_end:]
    
    # Save
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)
    
    with open(f"{output_path}_train.txt", 'w', encoding='utf-8') as f:
        for text in train_texts:
            f.write(text.replace('\n', ' ') + '\n')
    
    with open(f"{output_path}_val.txt", 'w', encoding='utf-8') as f:
        for text in val_texts:
            f.write(text.replace('\n', ' ') + '\n')
    
    with open(f"{output_path}_test.txt", 'w', encoding='utf-8') as f:
        for text in test_texts:
            f.write(text.replace('\n', ' ') + '\n')
    
    print(f"Saved {len(train_texts)} train, {len(val_texts)} val, {len(test_texts)} test samples")
    
    return len(train_texts), len(val_texts), len(test_texts)


def save_code_dataset(
    code_data: Dict[str, List[str]],
    output_dir: str,
    split_ratio: tuple = (0.95, 0.025, 0.025)
):
    """Save code dataset with language-specific files."""
    os.makedirs(output_dir, exist_ok=True)
    
    all_codes = []
    stats = {}
    
    for lang, codes in code_data.items():
        print(f"\nSaving {lang} code...")
        
        # Save language-specific file
        lang_dir = os.path.join(output_dir, 'code', lang)
        os.makedirs(lang_dir, exist_ok=True)
        
        train_n, val_n, test_n = save_dataset(codes, os.path.join(lang_dir, lang))
        stats[lang] = {'train': train_n, 'val': val_n, 'test': test_n, 'total': len(codes)}
        
        all_codes.extend(codes)
    
    # Save combined code file
    combined_dir = os.path.join(output_dir, 'code', 'combined')
    os.makedirs(combined_dir, exist_ok=True)
    
    print("\nSaving combined code dataset...")
    train_n, val_n, test_n = save_dataset(all_codes, os.path.join(combined_dir, 'all_code'))
    stats['combined'] = {'train': train_n, 'val': val_n, 'test': test_n, 'total': len(all_codes)}
    
    return stats


def main():
    """Main download function."""
    args = parse_args()
    
    # Determine what to download
    download_nlp = args.nlp or args.nlp_only or (not args.nlp and not args.code and not args.nlp_only and not args.code_only)
    download_code = args.code or args.code_only or (not args.nlp and not args.code and not args.nlp_only and not args.code_only)
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    all_stats = {}
    
    # Download NLP datasets
    if download_nlp:
        print("\n" + "="*60)
        print("DOWNLOADING NLP DATASETS")
        print("="*60)
        
        nlp_texts = []
        nlp_datasets = args.nlp_datasets.split(',')
        
        for dataset in nlp_datasets:
            dataset = dataset.strip().lower()
            
            if dataset == 'wikitext':
                texts = download_wikitext(args.output, args.max_nlp_samples // len(nlp_datasets))
            elif dataset == 'openwebtext':
                texts = download_openwebtext(args.output, args.max_nlp_samples // len(nlp_datasets))
            elif dataset == 'bookcorpus':
                texts = download_bookcorpus(args.output, args.max_nlp_samples // len(nlp_datasets))
            elif dataset == 'c4':
                texts = download_c4_subset(args.output, args.max_nlp_samples // len(nlp_datasets))
            else:
                print(f"Unknown NLP dataset: {dataset}")
                continue
            
            nlp_texts.extend(texts)
        
        # Process NLP data
        print("\nProcessing NLP texts...")
        nlp_texts = process_texts(nlp_texts, args.min_length, args.max_length, args.chunk_size)
        
        # Remove duplicates
        nlp_texts = list(set(nlp_texts))
        
        # Save NLP data
        nlp_dir = os.path.join(args.output, 'nlp')
        os.makedirs(nlp_dir, exist_ok=True)
        
        print("\nSaving NLP dataset...")
        train_n, val_n, test_n = save_dataset(nlp_texts, os.path.join(nlp_dir, 'nlp'))
        all_stats['nlp'] = {'train': train_n, 'val': val_n, 'test': test_n, 'total': len(nlp_texts)}
    
    # Download code datasets
    if download_code:
        print("\n" + "="*60)
        print("DOWNLOADING CODE DATASETS")
        print("="*60)
        
        code_data = {}
        languages = [l.strip() for l in args.languages.split(',')]
        code_datasets = args.code_datasets.split(',')
        
        for dataset in code_datasets:
            dataset = dataset.strip().lower()
            
            if dataset == 'thestack':
                stack_data = download_the_stack(
                    args.output, 
                    languages, 
                    args.max_code_samples // len(languages)
                )
                for lang, codes in stack_data.items():
                    if lang not in code_data:
                        code_data[lang] = []
                    code_data[lang].extend(codes)
                    
            elif dataset == 'codeparrot':
                codes = download_codeparrot(args.output, args.max_code_samples)
                if 'python' not in code_data:
                    code_data['python'] = []
                code_data['python'].extend(codes)
                
            elif dataset == 'codesearchnet':
                csn_data = download_codesearchnet(
                    args.output,
                    languages,
                    args.max_code_samples // len(languages)
                )
                for lang, codes in csn_data.items():
                    if lang not in code_data:
                        code_data[lang] = []
                    code_data[lang].extend(codes)
                    
            elif dataset == 'github':
                gh_data = download_github_code(args.output, languages, args.max_code_samples)
                for lang, codes in gh_data.items():
                    if lang not in code_data:
                        code_data[lang] = []
                    code_data[lang].extend(codes)
            else:
                print(f"Unknown code dataset: {dataset}")
                continue
        
        # Process code data
        print("\nProcessing code...")
        for lang in code_data:
            code_data[lang] = process_code(
                code_data[lang], 
                args.min_length, 
                args.max_length
            )
            # Remove duplicates
            code_data[lang] = list(set(code_data[lang]))
        
        # Save code data
        code_stats = save_code_dataset(code_data, args.output)
        all_stats['code'] = code_stats
    
    # Save overall statistics
    stats_path = os.path.join(args.output, 'dataset_stats.json')
    with open(stats_path, 'w') as f:
        json.dump(all_stats, f, indent=2)
    
    print("\n" + "="*60)
    print("DOWNLOAD COMPLETE")
    print("="*60)
    print(f"\nDatasets saved to: {args.output}")
    print(f"Statistics saved to: {stats_path}")
    
    # Print summary
    print("\nDataset Summary:")
    print("-" * 40)
    if 'nlp' in all_stats:
        print(f"NLP Dataset:")
        print(f"  Total: {all_stats['nlp']['total']:,} samples")
        print(f"  Train: {all_stats['nlp']['train']:,}")
        print(f"  Val: {all_stats['nlp']['val']:,}")
        print(f"  Test: {all_stats['nlp']['test']:,}")
    
    if 'code' in all_stats:
        print(f"\nCode Dataset:")
        for lang, stats in all_stats['code'].items():
            if isinstance(stats, dict) and 'total' in stats:
                print(f"  {lang}: {stats['total']:,} samples")


if __name__ == '__main__':
    main()
