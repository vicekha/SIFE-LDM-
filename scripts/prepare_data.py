#!/usr/bin/env python3
"""
SIFE-LDM Data Preparation Script
=================================

Prepare training data for SIFE-LDM from various sources:
- Text files
- Code repositories
- HuggingFace datasets
- Custom JSON/JSONL files

Usage:
    python prepare_data.py --source /path/to/data --output processed/
    python prepare_data.py --dataset wikitext --output processed/
    python prepare_data.py --code /path/to/repo --output processed/

Author: SIFE-LDM Research Team
License: MIT
"""

import argparse
import json
import os
import sys
import glob
import re
from pathlib import Path
from typing import List, Dict, Any, Optional
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sife.tokenizer import Vocabulary


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Prepare data for SIFE-LDM')
    
    # Input sources
    parser.add_argument('--source', type=str, default=None,
                        help='Path to text file or directory')
    parser.add_argument('--dataset', type=str, default=None,
                        help='HuggingFace dataset name')
    parser.add_argument('--code', type=str, default=None,
                        help='Path to code repository')
    parser.add_argument('--jsonl', type=str, default=None,
                        help='Path to JSONL file')
    parser.add_argument('--manifest', type=str, default=None,
                        help='Path to JSON manifest for multi-domain data')
    
    # HuggingFace Pipeline options
    parser.add_argument('--hf_pipeline', action='store_true',
                        help='Use the unified HuggingFace data pipeline')
    parser.add_argument('--hf_dataset', type=str, default='wikitext',
                        help='HuggingFace dataset path')
    parser.add_argument('--hf_config', type=str, default=None,
                        help='HuggingFace dataset config')
    parser.add_argument('--hf_samples', type=int, default=1000,
                        help='Maximum number of samples from HF pipeline')
    parser.add_argument('--hf_type', type=str, default='nlp',
                        choices=['nlp', 'code'],
                        help='Type of data from HF pipeline')
    
    # Output
    parser.add_argument('--output', type=str, required=True,
                        help='Output directory')
    
    # Processing options
    parser.add_argument('--min_length', type=int, default=10,
                        help='Minimum sequence length')
    parser.add_argument('--max_length', type=int, default=8192,
                        help='Maximum sequence length')
    parser.add_argument('--split_sentences', action='store_true',
                        help='Split into sentences')
    
    # Vocabulary options
    parser.add_argument('--vocab_size', type=int, default=32000,
                        help='Maximum vocabulary size')
    parser.add_argument('--min_freq', type=int, default=2,
                        help='Minimum token frequency')
    parser.add_argument('--vocab_type', type=str, default='word',
                        choices=['word', 'bpe', 'char'],
                        help='Vocabulary type')
    
    # Code-specific options
    parser.add_argument('--extensions', type=str, default='.py,.js,.ts,.java,.cpp,.c,.go,.rs',
                        help='Comma-separated list of file extensions for code')
    
    # Dataset options
    parser.add_argument('--split', type=str, default='train',
                        help='Dataset split to use')
    parser.add_argument('--text_field', type=str, default='text',
                        help='Field name for text in dataset')
    
    return parser.parse_args()


def clean_text(text: str) -> str:
    """Clean and normalize text."""
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    # Remove control characters
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', text)
    return text.strip()


def load_text_file(path: str, min_length: int, max_length: int) -> List[str]:
    """Load text from a file."""
    texts = []
    
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = clean_text(line)
            if min_length <= len(line) <= max_length:
                texts.append(line)
    
    return texts


def load_text_directory(
    directory: str,
    min_length: int,
    max_length: int,
    extensions: Optional[List[str]] = None
) -> List[str]:
    """Load text files from a directory."""
    texts = []
    
    if extensions:
        patterns = [os.path.join(directory, f'**/*{ext}') for ext in extensions]
    else:
        patterns = [os.path.join(directory, '**/*')]
    
    for pattern in patterns:
        for filepath in glob.glob(pattern, recursive=True):
            if os.path.isfile(filepath):
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    # Split into chunks if too long
                    if len(content) > max_length:
                        chunks = [content[i:i+max_length] for i in range(0, len(content), max_length)]
                        for chunk in chunks:
                            chunk = clean_text(chunk)
                            if len(chunk) >= min_length:
                                texts.append(chunk)
                    else:
                        content = clean_text(content)
                        if len(content) >= min_length:
                            texts.append(content)
                            
                except Exception as e:
                    print(f"Warning: Could not read {filepath}: {e}")
    
    return texts


def load_code_repository(
    directory: str,
    extensions: List[str],
    min_length: int,
    max_length: int
) -> List[str]:
    """Load code files from a repository."""
    texts = []
    
    for ext in extensions:
        pattern = os.path.join(directory, f'**/*{ext}')
        
        for filepath in tqdm(glob.glob(pattern, recursive=True), desc=f"Loading {ext} files"):
            if os.path.isfile(filepath):
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        code = f.read()
                    
                    # Skip very short or very long files
                    if min_length <= len(code) <= max_length:
                        texts.append(code)
                        
                except Exception as e:
                    print(f"Warning: Could not read {filepath}: {e}")
    
    return texts


def load_jsonl(path: str, text_field: str, min_length: int, max_length: int) -> List[str]:
    """Load text from a JSONL file."""
    texts = []
    
    with open(path, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="Loading JSONL"):
            try:
                data = json.loads(line)
                text = data.get(text_field, '')
                text = clean_text(text)
                
                if min_length <= len(text) <= max_length:
                    texts.append(text)
                    
            except json.JSONDecodeError:
                continue
    
    return texts


def load_huggingface_dataset(
    dataset_name: str,
    split: str,
    text_field: str,
    min_length: int,
    max_length: int
) -> List[str]:
    """Load data from HuggingFace datasets."""
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("Please install datasets: pip install datasets")
    
    print(f"Loading dataset: {dataset_name}")
    dataset = load_dataset(dataset_name, split=split)
    
    texts = []
    for item in tqdm(dataset, desc="Processing dataset"):
        text = item.get(text_field, '')
        text = clean_text(text)
        
        if min_length <= len(text) <= max_length:
            texts.append(text)
    
    return texts


def split_into_sentences(texts: List[str]) -> List[str]:
    """Split texts into individual sentences."""
    try:
        import nltk
        nltk.download('punkt', quiet=True)
        
        sentences = []
        for text in texts:
            for sent in nltk.sent_tokenize(text):
                sent = sent.strip()
                if len(sent) >= 10:  # Minimum sentence length
                    sentences.append(sent)
        
        return sentences
        
    except ImportError:
        # Simple sentence splitting
        sentences = []
        for text in texts:
            for sent in re.split(r'[.!?]+', text):
                sent = sent.strip()
                if len(sent) >= 10:
                    sentences.append(sent)
        
        return sentences


def build_vocabulary(
    texts: List[str],
    vocab_size: int,
    min_freq: int,
    vocab_type: str
) -> Vocabulary:
    """Build vocabulary from texts."""
    print(f"Building {vocab_type} vocabulary...")
    
    vocab = Vocabulary(min_freq=min_freq, max_size=vocab_size)
    
    if vocab_type == 'bpe':
        # Use tokenizers library for BPE
        try:
            from tokenizers import Tokenizer
            from tokenizers.models import BPE
            from tokenizers.trainers import BpeTrainer
            from tokenizers.pre_tokenizers import Whitespace
            
            tokenizer = Tokenizer(BPE())
            tokenizer.pre_tokenizer = Whitespace()
            trainer = BpeTrainer(
                vocab_size=vocab_size,
                min_frequency=min_freq,
                special_tokens=['<pad>', '<unk>', '<bos>', '<eos>', '<mask>']
            )
            
            tokenizer.train_from_iterator(texts, trainer)
            
            # Convert to our vocabulary format
            vocab_data = tokenizer.get_vocab()
            for token, idx in sorted(vocab_data.items(), key=lambda x: x[1]):
                vocab.token_to_id[token] = idx
                vocab.id_to_token[idx] = token
            
            vocab._frozen = True
            
        except ImportError:
            print("Warning: tokenizers not installed, using word vocabulary")
            vocab.build_from_texts(texts)
    else:
        vocab.build_from_texts(texts)
    
    print(f"Vocabulary size: {len(vocab)}")
    return vocab


def main():
    """Main data preparation function."""
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Load data from source
    texts = []
    
    if args.source:
        print(f"Loading data from: {args.source}")
        if os.path.isfile(args.source):
            texts = load_text_file(args.source, args.min_length, args.max_length)
        elif os.path.isdir(args.source):
            texts = load_text_directory(args.source, args.min_length, args.max_length)
    
    elif args.code:
        print(f"Loading code from: {args.code}")
        extensions = [ext.strip() for ext in args.extensions.split(',')]
        texts = load_code_repository(args.code, extensions, args.min_length, args.max_length)
    
    elif args.dataset:
        print(f"Loading HuggingFace dataset: {args.dataset}")
        texts = load_huggingface_dataset(
            args.dataset, args.split, args.text_field,
            args.min_length, args.max_length
        )
    
    elif args.jsonl:
        print(f"Loading JSONL from: {args.jsonl}")
        texts = load_jsonl(args.jsonl, args.text_field, args.min_length, args.max_length)
    
    elif args.hf_pipeline:
        print(f"Using HuggingFace pipeline for: {args.hf_dataset}")
        import subprocess
        
        # Define temporary path for pipeline output
        temp_output = os.path.join(args.output, "temp_hf_output.txt")
        os.makedirs(args.output, exist_ok=True)
        
        cmd = [
            sys.executable,
            os.path.join(os.path.dirname(__file__), "hf_data_pipeline.py"),
            "--type", args.hf_type,
            "--dataset", args.hf_dataset,
            "--max_samples", str(args.hf_samples),
            "--output", temp_output
        ]
        if args.hf_config:
            cmd.extend(["--config", args.hf_config])
            
        try:
            subprocess.run(cmd, check=True)
            if os.path.exists(temp_output):
                with open(temp_output, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if args.min_length <= len(line) <= args.max_length:
                            texts.append(line)
                # os.remove(temp_output) # Keep for debugging or let user clean up
        except subprocess.CalledProcessError as e:
            print(f"Error running HF pipeline: {e}")
            sys.exit(1)
    
    elif args.manifest:
        print(f"Loading multi-domain manifest from: {args.manifest}")
        with open(args.manifest, 'r') as f:
            manifest = json.load(f)
        
        # We will collect samples from each domain in the manifest
        # and interleave them to ensure a balanced mix.
        all_domain_texts = []
        for entry in manifest:
            domain_name = entry.get('dataset', 'unknown')
            print(f"  -> Loading domain: {domain_name}")
            
            # Use hf_data_pipeline for each entry
            temp_output = os.path.join(args.output, f"temp_{domain_name}.txt")
            cmd = [
                sys.executable,
                os.path.join(os.path.dirname(__file__), "hf_data_pipeline.py"),
                "--type", entry.get('type', 'nlp'),
                "--dataset", entry.get('dataset'),
                "--max_samples", str(entry.get('samples', 1000)),
                "--output", temp_output
            ]
            if entry.get('config'):
                cmd.extend(["--config", entry.get('config')])
                
            try:
                import subprocess
                subprocess.run(cmd, check=True)
                if os.path.exists(temp_output):
                    domain_texts = []
                    with open(temp_output, 'r', encoding='utf-8') as f:
                        for line in f:
                            line = line.strip()
                            if args.min_length <= len(line) <= args.max_length:
                                domain_texts.append(line)
                    all_domain_texts.append(domain_texts)
                    print(f"     Loaded {len(domain_texts)} samples from {domain_name}")
                    # os.remove(temp_output)
            except Exception as e:
                print(f"Error loading domain {domain_name}: {e}")
        
        # Interleave samples
        max_len = max(len(d) for d in all_domain_texts) if all_domain_texts else 0
        for i in range(max_len):
            for domain in all_domain_texts:
                if i < len(domain):
                    texts.append(domain[i])
    
    else:
        raise ValueError("Please specify one of: --source, --dataset, --code, or --jsonl")
    
    print(f"Loaded {len(texts)} samples")
    
    # Split into sentences if requested
    if args.split_sentences:
        texts = split_into_sentences(texts)
        print(f"After sentence splitting: {len(texts)} samples")
    
    # Split into train/val/test
    print("Splitting into train/val/test...")
    
    import random
    random.seed(42)
    random.shuffle(texts)
    
    n = len(texts)
    train_texts = texts[:int(0.95 * n)]
    val_texts = texts[int(0.95 * n):int(0.975 * n)]
    test_texts = texts[int(0.975 * n):]
    
    print(f"Train: {len(train_texts)}, Val: {len(val_texts)}, Test: {len(test_texts)}")
    
    # Build vocabulary
    vocab = build_vocabulary(
        train_texts,
        args.vocab_size,
        args.min_freq,
        args.vocab_type
    )
    
    # Save everything
    print("Saving processed data...")
    
    # Save texts
    def save_texts(texts, filename):
        path = os.path.join(args.output, filename)
        with open(path, 'w', encoding='utf-8') as f:
            for text in texts:
                f.write(text + '\n')
        print(f"Saved {len(texts)} samples to {path}")
    
    save_texts(train_texts, 'train.txt')
    save_texts(val_texts, 'val.txt')
    save_texts(test_texts, 'test.txt')
    
    # Save vocabulary
    vocab_path = os.path.join(args.output, 'vocab.json')
    vocab.save(vocab_path)
    print(f"Saved vocabulary to {vocab_path}")
    
    # Save statistics
    stats = {
        'total_samples': n,
        'train_samples': len(train_texts),
        'val_samples': len(val_texts),
        'test_samples': len(test_texts),
        'vocab_size': len(vocab),
        'avg_length': sum(len(t) for t in texts) / len(texts) if texts else 0,
        'min_length': min(len(t) for t in texts) if texts else 0,
        'max_length': max(len(t) for t in texts) if texts else 0
    }
    
    stats_path = os.path.join(args.output, 'stats.json')
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"Saved statistics to {stats_path}")
    
    print("\nData preparation complete!")
    print(f"Total samples: {n}")
    print(f"Vocabulary size: {len(vocab)}")
    print(f"Average length: {stats['avg_length']:.1f} characters")


if __name__ == '__main__':
    main()
