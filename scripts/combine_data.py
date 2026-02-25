#!/usr/bin/env python3
"""
Combine all downloaded data into training-ready format.
Builds vocabulary and creates final datasets for SIFE-LDM training.
"""

import os
import sys
import json
import random
from pathlib import Path
from tqdm import tqdm
from collections import Counter
import re

def clean_text(text):
    """Clean and normalize text."""
    # Unescape newlines
    text = text.replace('\\n', '\n')
    # Remove excessive whitespace
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()

def load_data_file(filepath):
    """Load data from a file."""
    texts = []
    if os.path.exists(filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                text = line.strip()
                if text:
                    texts.append(clean_text(text))
    return texts

def tokenize_text(text):
    """Simple word tokenization."""
    # Split on whitespace and punctuation
    tokens = re.findall(r'\b\w+\b|[^\w\s]', text.lower())
    return tokens

def build_vocabulary(texts, min_freq=2, max_size=32000, special_tokens=None):
    """Build vocabulary from texts."""
    if special_tokens is None:
        special_tokens = ['<pad>', '<unk>', '<bos>', '<eos>', '<mask>']
    
    print("Building vocabulary...")
    
    # Count tokens
    token_counts = Counter()
    for text in tqdm(texts, desc="Counting tokens"):
        tokens = tokenize_text(text)
        token_counts.update(tokens)
    
    # Sort by frequency
    sorted_tokens = sorted(token_counts.items(), key=lambda x: (-x[1], x[0]))
    
    # Build vocabulary
    vocab = {token: i for i, token in enumerate(special_tokens)}
    
    for token, count in sorted_tokens:
        if count >= min_freq and len(vocab) < max_size:
            if token not in vocab:
                vocab[token] = len(vocab)
    
    print(f"Vocabulary size: {len(vocab)}")
    return vocab

def main():
    base_dir = Path('/home/z/my-project/download/sife-ldm')
    data_dir = base_dir / 'data'
    output_dir = data_dir / 'combined'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("COMBINING DATASETS FOR TRAINING")
    print("="*60)
    
    # Load all NLP data
    print("\n📚 Loading NLP data...")
    nlp_texts = []
    
    nlp_dir = data_dir / 'nlp'
    for split in ['train', 'val', 'test']:
        filepath = nlp_dir / f'nlp_{split}.txt'
        texts = load_data_file(filepath)
        nlp_texts.extend(texts)
        print(f"  Loaded {len(texts)} NLP samples from {split}")
    
    print(f"  Total NLP: {len(nlp_texts):,} samples")
    
    # Load all Code data
    print("\n💻 Loading Code data...")
    code_texts = []
    
    code_dir = data_dir / 'code' / 'combined'
    for filepath in code_dir.glob('*.txt'):
        texts = load_data_file(filepath)
        code_texts.extend(texts)
        print(f"  Loaded {len(texts)} code samples from {filepath.name}")
    
    print(f"  Total Code: {len(code_texts):,} samples")
    
    # Combine and shuffle
    print("\n🔀 Combining and shuffling...")
    
    # Tag samples with source type
    nlp_tagged = [(text, 'nlp') for text in nlp_texts]
    code_tagged = [(text, 'code') for text in code_texts]
    
    combined = nlp_tagged + code_tagged
    random.seed(42)
    random.shuffle(combined)
    
    print(f"  Combined total: {len(combined):,} samples")
    
    # Split into train/val/test
    print("\n✂️ Splitting into train/val/test...")
    
    n = len(combined)
    train_end = int(n * 0.95)
    val_end = int(n * 0.975)
    
    train_data = combined[:train_end]
    val_data = combined[train_end:val_end]
    test_data = combined[val_end:]
    
    print(f"  Train: {len(train_data):,}")
    print(f"  Val: {len(val_data):,}")
    print(f"  Test: {len(test_data):,}")
    
    # Save combined datasets
    print("\n💾 Saving combined datasets...")
    
    def save_split(data, split_name):
        nlp_count = 0
        code_count = 0
        
        # Main text file
        text_path = output_dir / f'{split_name}.txt'
        with open(text_path, 'w', encoding='utf-8') as f:
            for text, source in data:
                # Single line per sample
                f.write(text.replace('\n', ' ') + '\n')
                if source == 'nlp':
                    nlp_count += 1
                else:
                    code_count += 1
        
        # JSONL with metadata
        jsonl_path = output_dir / f'{split_name}.jsonl'
        with open(jsonl_path, 'w', encoding='utf-8') as f:
            for text, source in data:
                record = {
                    'text': text,
                    'source': source,
                    'length': len(text)
                }
                f.write(json.dumps(record) + '\n')
        
        return nlp_count, code_count
    
    train_nlp, train_code = save_split(train_data, 'train')
    val_nlp, val_code = save_split(val_data, 'val')
    test_nlp, test_code = save_split(test_data, 'test')
    
    # Build vocabulary
    print("\n📖 Building vocabulary...")
    all_texts = [text for text, _ in combined]
    vocab = build_vocabulary(all_texts, min_freq=2, max_size=32000)
    
    # Save vocabulary
    vocab_path = output_dir / 'vocab.json'
    with open(vocab_path, 'w', encoding='utf-8') as f:
        json.dump(vocab, f, indent=2)
    print(f"  Saved vocabulary to {vocab_path}")
    
    # Create statistics
    stats = {
        'total_samples': len(combined),
        'nlp_samples': len(nlp_texts),
        'code_samples': len(code_texts),
        'splits': {
            'train': {
                'total': len(train_data),
                'nlp': train_nlp,
                'code': train_code
            },
            'val': {
                'total': len(val_data),
                'nlp': val_nlp,
                'code': val_code
            },
            'test': {
                'total': len(test_data),
                'nlp': test_nlp,
                'code': test_code
            }
        },
        'vocabulary_size': len(vocab),
        'avg_length': sum(len(t) for t, _ in combined) / len(combined)
    }
    
    stats_path = output_dir / 'stats.json'
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2)
    
    # Create separate NLP and Code training files
    print("\n📁 Creating task-specific datasets...")
    
    # NLP-only
    nlp_dir = output_dir / 'nlp_only'
    nlp_dir.mkdir(exist_ok=True)
    
    random.shuffle(nlp_tagged)
    nlp_train = nlp_tagged[:int(len(nlp_tagged) * 0.95)]
    nlp_val = nlp_tagged[int(len(nlp_tagged) * 0.95):int(len(nlp_tagged) * 0.975)]
    nlp_test = nlp_tagged[int(len(nlp_tagged) * 0.975):]
    
    for split_name, split_data in [('train', nlp_train), ('val', nlp_val), ('test', nlp_test)]:
        with open(nlp_dir / f'{split_name}.txt', 'w', encoding='utf-8') as f:
            for text, _ in split_data:
                f.write(text.replace('\n', ' ') + '\n')
    
    print(f"  NLP-only: train={len(nlp_train)}, val={len(nlp_val)}, test={len(nlp_test)}")
    
    # Code-only
    code_dir = output_dir / 'code_only'
    code_dir.mkdir(exist_ok=True)
    
    random.shuffle(code_tagged)
    code_train = code_tagged[:int(len(code_tagged) * 0.95)]
    code_val = code_tagged[int(len(code_tagged) * 0.95):int(len(code_tagged) * 0.975)]
    code_test = code_tagged[int(len(code_tagged) * 0.975):]
    
    for split_name, split_data in [('train', code_train), ('val', code_val), ('test', code_test)]:
        with open(code_dir / f'{split_name}.txt', 'w', encoding='utf-8') as f:
            for text, _ in split_data:
                f.write(text.replace('\n', ' ') + '\n')
    
    print(f"  Code-only: train={len(code_train)}, val={len(code_val)}, test={len(code_test)}")
    
    # Print summary
    print("\n" + "="*60)
    print("DATA PREPARATION COMPLETE!")
    print("="*60)
    
    print(f"\n📁 Output directory: {output_dir}")
    print(f"\n📊 Dataset Statistics:")
    print(f"  Total samples: {len(combined):,}")
    print(f"  - NLP samples: {len(nlp_texts):,}")
    print(f"  - Code samples: {len(code_texts):,}")
    print(f"  Vocabulary size: {len(vocab):,}")
    print(f"  Average text length: {stats['avg_length']:.0f} characters")
    
    print(f"\n📁 Files created:")
    print(f"  - Combined: train.txt, val.txt, test.txt")
    print(f"  - JSONL: train.jsonl, val.jsonl, test.jsonl")
    print(f"  - Vocabulary: vocab.json")
    print(f"  - Task-specific: nlp_only/, code_only/")
    
    # Show file sizes
    print(f"\n📏 File sizes:")
    for f in sorted(output_dir.glob('*.txt')):
        size_mb = os.path.getsize(f) / (1024 * 1024)
        print(f"  {f.name}: {size_mb:.2f} MB")
    
    return stats

if __name__ == '__main__':
    main()
