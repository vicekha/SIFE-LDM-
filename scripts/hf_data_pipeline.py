#!/usr/bin/env python3
"""
Hugging Face Unified Data Pipeline
==================================

A memory-efficient, streaming-first pipeline for downloading, processing, 
and managing NLP and Coding datasets from the Hugging Face Hub.

Main Features:
- Streaming mode enabled by default for large-scale datasets.
- Support for top NLP datasets (WikiText, C4, OpenWebText).
- Support for key Coding datasets (The Stack, CodeParrot, MBPP).
- Automated cleaning and basic tokenization.
- Support for pushing processed datasets back to the HF Hub.

Usage:
    python hf_data_pipeline.py --mode stream --type nlp --dataset wikitext
    python hf_data_pipeline.py --mode download --type code --dataset codeparrot --max_samples 10000
"""

import argparse
import os
import sys
import json
import logging
from typing import Iterator, Dict, Any, List, Optional
from tqdm import tqdm
from datasets import load_dataset, Dataset, IterableDataset
from huggingface_hub import HfApi, login

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("HF-Pipeline")

class HuggingFacePipeline:
    def __init__(self, cache_dir: str = None):
        self.cache_dir = cache_dir
        self.api = HfApi()

    def get_dataset(
        self, 
        path: str, 
        name: str = None, 
        split: str = "train", 
        streaming: bool = True
    ) -> Iterator[Dict[str, Any]]:
        """Load a dataset from Hugging Face in streaming or download mode."""
        logger.info(f"Loading {path} ({name or 'default'}) split={split} streaming={streaming}")
        try:
            ds = load_dataset(
                path, 
                name=name, 
                split=split, 
                streaming=streaming, 
                cache_dir=self.cache_dir,
                trust_remote_code=True
            )
            return ds
        except Exception as e:
            logger.error(f"Failed to load dataset {path}: {e}")
            return None

    def process_nlp(self, item: Dict[str, Any], text_field: str = "text") -> str:
        """Basic cleaning for NLP data."""
        text = item.get(text_field, "")
        if not text:
            return None
        # Basic normalization (more can be added here)
        text = text.strip()
        if len(text) < 20: # Filter short noise
            return None
        return text

    def process_code(self, item: Dict[str, Any], code_field: str = "content") -> str:
        """Basic cleaning for Coding data."""
        code = item.get(code_field, "")
        if not code:
            return None
        # Potentially add license filtering or comment stripping here
        code = code.strip()
        if len(code) < 50: # Filter trivial snippets
            return None
        return code

    def stream_to_file(self, dataset, output_file: str, max_samples: int = None, type: str = "nlp"):
        """Stream dataset to a local text file."""
        logger.info(f"Streaming data to {output_file}...")
        count = 0
        
        # Mapping fields based on common dataset schemas
        field_map = {
            "wikitext": "text",
            "c4": "text",
            "openwebtext": "text",
            "codeparrot": "content",
            "the-stack": "content",
            "the-stack-smol": "content",
            "mbpp": "code"
        }
        
        # Heuristic to find text field if not in map
        text_field = "text"
        if isinstance(dataset, (Dataset, IterableDataset)):
            features = dataset.features if hasattr(dataset, "features") else {}
            if "content" in features: text_field = "content"
            elif "code" in features: text_field = "code"

        with open(output_file, 'w', encoding='utf-8') as f:
            for item in tqdm(dataset, total=max_samples if not isinstance(dataset, IterableDataset) else None):
                if type == "nlp":
                    processed = self.process_nlp(item, text_field)
                else:
                    processed = self.process_code(item, text_field)
                
                if processed:
                    # Replace interior newlines with spaces for line-by-line format if needed
                    # but for code we might want to preserve them and use a separator.
                    # Here we follow a simple line-delimited text format.
                    clean_line = processed.replace('\n', ' ')
                    f.write(clean_line + '\n')
                    count += 1
                
                if max_samples and count >= max_samples:
                    break
        
        logger.info(f"Finished. Saved {count} samples to {output_file}")

    def push_to_hub(self, file_path: str, repo_id: str, private: bool = True):
        """Upload a processed file or directory to the HF Hub."""
        logger.info(f"Uploading {file_path} to HF Hub as {repo_id}...")
        try:
            self.api.upload_file(
                path_or_fileobj=file_path,
                path_in_repo=os.path.basename(file_path),
                repo_id=repo_id,
                repo_type="dataset",
                commit_message=f"Upload processed data: {os.path.basename(file_path)}"
            )
            logger.info("Upload successful.")
        except Exception as e:
            logger.error(f"Upload failed: {e}")

def main():
    parser = argparse.ArgumentParser(description="Hugging Face Data Pipeline")
    parser.add_argument("--type", choices=["nlp", "code"], required=True)
    parser.add_argument("--dataset", type=str, required=True, help="HF dataset path (e.g., 'wikitext')")
    parser.add_argument("--config", type=str, default=None, help="HF dataset config (e.g., 'wikitext-103-raw-v1')")
    parser.add_argument("--mode", choices=["stream", "download"], default="stream")
    parser.add_argument("--max_samples", type=int, default=1000)
    parser.add_argument("--output", type=str, default="data/processed_hf.txt")
    parser.add_argument("--push", type=str, help="HF repo_id to push to (requires login)")
    parser.add_argument("--cache", type=str, default=None, help="Cache directory")

    args = parser.parse_args()

    pipeline = HuggingFacePipeline(cache_dir=args.cache)
    
    # Load
    dataset = pipeline.get_dataset(
        args.dataset, 
        name=args.config, 
        streaming=(args.mode == "stream")
    )
    
    if not dataset:
        sys.exit(1)

    # Process and Save
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    pipeline.stream_to_file(
        dataset, 
        args.output, 
        max_samples=args.max_samples, 
        type=args.type
    )

    # Push
    if args.push:
        pipeline.push_to_hub(args.output, args.push)

if __name__ == "__main__":
    main()
