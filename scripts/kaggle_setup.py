#!/usr/bin/env python3
"""
Kaggle Setup Helper
===================

Helps migrate SIFE-LDM to Kaggle by handling path mapping,
environment detection, and dependency installation.
"""

import os
import sys
import subprocess
from pathlib import Path

def is_kaggle():
    """Detect if running in a Kaggle environment."""
    return 'KAGGLE_KERNEL_RUN_TYPE' in os.environ or os.path.exists('/kaggle/working')

def setup_environment():
    """Install dependencies and prepare paths."""
    print("Checking environment...")
    
    if not is_kaggle():
        print("Not running on Kaggle. Skipping Kaggle-specific setup.")
        return

    print("Kaggle environment detected. Setting up...")
    
    # Define standard Kaggle paths
    working_dir = Path("/kaggle/working")
    input_dir = Path("/kaggle/input")
    
    # 1. Install missing dependencies
    # Kaggle already has jax, numpy, tqdm, etc. 
    # But might need datasets/transformers if not in the image.
    try:
        import datasets
        import transformers
    except ImportError:
        print("Installing additional dependencies...")
        subprocess.run([sys.executable, "-m", "pip", "install", "datasets", "transformers", "huggingface_hub", "-q"], check=True)

    # 2. Create necessary directories
    (working_dir / "checkpoints").mkdir(exist_ok=True)
    (working_dir / "logs").mkdir(exist_ok=True)
    (working_dir / "output").mkdir(exist_ok=True)
    
    print(f"Working directory prepared at: {working_dir}")
    print("Setup complete.")

def get_base_path():
    """Get the appropriate base path for the environment."""
    if is_kaggle():
        return "/kaggle/working"
    else:
        # Default to current directory if not on Kaggle
        return os.getcwd()

if __name__ == "__main__":
    setup_environment()
