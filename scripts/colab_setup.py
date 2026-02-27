#!/usr/bin/env python3
"""
Colab Setup Helper
===================

Helps migrate SIFE-LDM to Colab by handling path mapping,
environment detection, and dependency installation.
"""

import os
import sys
import subprocess
from pathlib import Path

def is_colab():
    """Detect if running in a Colab environment."""
    return 'COLAB_JUPYTER_IP' in os.environ or os.path.exists('/content')

def setup_environment():
    """Install dependencies and prepare paths."""
    print("Checking environment...")
    
    if not is_colab():
        print("Not running on Colab. Skipping Colab-specific setup.")
        return

    print("Colab environment detected. Setting up...")
    
    # Define standard Colab paths
    working_dir = Path("/content")
    input_dir = Path("/content/drive/MyDrive")
    
    # 1. Install missing dependencies
    # Colab already has jax, numpy, tqdm, etc. 
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
    if is_colab():
        return "/content"
    else:
        # Default to current directory if not on Colab
        return os.getcwd()

if __name__ == "__main__":
    setup_environment()
