import json
import os

notebook_path = 'sife_ldm_colab.ipynb'
if not os.path.exists(notebook_path):
    print(f"Error: {notebook_path} not found.")
    exit(1)

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# --- Patch Cell 1 (Setup) at Index 2 ---
nb['cells'][2]['source'] = [
    "import os, sys, subprocess, shutil\n",
    "from pathlib import Path\n",
    "\n",
    "# --- 1. Install Missing Dependencies ---\n",
    "print('\ud83d\udce6  Installing/Verifying required libraries...')\n",
    "try:\n",
    "    import flax, optax\n",
    "except ImportError:\n",
    "    os.system('pip install -q flax optax')\n",
    "    print('\u2705  Libraries installed.')\n",
    "\n",
    "# --- 2. Workspace Detection & Sync ---\n",
    "WORKING_DIR = '/content/sife-ldm'\n",
    "GITHUB_REPO = 'https://github.com/vicekha/SIFE-LDM-.git'\n",
    "\n",
    "def find_project_root(search_dir):\n",
    "    for root, dirs, files in os.walk(search_dir):\n",
    "        if 'scripts' in dirs and 'sife' in dirs:\n",
    "            return root\n",
    "    return None\n",
    "\n",
    "# Check if we are in an Antigravity synced environment\n",
    "if os.path.exists('/content/sife') and os.path.exists('/content/scripts'):\n",
    "    WORKING_DIR = '/content'\n",
    "    print('\u2705  Antigravity Extension detected.')\n",
    "else:\n",
    "    # Try to find or clone the repo\n",
    "    found_root = find_project_root('/content')\n",
    "    if found_root:\n",
    "        WORKING_DIR = found_root\n",
    "        print(f'\ud83d\udcc2  Found project at: {WORKING_DIR}')\n",
    "        os.chdir(WORKING_DIR)\n",
    "        print('\ud83d\udd04  Syncing latest changes from GitHub...')\n",
    "        os.system('git pull origin main')\n",
    "    else:\n",
    "        print(f'\ud83d\udca1  Cloning project from {GITHUB_REPO}...')\n",
    "        os.system(f'git clone {GITHUB_REPO} {WORKING_DIR}')\n",
    "        os.chdir(WORKING_DIR)\n",
    "\n",
    "os.environ['PYTHONPATH'] = os.getcwd()\n",
    "print(f'\\n\ud83d\udcc2  Working directory set to: {os.getcwd()}')\n",
    "print(f'\ud83d\udcc4  Files in root: {os.listdir(\".\")[:12]}...')\n",
    "\n",
    "# Verify critical files\n",
    "critical = ['sife/model.py', 'scripts/train_vision_gpu.py', 'scripts/download_cifar.py']\n",
    "missing = [f for f in critical if not os.path.exists(f)]\n",
    "if missing:\n",
    "    print(f'\\n\u274c  CRITICAL ERROR: Missing files: {missing}')\n",
    "else:\n",
    "    print('\\n\u2705  Environment ready for SIFE-LDM training.')"
]

# --- Patch Cell 2 (Accelerator Detection) at Index 4 (usually index 4 in nbformat 4 with markdown headers) ---
# Note: In the actual notebook, Cells 1, 3, 5, 7 are markdown. Cells 0, 2, 4, 6 are code?
# Let's count properly:
# 0: Markdown (Title)
# 1: Markdown (Cell 1 header)
# 2: Code (Setup) - [INDEX 2]
# 3: Markdown (Cell 2 header)
# 4: Code (TPU Init) - [INDEX 4]
# 5: Markdown (Cell 3 header)
# 6: Code (Data) - [INDEX 6]
# 7: Markdown (Cell 4 header)
# 8: Code (Training) - [INDEX 8]

nb['cells'][4]['source'] = [
    "import os, sys, json, jax, jax.numpy as jnp\n",
    "\n",
    "# Prevent JAX from pre-allocating all memory\n",
    "os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'\n",
    "os.environ['XLA_PYTHON_CLIENT_ALLOCATOR'] = 'platform'\n",
    "\n",
    "print(f'Python: {sys.version.split()[0]}')\n",
    "print(f'JAX:    {jax.__version__}')\n",
    "\n",
    "def setup_accelerator_and_scale():\n",
    "    \"\"\"Detect TPU/GPU type and return optimal training params.\"\"\"\n",
    "    try:\n",
    "        import jax.tools.colab_tpu\n",
    "        jax.tools.colab_tpu.setup_tpu()\n",
    "    except (KeyError, ImportError):\n",
    "        pass\n",
    "        \n",
    "    devices = jax.devices()\n",
    "    platform = devices[0].platform\n",
    "    num_cores = len(devices)\n",
    "    \n",
    "    if platform == 'tpu':\n",
    "        tpu_name = str(devices[0])\n",
    "        print(f'\\n\ud83d\udda5\ufe0f  TPU Detected: {tpu_name} ({num_cores} cores)')\n",
    "        if num_cores >= 8:\n",
    "            cfg = {'batch_size': 128, 'embed_dim': 256, 'base_features': 128,\n",
    "                   'max_steps': 100000, 'accel_name': 'TPU v2-8', 'num_cores': num_cores}\n",
    "        else:\n",
    "            cfg = {'batch_size': 64, 'embed_dim': 192, 'base_features': 96,\n",
    "                   'max_steps': 75000, 'accel_name': 'TPU', 'num_cores': num_cores}\n",
    "    \n",
    "    elif platform == 'gpu':\n",
    "        gpu_kind = devices[0].device_kind\n",
    "        print(f'\\n\ud83d\udda5\ufe0f  GPU Detected: {gpu_kind} ({num_cores} device)')\n",
    "        \n",
    "        if 'A100' in gpu_kind:\n",
    "            # A100 40GB/80GB can handle much larger embed dims and batch sizes\n",
    "            cfg = {'batch_size': 128, 'embed_dim': 512, 'base_features': 128,\n",
    "                   'max_steps': 100000, 'accel_name': 'A100 GPU', 'num_cores': num_cores}\n",
    "            print('\ud83d\ude80  NVIDIA A100 Detected \u2014 Enabling ULTRA-PERFORMANCE configuration')\n",
    "        elif 'V100' in gpu_kind:\n",
    "            cfg = {'batch_size': 96, 'embed_dim': 256, 'base_features': 128,\n",
    "                   'max_steps': 80000, 'accel_name': 'V100 GPU', 'num_cores': num_cores}\n",
    "            print('\u26a1  NVIDIA V100 Detected \u2014 Enabling HIGH-PERFORMANCE configuration')\n",
    "        elif 'L4' in gpu_kind:\n",
    "            cfg = {'batch_size': 64, 'embed_dim': 192, 'base_features': 96,\n",
    "                   'max_steps': 75000, 'accel_name': 'L4 GPU', 'num_cores': num_cores}\n",
    "            print('\u26a1  NVIDIA L4 Detected \u2014 Enabling MEDIUM configuration')\n",
    "        else:\n",
    "            cfg = {'batch_size': 32, 'embed_dim': 128, 'base_features': 64,\n",
    "                   'max_steps': 50000, 'accel_name': 'T4 GPU', 'num_cores': num_cores}\n",
    "            print(f'\ud83d\udca1  Standard GPU ({gpu_kind}) Detected \u2014 Enabling BASE configuration')\n",
    "            \n",
    "    else:\n",
    "        print('\\n\u26a0\ufe0f  No Accelerator detected! Using CPU.')\n",
    "        cfg = {'batch_size': 4, 'embed_dim': 64, 'base_features': 32,\n",
    "               'max_steps': 10000, 'accel_name': 'CPU', 'num_cores': 0}\n",
    "               \n",
    "    print(f'\\n\ud83d\udccb  Optimization Profile:')\n",
    "    for k, v in cfg.items():\n",
    "        print(f'    {k:15s} = {v}')\n",
    "    return cfg\n",
    "\n",
    "ACCEL_CONFIG = setup_accelerator_and_scale()\n",
    "with open('./accel_config.json', 'w') as f:\n",
    "    json.dump(ACCEL_CONFIG, f, indent=2)\n",
    "print(f'\\n\u2705  Optimization profile saved.')"
]

# --- Patch Cell 8 (Benchmark) at Index 16 ---
nb['cells'][16]['source'] = [
    "import sys, os, pickle, jax, jax.numpy as jnp\n",
    "os.environ['PYTHONPATH'] = '.'\n",
    "sys.path.insert(0, '.')\n",
    "\n",
    "from sife.model import SIFELDM, SIFELDMConfig, create_model\n",
    "from sife.field import SIFEConfig\n",
    "from sife.diffusion import DiffusionConfig, GaussianDiffusion, EulerMaruyamaSampler\n",
    "from sife.multiscale import create_multiscale_config\n",
    "\n",
    "# Load checkpoint\n",
    "ckpt_path = 'checkpoints/vision/final_vision_model.pkl'\n",
    "if not os.path.exists(ckpt_path):\n",
    "    print(f'\u26a0\ufe0f Checkpoint not found at {ckpt_path}. Run training first!')\n",
    "else:\n",
    "    with open(ckpt_path, 'rb') as f:\n",
    "        data = pickle.load(f)\n",
    "    params = data.params if hasattr(data, 'params') else data\n",
    "\n",
    "    # Correct Config for V2 Architecture\n",
    "    config = SIFELDMConfig(\n",
    "        sife=SIFEConfig(), \n",
    "        diffusion=DiffusionConfig(num_timesteps=1000),\n",
    "        multiscale=create_multiscale_config(num_levels=3, base_features=64),\n",
    "        is_image=True, \n",
    "        image_size=(32, 32), \n",
    "        embed_dim=128, \n",
    "        batch_size=4,\n",
    "        num_classes=100  # Enable V2 class conditioning\n",
    "    )\n",
    "    \n",
    "    key = jax.random.PRNGKey(42)\n",
    "    model, _ = create_model(config, key)\n",
    "    diffusion = GaussianDiffusion(DiffusionConfig(num_timesteps=1000))\n",
    "    sampler = EulerMaruyamaSampler(diffusion)\n",
    "\n",
    "    def model_fn(x, t, context=None):\n",
    "        return model.apply(params, x, t, context=context, deterministic=True)\n",
    "\n",
    "    # Run 5 samples and measure steps saved\n",
    "    shape = (4, 32, 32, 128)\n",
    "    results = []\n",
    "    for trial in range(5):\n",
    "        key, subkey = jax.random.split(key)\n",
    "        _, steps_used = sampler.sample(\n",
    "            model_fn, shape, subkey,\n",
    "            num_steps=100,\n",
    "            sife_config=config.sife,\n",
    "            stability_threshold=0.5\n",
    "        )\n",
    "        pct_saved = (100 - steps_used) / 100 * 100\n",
    "        results.append((steps_used, pct_saved))\n",
    "        print(f'Trial {trial+1}: {steps_used}/100 steps used \u2192 {pct_saved:.1f}% saved')\n",
    "\n",
    "    avg_steps = sum(r[0] for r in results) / len(results)\n",
    "    avg_saved = sum(r[1] for r in results) / len(results)\n",
    "    print(f'\\nAverage: {avg_steps:.1f} steps, {avg_saved:.1f}% compute saved via attractor stopping')"
]

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("Notebook patched successfully.")
