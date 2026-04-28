import json
import os

notebook_path = 'sife_ldm_colab_v3.ipynb'
if not os.path.exists(notebook_path):
    print(f"Error: {notebook_path} not found.")
    exit(1)

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

def patch_cell_by_id(nb, cell_id, new_source):
    for cell in nb['cells']:
        if cell.get('metadata', {}).get('id') == cell_id:
            cell['source'] = new_source
            return True
    return False

# --- 1. Patch Setup Cell ---
setup_source = [
    "import os, sys, subprocess, shutil\n",
    "from pathlib import Path\n",
    "\n",
    "# --- 1. Install Missing Dependencies ---\n",
    "print('\ud83d\udce6  Installing/Verifying required libraries...')\n",
    "try:\n",
    "    import flax, optax\n",
    "except ImportError:\n",
    "    os.system('pip install -q flax optax datasets transformers')\n",
    "    print('\u2705  Libraries installed.')\n",
    "\n",
    "# --- 2. Workspace Detection & Sync ---\n",
    "WORKING_DIR = '/content/sife-ldm'\n",
    "GITHUB_REPO = 'https://github.com/vicekha/SIFE-LDM-.git'\n",
    "\n",
    "if os.path.exists('/content/sife') and os.path.exists('/content/scripts'):\n",
    "    WORKING_DIR = '/content'\n",
    "    print('\u2705  Antigravity Extension detected.')\n",
    "else:\n",
    "    if not os.path.exists(WORKING_DIR):\n",
    "        print(f'\ud83d\udca1  Cloning project from {GITHUB_REPO}...')\n",
    "        os.system(f'git clone {GITHUB_REPO} {WORKING_DIR}')\n",
    "    os.chdir(WORKING_DIR)\n",
    "\n",
    "os.environ['PYTHONPATH'] = os.getcwd()\n",
    "sys.path.insert(0, os.getcwd())\n",
    "print(f'\\n\ud83d\udcc2  Working directory: {os.getcwd()}')\n"
]
patch_cell_by_id(nb, 'upload_project', setup_source)

# --- 2. Patch Accelerator Config ---
accel_source = [
    "import os, sys, json, jax, jax.numpy as jnp\n",
    "from sife.model import SIFELDMConfig\n",
    "\n",
    "def get_optimal_config(mode='vision'):\n",
    "    devices = jax.devices()\n",
    "    platform = devices[0].platform\n",
    "    gpu_kind = devices[0].device_kind if platform == 'gpu' else 'cpu'\n",
    "    \n",
    "    if 'A100' in gpu_kind:\n",
    "        return {'batch_size': 128, 'embed_dim': 512, 'max_steps': 100000}\n",
    "    elif 'V100' in gpu_kind:\n",
    "        return {'batch_size': 64, 'embed_dim': 256, 'max_steps': 75000}\n",
    "    else:\n",
    "        return {'batch_size': 32, 'embed_dim': 128, 'max_steps': 50000}\n",
    "\n",
    "OPTIMAL_CFG = get_optimal_config()\n",
    "print(f'\u2705  Acceleration optimized for: {jax.devices()[0].device_kind}')\n"
]
patch_cell_by_id(nb, 'training_mode', accel_source)

# --- 3. Patch Image Generation (Remove Fallbacks) ---
gen_source = [
    "import matplotlib.pyplot as plt\n",
    "if TRAINING_MODE in ('vision', 'both'):\n",
    "    def generate_image(label_id, key, steps=50):\n",
    "        H, W = config.image_size\n",
    "        shape = (1, H, W, config.embed_dim)\n",
    "        ddim = DDIMSampler(diffusion)\n",
    "        lbl = jnp.array([label_id], dtype=jnp.int32)\n",
    "        def model_fn(x, t, ctx):\n",
    "            return model.apply(state.params, x, t, labels=ctx, deterministic=True)\n",
    "        latent = ddim.sample(model_fn, shape, key, lbl, num_steps=steps)\n",
    "        # The new V3.1 architecture is stable for direct decoding\n",
    "        rgb = model.apply(state.params, latent, method=model.decode_images)\n",
    "        return np.clip(np.array(rgb[0]), 0, 1)\n"
]
patch_cell_by_id(nb, 'generate_images', gen_source)

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print(f"Notebook {notebook_path} updated successfully with V3.1 stability fixes.")
