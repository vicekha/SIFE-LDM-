import json
import os

notebook_path = 'sife_ldm_colab.ipynb'
if not os.path.exists(notebook_path):
    print(f"Error: {notebook_path} not found.")
    exit(1)

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# 0: Title & Strategy Pivot
nb['cells'][0]['source'] = [
    "# SIFE-LDM V2 \u2014 NLP & Code Generation Pivot\n",
    "\n",
    "**Architecture:** Unified SIFE Transformer (Complex-valued) with Phase-Coherent Diffusion.\n",
    "\n",
    "**Dataset:** WikiText-103, CodeAlpaca, MBPP.\n",
    "\n",
    "**Hardware:** Automatically scales for **TPU v2-8**, **L4**, or **A100** GPUs.\n",
    "\n",
    "**Goal:** Generate coherent text and programming logic using physics-guided latent fields."
]

# 2: Setup (Check for train.py)
nb['cells'][2]['source'] = [
    "import os, sys, subprocess, shutil\n",
    "from pathlib import Path\n",
    "\n",
    "# --- 1. Install Missing Dependencies ---\n",
    "print('\ud83d\udce6  Installing/Verifying required libraries...')\n",
    "os.system('pip install -q flax optax datasets')\n",
    "\n",
    "# --- 2. Workspace Detection & Sync ---\n",
    "WORKING_DIR = '/content/sife-ldm'\n",
    "GITHUB_REPO = 'https://github.com/vicekha/SIFE-LDM-.git'\n",
    "\n",
    "if os.path.exists('/content/sife') and os.path.exists('/content/scripts'):\n",
    "    WORKING_DIR = '/content'\n",
    "else:\n",
    "    if not os.path.exists(WORKING_DIR):\n",
    "        os.system(f'git clone {GITHUB_REPO} {WORKING_DIR}')\n",
    "    os.chdir(WORKING_DIR)\n",
    "    os.system('git pull origin main')\n",
    "\n",
    "os.environ['PYTHONPATH'] = os.getcwd()\n",
    "\n",
    "# Verify critical files\n",
    "critical = ['sife/model.py', 'train.py', 'scripts/quick_start_data.py']\n",
    "missing = [f for f in critical if not os.path.exists(f)]\n",
    "if missing:\n",
    "    print(f'\u274c  MISSING FILES: {missing}')\n",
    "else:\n",
    "    print('\u2705  Environment ready for SIFE-NLP training.')"
]

# 6: Data Download (NLP/Code)
nb['cells'][6]['source'] = [
    "import os\n",
    "os.environ['PYTHONPATH'] = '.'\n",
    "\n",
    "print('\ud83d\udce5  Downloading NLP and Code datasets (WikiText-103, CodeAlpaca)...')\n",
    "os.system('python scripts/quick_start_data.py --mode both')\n",
    "\n",
    "data_path = './datasets/nlp/nlp_combined_train.txt'\n",
    "if os.path.exists(data_path):\n",
    "    print(f'\u2705  Data ready at: {data_path}')\n",
    "    with open(data_path, 'r', encoding='utf-8') as f:\n",
    "        preview = [next(f) for _ in range(5)]\n",
    "        print('\\nPreview:\\n' + ''.join(preview))\n",
    "else:\n",
    "    print('\u274c  Download failed.')"
]

# 8: Training (NLP)
nb['cells'][8]['source'] = [
    "import os, json\n",
    "os.environ['PYTHONPATH'] = '.'\n",
    "\n",
    "with open('./accel_config.json', 'r') as f:\n",
    "    accel_cfg = json.load(f)\n",
    "\n",
    "print(f\"\ud83d\ude80  Training NLP-Transformer on {accel_cfg['accel_name']}...\")\n",
    "print(f\"    Batch Size: {accel_cfg['batch_size']}\")\n",
    "\n",
    "cmd = (f\"python train.py --data ./datasets/nlp/nlp_combined_train.txt \"\n",
    "       f\"--embed_dim {accel_cfg['embed_dim']} --batch_size {accel_cfg['batch_size']} \"\n",
    "       f\"--max_steps {accel_cfg['max_steps']} --max_seq_len 1024\")\n",
    "\n",
    "os.system(cmd)"
]

# 12: Generation (Inference)
nb['cells'][11]['source'] = [
    "## Cell 6: SIFE-Transformer Inference\n",
    "Generates text using the physics-guided `inference.py` script."
]
nb['cells'][12]['source'] = [
    "import os\n",
    "os.environ['PYTHONPATH'] = '.'\n",
    "\n",
    "PROMPT = 'Biological intelligence is rooted in'\n",
    "print(f'Generating from prompt: \"{PROMPT}\"')\n",
    "\n",
    "os.system(f'python inference.py --checkpoint checkpoints/checkpoint_final --prompt \"{PROMPT}\" --num_steps 50')"
]

# Remove extra vision cells if they exist (13, 14, 15, 16)
if len(nb['cells']) > 13:
    nb['cells'] = nb['cells'][:13]

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("Colab Notebook updated for NLP integration.")
