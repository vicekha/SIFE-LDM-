import streamlit as st
import os
import sys
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
import time
import argparse

# Add parent directory to path to import sife modules
sys.path.insert(0, str(Path(__file__).parent))

from sife.model import SIFELDM, SIFELDMConfig, load_checkpoint
from sife.field import SIFEConfig
from sife.diffusion import DiffusionConfig, GaussianDiffusion, DDIMSampler
from sife.multiscale import create_multiscale_config
from sife.tokenizer import Vocabulary, SIFETokenizer
from inference import load_model, load_tokenizer, apply_attractor_relaxation, decode_embedding

# Setup Page Configuration
st.set_page_config(
    page_title="SIFE-LDM Dashboard",
    page_icon="🌌",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
.stApp {
    background-color: #0E1117;
    color: #FAFAFA;
}
.stButton>button {
    background-color: #4CAF50;
    color: white;
    font-weight: bold;
    border-radius: 8px;
    border: none;
    padding: 10px 24px;
}
.stButton>button:hover {
    background-color: #45a049;
    color: white;
}
.metric-container {
    background-color: #1E2329;
    border-radius: 8px;
    padding: 15px;
    margin-bottom: 10px;
    border: 1px solid #333;
}
h1, h2, h3 {
    color: #E2E8F0;
}
</style>
""", unsafe_allow_html=True)

st.title("🌌 SIFE-LDM Visual Dashboard")
st.markdown("*Structured Intelligence Field Latent Diffusion Model*")

@st.cache_resource
def get_model_and_tokenizer(checkpoint_path, vocab_path):
    # Dummy args object
    class Args:
        pass
    
    with st.spinner("Loading Model & Tokenizer..."):
        model, state, diffusion, config = load_model(checkpoint_path, None)
        tokenizer = load_tokenizer(vocab_path, config)
        return model, state, diffusion, config, tokenizer

# Sidebar configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    
    checkpoints_dir = os.path.join(Path(__file__).parent, "checkpoints")
    available_ckpts = []
    if os.path.exists(checkpoints_dir):
        # Recursively find directories with 'checkpoint' prefix or specific naming
        for root, dirs, files in os.walk(checkpoints_dir):
            if any(f == 'checkpoint' for f in files):
                available_ckpts.append(root)

    # Use a default if nothing is found
    ckpt_path = st.text_input("Checkpoint Path", value="checkpoints/best" if not available_ckpts else available_ckpts[0])
    vocab_path = st.text_input("Vocabulary Path", value="vocab.json")
    
    st.markdown("---")
    st.subheader("Generation Settings")
    max_length = st.number_input("Max Length", min_value=16, max_value=2048, value=256, step=16)
    num_steps = st.slider("Diffusion Steps", min_value=10, max_value=200, value=50, step=10)
    
    st.markdown("---")
    st.subheader("Physics & Guidance")
    eta = st.slider("DDIM η (Noise)", min_value=0.0, max_value=1.0, value=0.0, step=0.1)
    
    st.markdown("---")
    if st.button("Load Engine"):
        if not os.path.exists(ckpt_path):
            st.error(f"Checkpoint not found at: {ckpt_path}")
        elif not os.path.exists(vocab_path):
            st.error(f"Vocabulary not found at: {vocab_path}")
        else:
            get_model_and_tokenizer(ckpt_path, vocab_path)
            st.success("Loaded correctly!")

# Main Area
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Command Center")
    prompt = st.text_area("Initial Prompt (Context)", value="def fibonacci(n):", height=150)
    generate_btn = st.button("🚀 Generate Response", use_container_width=True)
    
    output_placeholder = st.empty()

with col2:
    st.subheader("Physics Telemetry")
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    metrics_placeholder = st.container()

def run_visual_generation(model, state, diffusion, tokenizer, prompt, config, max_length, num_steps, eta):
    key = jax.random.PRNGKey(int(time.time()))
    
    # 1. Encode prompt
    inner_ids = tokenizer.vocab.encode(prompt)
    prompt_ids = [tokenizer.vocab.bos_id] + inner_ids
    prompt_len = len(prompt_ids)
    
    token_emb = tokenizer.token_embedding(jnp.array(prompt_ids)[jnp.newaxis, :])[0]
    pos_emb = tokenizer.positional_embedding(prompt_len)
    context = (token_emb * pos_emb)[jnp.newaxis, :, :]
    
    if prompt_len >= max_length:
        st.warning(f"Prompt too long ({prompt_len} tokens). Truncating.")
        prompt_len = max_length - 1
        context = context[:, :prompt_len, :]

    # 2. Setup Generation Field
    key, subkey = jax.random.split(key)
    noise_shape = (1, max_length, config.embed_dim)
    
    x = jax.random.normal(subkey, noise_shape, dtype=jnp.float32)
    x = x + 1j * jax.random.normal(jax.random.split(subkey)[0], noise_shape, dtype=jnp.float32)
    
    # 3. Sampling
    ddim = DDIMSampler(diffusion)
    c = diffusion.num_timesteps // num_steps
    ddim_timesteps = jnp.arange(0, diffusion.num_timesteps, c)[::-1]
    
    def model_fn(x, t, ctx=None):
        return model.apply(state.params, x, t, context=ctx, deterministic=True)

    status_text.text(f"Phase 1: Diffusion Denoising ({num_steps} steps)")
    
    for i, t in enumerate(ddim_timesteps):
        t_prev = ddim_timesteps[i + 1] if i + 1 < len(ddim_timesteps) else -1
        
        # Denoising step
        key, subkey = jax.random.split(key)
        x = ddim.ddim_step(model_fn, x, int(t), int(t_prev), subkey)
        
        # In-painting: Fix the prefix
        if t_prev >= 0:
            alpha_t = diffusion.alphas_cumprod[int(t)]
            key, subkey1 = jax.random.split(key)
            noise = jax.random.normal(subkey1, context.shape, dtype=jnp.float32)
            noise = noise + 1j * jax.random.normal(jax.random.split(subkey1)[0], context.shape, dtype=jnp.float32)
            
            noisy_prompt = jnp.sqrt(alpha_t) * context + jnp.sqrt(1 - alpha_t) * noise
            x = x.at[:, :prompt_len, :].set(noisy_prompt)
        else:
            x = x.at[:, :prompt_len, :].set(context)
            
        progress_bar.progress((i + 1) / len(ddim_timesteps))
        
        if i % 5 == 0:
            with metrics_placeholder:
                amp_mean = float(jnp.mean(jnp.abs(x)))
                phase_std = float(jnp.std(jnp.angle(x)))
                st.markdown(f"**Step {i+1}/{num_steps}**")
                st.markdown(f"- Mean Amplitude: `{amp_mean:.4f}`")
                st.markdown(f"- Phase Deviation: `{phase_std:.4f}`")
        
    status_text.text("Phase 2: AGI Physical Relaxation (System 2)")
    
    # Phase 2: System 2 Reasoning
    max_thought_steps = 100
    thought_steps = 0
    is_stable = False
    
    progress_bar.progress(0)
    
    while not is_stable and thought_steps < max_thought_steps:
        x, is_stable = apply_attractor_relaxation(x, config, eta=0.1, num_steps=1)
        x = x.at[:, :prompt_len, :].set(context)
        thought_steps += 1
        
        progress_bar.progress(thought_steps / max_thought_steps)
        
        if thought_steps % 10 == 0:
            with metrics_placeholder:
                # Clear previous metrics and show relaxation
                st.empty()
                st.markdown(f"**Relaxation Step {thought_steps}**")
                st.markdown(f"- Stable: `{'✅ Yes' if is_stable else '⏳ Computing...'}`")
                
                # Show partial decode
                partial_text = decode_embedding(x[0], tokenizer)
                output_placeholder.code(partial_text, language="python")

    status_text.text("Generation Complete!")
    progress_bar.progress(1.0)
    
    final_text = decode_embedding(x[0], tokenizer)
    return final_text

if generate_btn:
    try:
        model, state, diffusion, config, tokenizer = get_model_and_tokenizer(ckpt_path, vocab_path)
        
        with output_placeholder.container():
            st.info("Generating field lattice...")
        
        final_output = run_visual_generation(
            model=model,
            state=state,
            diffusion=diffusion,
            tokenizer=tokenizer,
            prompt=prompt,
            config=config,
            max_length=max_length,
            num_steps=num_steps,
            eta=eta
        )
        
        output_placeholder.success("Success!")
        st.subheader("Final Result")
        st.code(final_output, language="python")
        
    except Exception as e:
        st.error(f"Error during generation: {e}")
