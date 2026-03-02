"""
SIFE-LDM: Complete Model Implementation
=======================================

Integrates all components into a complete SIFE-LDM model:
- Field dynamics (SIFE equations)
- Complex-valued U-Net architecture
- Diffusion process
- Multi-scale hierarchy
- Training and inference

Author: SIFE-LDM Research Team
License: MIT
"""

import jax
import jax.numpy as jnp
from jax import jit, vmap, grad, pmap, value_and_grad
from jax import lax
import flax
import flax.linen as nn
from flax.training import train_state
from flax import struct
import optax
from typing import Tuple, Optional, List, Dict, Any, Callable, NamedTuple, Sequence
from functools import partial
import time
import json
import os

# Import local modules
from .field import (
    SIFEConfig, SIFField, initialize_field, compute_hamiltonian,
    truth_potential, evolve_field, leapfrog_step
)
from .unet import ComplexUNet1D, SIFEUNet
from .diffusion import (
    GaussianDiffusion, DiffusionConfig, DDIMSampler, SIFEDiffusion,
    compute_loss, cosine_lr_schedule
)
from .multiscale import (
    MultiScaleConfig, MultiScaleSIFEUNet, create_multiscale_config,
    initialize_hierarchical_field, HierarchicalField
)

# Type aliases
Array = jnp.ndarray
PRNGKey = jnp.ndarray


@struct.dataclass
class TrainState:
    """Training state for SIFE-LDM."""
    step: int
    params: dict
    opt_state: optax.OptState
    key: Array


class SIFELDMConfig(NamedTuple):
    """Complete configuration for SIFE-LDM."""
    # Field configuration
    sife: SIFEConfig = SIFEConfig()
    
    # Diffusion configuration
    diffusion: DiffusionConfig = DiffusionConfig()
    
    # Multi-scale configuration
    multiscale: MultiScaleConfig = create_multiscale_config()
    
    # Model configuration
    embed_dim: int = 256
    num_heads: int = 8
    num_blocks: int = 4
    dropout_rate: float = 0.1
    
    # Training configuration
    batch_size: int = 8
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_steps: int = 1000
    max_steps: int = 100000
    phase_coupling_lambda: float = 0.1
    stability_lambda: float = 0.01
    symbols_vocab_size: int = 1024
    
    # Sequence configuration
    max_seq_len: int = 1024
    vocab_size: int = 32000
    
    # MoE configuration
    num_experts: int = 0
    num_experts_per_token: int = 1
    
    # Image configuration
    is_image: bool = False
    image_size: Tuple[int, int] = (32, 32)
    
    # Conditional generation configuration
    num_classes: int = 0  # 0 = unconditional; > 0 enables CFG class conditioning


class ImageEncoder(nn.Module):
    """
    Encodes RGB images into complex-valued latent fields.
    """
    features: int
    
    @nn.compact
    def __call__(self, x: Array) -> Array:
        """
        x: RGB image (B, H, W, 3) in range [0, 1]
        
        Maps brightness to amplitude, and hue to phase.
        This provides a physically meaningful starting point for the SIFE field.
        """
        # Split into RGB channels
        r = x[..., 0:1]
        g = x[..., 1:2]
        b = x[..., 2:3]
        
        # Approximate brightness -> Amplitude [0, 1]
        brightness = (r + g + b) / 3.0
        
        # Approximate hue -> Phase [-pi, pi]
        # We define a 2D color vector (u, v) using RGB basis vectors at 0, 120, 240 degrees
        u = r - 0.5 * g - 0.5 * b
        v = (jnp.sqrt(3.0) / 2.0) * (g - b)
        
        # Compute phase angle
        phase = jnp.arctan2(v, u)
        
        # Create initial physically-grounded field
        # Broadcast the single phase/amp to all feature dimensions
        # The subsequent Unet layers will mix these features
        complex_x = brightness * jnp.exp(1j * phase)
        
        # Project using ComplexLinear to distribute into full feature space
        from .unet import ComplexLinear
        h = ComplexLinear(self.features)(complex_x)
        
        # Latent Scaling: Ensure latents have roughly unit variance for diffusion.
        # This is critical for SNR in early training.
        # Standard LDMs use ~0.18 (down-scaling), but since our random init is small, we scale UP.
        # Scale 20.0 targets variance ~0.4, making the signal visible amidst variance-2.0 noise.
        return h * 20.0


class ImageDecoder(nn.Module):
    """
    Decodes complex-valued latent fields back into RGB images.
    """
    @nn.compact
    def __call__(self, x: Array) -> Array:
        """x: Complex field (B, H, W, features)"""
        from .unet import ComplexLinear
        # Project back to 3 channels
        h = nn.Dense(3)(jnp.abs(x)) 
        return jnp.clip(h, 0.0, 1.0)


class LabelEncoder(nn.Module):
    """
    Encodes integer class labels into complex-valued context embeddings
    for class-conditional generation via Phase-Coherent CFG.
    
    Amplitude encodes class salience; phase encodes category geometry.
    """
    num_classes: int
    features: int
    
    @nn.compact
    def __call__(self, labels: Array) -> Array:
        """labels: (B,) int32 → (B, features) complex64"""
        # Separate real/imag embeddings to form a complex class vector
        real_emb = nn.Embed(self.num_classes, self.features)(labels)  # (B, features)
        imag_emb = nn.Embed(self.num_classes, self.features)(labels)  # (B, features)
        # Scale imaginary part to be phase-like (small, [-pi, pi])
        imag_emb = jnp.pi * jnp.tanh(imag_emb)
        return real_emb.astype(jnp.complex64) + 1j * imag_emb.astype(jnp.complex64)


def predict_meta_physics(
    context: Optional[Array], 
    batch_size: int,
    base_v: float = 1.0,
    base_lambda_res: float = 0.5,
    base_omega_0: float = 1.0
) -> Dict[str, Array]:
    """
    Predicts optimal SIFE Hamiltonian physical parameters dynamically based on context.
    Outputs overrides for (v, lambda_res, omega_0) scaled around their base config values.
    """
    if context is None:
        # If no context, just return static base parameters
        return {
            'v': jnp.full((batch_size, 1), base_v),
            'lambda_res': jnp.full((batch_size, 1), base_lambda_res),
            'omega_0': jnp.full((batch_size, 1), base_omega_0),
        }
    
    # Pool context (assume shape B, seq_len, context_dim)
    ctx_pooled = jnp.mean(jnp.abs(context), axis=1) # Convert complex context to real pool
    
    # Deterministic modifiers based on context variance (parameter-free)
    # High variance equals high structural complexity -> requires dynamic physical shifting
    ctx_var = jnp.var(ctx_pooled, axis=-1, keepdims=True) # (B, 1)
    
    v_mod = jnp.clip(ctx_var, 0.0, 1.0)
    lambda_mod = jnp.clip(1.0 - ctx_var, -0.5, 0.5)
    omega_mod = jnp.clip(ctx_var, 0.0, 1.0)
    
    return {
        'v': base_v * (0.5 + v_mod),
        'lambda_res': base_lambda_res + lambda_mod * 0.5,
        'omega_0': base_omega_0 * (0.5 + omega_mod)
    }


class SIFELDM(nn.Module):
    """
    Complete SIFE-LDM model for NLP and coding tasks.
    
    This model integrates:
    1. Complex-valued embeddings that preserve phase information
    2. Multi-scale U-Net architecture
    3. SIFE field dynamics for guidance
    4. Latent diffusion for generation
    5. Vision support (2D lattices)
    """
    config: SIFELDMConfig
    
    @nn.compact
    def __call__(
        self,
        x: Array,
        t: Array,
        context: Optional[Array] = None,
        mask: Optional[Array] = None,
        deterministic: bool = False,
        abs_phase: Optional[Array] = None,
        action: Optional[Array] = None,
        labels: Optional[Array] = None,
        use_context_mask: Optional[Array] = None
    ) -> Array:
        """
        Forward pass: predict noise given noisy input and timestep.
        
        Args:
            x: Input complex field of shape (batch, seq_len, embed_dim) OR RGB image
            t: Timestep of shape (batch,)
            context: Optional conditioning context
            mask: Optional attention mask
            deterministic: Whether to use dropout
            abs_phase: Global phase conditioning
            action: Optional agent action
            labels: Optional class labels for CFG
        
        Returns:
            Predicted noise in complex field
        """
        # 1. Feature Encoding (RGB -> Complex Latent)
        is_rgb = x.ndim == 4 and x.shape[-1] == 3
        if is_rgb:
            x = ImageEncoder(features=self.config.embed_dim, name='image_encoder')(x)
            
        # 2. Contextual Encoding (Labels -> Complex Context)
        if labels is not None and self.config.num_classes > 0:
            label_context = LabelEncoder(
                num_classes=self.config.num_classes, 
                features=self.config.embed_dim,
                name='label_encoder'
            )(labels)
            
            # Apply CFG dropout if mask is provided (for training)
            if use_context_mask is not None:
                mask_val = use_context_mask[:, jnp.newaxis].astype(jnp.complex64)
                label_context = label_context * mask_val
                
            # Add sequence dimension for transformer cross-attention
            label_context = label_context[:, jnp.newaxis, :]
            
            # If explicit context provided, merge; otherwise use labels
            if context is None:
                context = label_context
            else:
                context = jnp.concatenate([context, label_context], axis=1)

        # 3. Patching for Images
        from .unet import ComplexPatchEncoder, UnifiedSIFETransformer, ComplexLinear
        
        is_4d = x.ndim == 4
        orig_shape = x.shape
        
        if self.config.is_image and is_4d:
            B, H, W, C = orig_shape
            P = 4 # Match patch_size in ComplexPatchEncoder
            # Encode down to 1D
            x = ComplexPatchEncoder(patch_size=P, embed_dim=self.config.embed_dim)(x)
            
        # 4. Core Transformer
        model = UnifiedSIFETransformer(
            features=self.config.embed_dim,
            num_heads=self.config.num_heads,
            dropout_rate=self.config.dropout_rate,
            num_experts=self.config.num_experts
        )
        
        out = model(x, t, context, abs_phase, action, mask, deterministic)
        
        # 5. Spatial Decoding (1D -> 4D)
        if self.config.is_image and is_4d:
            # Decode 1D back to 4D
            B, H, W, C = orig_shape
            P = 4
            # Project back to patch features
            out = ComplexLinear(P * P * C)(out)
            # Unflatten
            out = out.reshape(B, H // P, W // P, P, P, C)
            out = jnp.transpose(out, (0, 1, 3, 2, 4, 5))
            out = out.reshape(B, H, W, C)
            
        return out
    
    def get_loss(
        self,
        params: dict,
        batch: Dict[str, Array],
        key: PRNGKey,
        diffusion: GaussianDiffusion
    ) -> Array:
        """Compute training loss."""
        # 1. Input Processing
        x_raw = batch.get('complex_embedding')
        if x_raw is None and 'images' in batch:
            # If raw images provided, encode them within the model's parameters
            # This requires initializing the ImageEncoder params inside the model
            # We apply the model's own encoding logic by calling the model with raw images
            # But get_loss needs the complex latent to noise it.
            # So we use a separate encoding pass.
            x = self.apply(
                params, batch['images'],
                method=self.encode_images
            )
        else:
            x = x_raw
            
        batch_size = x.shape[0]
        
        # 2. Sample timesteps
        t = jax.random.randint(
            key, (batch_size,), 0, diffusion.num_timesteps
        )
        
        # 3. Sample and add noise
        key, noise_key = jax.random.split(key)
        noise = jax.random.normal(noise_key, x.shape, dtype=jnp.float32)
        noise = noise + 1j * jax.random.normal(
            jax.random.split(noise_key)[0], x.shape, dtype=jnp.float32
        )
        
        x_t = diffusion.q_sample(x, t, key, noise)
        
        # 4. Context Processing
        context = batch.get('context')
        labels = batch.get('labels')
        use_context_mask = batch.get('use_context_mask')
        
        # 5. Predict noise
        abs_phase = self.config.sife.omega_0 * t.astype(jnp.float32)
        epsilon_pred = self.apply(
            params, x_t, t,
            context=context,
            labels=labels,
            use_context_mask=use_context_mask,
            deterministic=True,
            abs_phase=abs_phase,
            action=batch.get('action')
        )
        
        # 6. Loss Calculation
        # MSE loss must be real
        mse_loss = jnp.mean(jnp.abs(epsilon_pred - noise) ** 2)
        
        # Physics-based regularization
        x_0_pred = diffusion.predict_x0_from_epsilon(x_t, epsilon_pred, t)
        safe_x_0_pred = x_0_pred + (1e-8 + 1j * 1e-8)
        
        theta = jnp.angle(safe_x_0_pred)
        if theta.ndim == 3: # (B, seq_len, C)
            phase_diff = theta[:, 1:, :] - theta[:, :-1, :]
            coupling_loss = jnp.mean(1 - jnp.cos(phase_diff))
        else: # (B, H, W, C)
            phase_diff_h = theta[:, 1:, :, :] - theta[:, :-1, :, :]
            phase_diff_w = theta[:, :, 1:, :] - theta[:, :, :-1, :]
            coupling_loss = 0.5 * (jnp.mean(1 - jnp.cos(phase_diff_h)) + jnp.mean(1 - jnp.cos(phase_diff_w)))
            
        physics_params = predict_meta_physics(
            context=context, 
            batch_size=batch_size,
            base_v=1.0,
            base_lambda_res=self.config.sife.lambda_res,
            base_omega_0=self.config.sife.omega_0
        )
        
        dyn_lambda = physics_params['lambda_res'].reshape(batch_size, 1)
        if dyn_lambda.ndim < coupling_loss.ndim:
            dyn_lambda = dyn_lambda.reshape(dyn_lambda.shape + (1,) * (coupling_loss.ndim - dyn_lambda.ndim))
            
        # Optimization: Physics Loss Annealing
        # As training progresses (t -> 0 in diffusion, but also globally over steps), 
        # we might want to reduce physics guidance to allow for more stochastic detail.
        # Here we use the diffusion timestep t as a proxy for 'detail level'.
        # Early diffusion steps (large t) should have strong physics.
        # Late diffusion steps (small t) should focus on pixel-perfect noise prediction.
        anneal_factor = (t.astype(jnp.float32) / diffusion.num_timesteps)
        anneal_factor = anneal_factor.reshape((batch_size,) + (1,) * (coupling_loss.ndim - 1))
        
        total_loss = mse_loss + jnp.mean(anneal_factor * dyn_lambda * self.config.phase_coupling_lambda * coupling_loss)
        
        if self.config.stability_lambda > 0:
            from .field import compute_landscape_curvature
            amp = jnp.abs(x_0_pred) + 1e-12
            curvature = compute_landscape_curvature(amp, self.config.sife, dynamic_v=physics_params['v'])
            stability_loss = jnp.mean(jnp.exp(-curvature))
            total_loss = total_loss + self.config.stability_lambda * anneal_factor.mean() * stability_loss
        
        # CRITICAL: Ensure loss is strictly real-valued to avoid JAX/XLA casting warnings
        return jnp.real(total_loss)

    @nn.compact
    def encode_images(self, images: Array) -> Array:
        """Helper for encoding images using model's own parameters."""
        return ImageEncoder(features=self.config.embed_dim, name='image_encoder')(images)


def create_model(
    config: SIFELDMConfig,
    key: PRNGKey
) -> Tuple[SIFELDM, dict]:
    """
    Create and initialize the SIFE-LDM model.
    
    Args:
        config: Model configuration
        key: Random key
    
    Returns:
        Tuple of (model, initial_params)
    """
    model = SIFELDM(config)
    
    # Initialize with raw RGB images and labels so all sub-encoders (ImageEncoder, LabelEncoder)
    # and attention kernels are registered in the parameter tree.
    if config.is_image:
        dummy_x = jnp.zeros((config.batch_size, config.image_size[0], config.image_size[1], 3), dtype=jnp.float32)
    else:
        dummy_x = jnp.zeros((config.batch_size, config.max_seq_len, config.embed_dim), dtype=jnp.complex64)
        
    dummy_t = jnp.zeros((config.batch_size,), dtype=jnp.int32)
    dummy_labels = jnp.zeros((config.batch_size,), dtype=jnp.int32) if config.num_classes > 0 else None
    
    # Initialize model
    params = model.init(key, dummy_x, dummy_t, labels=dummy_labels, deterministic=True)
    
    return model, params


def create_optimizer(
    config: SIFELDMConfig
) -> optax.GradientTransformation:
    """
    Create the optimizer with learning rate schedule.
    
    Uses AdamW with cosine decay and linear warmup.
    """
    # Learning rate schedule using native optax implementation
    # Note: Optax's warmup_cosine_decay_schedule expects decay_steps to be the 
    # number of steps AFTER the warmup. So we calculate it carefully.
    actual_warmup = min(config.warmup_steps, max(1, config.max_steps // 10))
    actual_decay = max(1, config.max_steps - actual_warmup)
    
    print(f"DEBUG: max_steps={config.max_steps}, warmup={actual_warmup}, decay={actual_decay}")
    
    lr_schedule = optax.warmup_cosine_decay_schedule(
        init_value=1e-7,
        peak_value=config.learning_rate,
        warmup_steps=actual_warmup,
        decay_steps=actual_decay,
        end_value=config.learning_rate * 0.05 # Decay to 5% of peak
    )
    
    # Combined optimizer with clipping
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(
            learning_rate=lr_schedule,
            b1=0.9,
            b2=0.99, # Slightly faster decay for noisy gradients
            eps=1e-8,
            weight_decay=config.weight_decay
        )
    )
    
    return optimizer


def create_train_state(
    config: SIFELDMConfig,
    key: PRNGKey
) -> Tuple[SIFELDM, TrainState, GaussianDiffusion]:
    """
    Create the initial training state.
    
    Args:
        config: Model configuration
        key: Random key
    
    Returns:
        Tuple of (model, train_state, diffusion)
    """
    # Create model
    model, params = create_model(config, key)
    
    # Create optimizer
    optimizer = create_optimizer(config)
    opt_state = optimizer.init(params)
    
    # Create diffusion
    diffusion = GaussianDiffusion(config.diffusion)
    
    # Create train state
    state = TrainState(
        step=0,
        params=params,
        opt_state=opt_state,
        key=jax.random.split(key)[0]
    )
    
    return model, state, diffusion


@partial(jit, static_argnums=(0, 3, 4))
def train_step(
    model: SIFELDM,
    state: TrainState,
    batch: Dict[str, Array],
    diffusion: GaussianDiffusion,
    optimizer: optax.GradientTransformation
) -> Tuple[TrainState, Dict[str, float]]:
    """
    Perform one training step.
    
    Args:
        model: SIFE-LDM model
        state: Current training state
        batch: Training batch
        diffusion: Diffusion process
        optimizer: Optimizer
    
    Returns:
        Tuple of (new_state, metrics)
    """
    key, subkey = jax.random.split(state.key)
    
    # Compute loss and gradients
    def loss_fn(p):
        return model.get_loss(p, batch, subkey, diffusion)
        
    loss, grads = jax.value_and_grad(loss_fn)(state.params)
    
    # CRITICAL: Resolve ComplexWarning by forcing real parameters to have real gradients.
    # JAX sometimes yields tiny imaginary residuals (eps*1j) during complex backprop.
    def force_real_grad(g, p):
        if g is None:
            return None
        if jnp.iscomplexobj(p):
            return g  # Keep complex if parameter is complex
        return jnp.real(g) # Strip imaginary noise for real parameters
        
    grads = jax.tree_util.tree_map(force_real_grad, grads, state.params)
    
    # Apply gradients
    updates, new_opt_state = optimizer.update(grads, state.opt_state, params=state.params)
    new_params = optax.apply_updates(state.params, updates)
    
    # Update state
    new_state = TrainState(
        step=state.step + 1,
        params=new_params,
        opt_state=new_opt_state,
        key=key
    )
    
    # Metrics
    metrics = {
        'loss': loss,
        'step': state.step + 1
    }
    
    return new_state, metrics


def train_epoch(
    model: SIFELDM,
    state: TrainState,
    dataset,
    diffusion: GaussianDiffusion,
    optimizer: optax.GradientTransformation,
    config: SIFELDMConfig,
    epoch: int
) -> Tuple[TrainState, Dict[str, float]]:
    """
    Train for one epoch.
    
    Args:
        model: SIFE-LDM model
        state: Current training state
        dataset: Training dataset
        diffusion: Diffusion process
        optimizer: Optimizer
        config: Configuration
        epoch: Current epoch number
    
    Returns:
        Tuple of (new_state, metrics)
    """
    total_loss = 0.0
    num_batches = 0
    
    start_time = time.time()
    
    for batch in dataset:
        state, metrics = train_step(model, state, batch, diffusion, optimizer)
        total_loss += metrics['loss']
        num_batches += 1
        
        # Log progress
        if num_batches % 100 == 0:
            elapsed = time.time() - start_time
            print(f"Epoch {epoch}, Step {metrics['step']}, "
                  f"Loss: {metrics['loss']:.4f}, "
                  f"Time: {elapsed:.2f}s")
    
    avg_loss = total_loss / num_batches
    elapsed = time.time() - start_time
    
    metrics = {
        'avg_loss': avg_loss,
        'epoch': epoch,
        'time': elapsed
    }
    
    return state, metrics


def generate(
    model: SIFELDM,
    params: dict,
    diffusion: GaussianDiffusion,
    key: PRNGKey,
    shape: Tuple[int, ...],
    context: Optional[Array] = None,
    num_steps: int = 50,
    use_sife_guidance: bool = True,
    sife_config: Optional[SIFEConfig] = None
) -> Array:
    """
    Generate samples using the trained model.
    
    Args:
        model: SIFE-LDM model
        params: Model parameters
        diffusion: Diffusion process
        key: Random key
        shape: Shape of samples to generate
        context: Optional conditioning context
        num_steps: Number of sampling steps
        use_sife_guidance: Whether to use SIFE guidance
        sife_config: SIFE config for guidance
    
    Returns:
        Generated samples
    """
    if use_sife_guidance and sife_config is not None:
        # Use SIFE-guided diffusion
        sife_diffusion = SIFEDiffusion(
            diffusion, sife_config,
            guidance_scale_hamiltonian=0.1,
            guidance_scale_truth=0.1
        )
        
        def model_fn(x, t, ctx):
            return model.apply(params, x, t, context=ctx, deterministic=True)
        
        return sife_diffusion.sample(
            model_fn, shape, key, context, num_steps=num_steps
        )
    else:
        # Standard DDIM sampling
        ddim = DDIMSampler(diffusion)
        
        def model_fn(x, t, ctx):
            return model.apply(params, x, t, context=ctx, deterministic=True)
        
        return ddim.sample(
            model_fn, shape, key, context, num_steps=num_steps
        )


# TPU-optimized training functions

def create_pmap_train_step(
    model: SIFELDM,
    diffusion: GaussianDiffusion,
    optimizer: optax.GradientTransformation
):
    """
    Create a pmap-compatible training step for TPU.
    
    This replicates the model across TPU cores and computes
    gradients in parallel.
    """
    @partial(pmap, axis_name='devices')
    def pmap_train_step(state, batch):
        key, subkey = jax.random.split(state.key)
        
        def loss_fn(params):
            return model.get_loss(params, batch, subkey, diffusion)
        
        loss, grads = value_and_grad(loss_fn)(state.params)
        
        # Average gradients across devices
        grads = lax.pmean(grads, axis_name='devices')
        
        # Apply gradients
        updates, opt_state = optimizer.update(grads, state.opt_state, params=state.params)
        params = optax.apply_updates(state.params, updates)
        
        new_state = TrainState(
            step=state.step + 1,
            params=params,
            opt_state=opt_state,
            key=key
        )
        
        # Average loss across devices
        loss = lax.pmean(loss, axis_name='devices')
        
        return new_state, loss
    
    return pmap_train_step


def replicate_state(state: TrainState, num_devices: int) -> TrainState:
    """Replicate training state across devices."""
    return flax.jax_utils.replicate(state)


def unreplicate_state(state: TrainState) -> TrainState:
    """Unreplicate training state from devices."""
    return flax.jax_utils.unreplicate(state)


def train_tpu(
    config: SIFELDMConfig,
    train_dataset,
    val_dataset,
    checkpoint_dir: str = '/tmp/sife-ldm-checkpoints',
    log_dir: str = '/tmp/sife-ldm-logs'
):
    """
    Train SIFE-LDM on TPU.
    
    Args:
        config: Model configuration
        train_dataset: Training dataset
        val_dataset: Validation dataset
        checkpoint_dir: Directory for checkpoints
        log_dir: Directory for logs
    """
    # Initialize TPU
    print("Initializing TPU...")
    jax.distributed.initialize()
    
    num_devices = jax.device_count()
    print(f"Using {num_devices} TPU devices")
    
    # Create model and state
    key = jax.random.PRNGKey(42)
    model, state, diffusion = create_train_state(config, key)
    optimizer = create_optimizer(config)
    
    # Replicate state for pmap
    state = replicate_state(state, num_devices)
    
    # Create pmap training step
    pmap_step = create_pmap_train_step(model, diffusion, optimizer)
    
    # Training loop
    print("Starting training...")
    best_val_loss = float('inf')
    
    for epoch in range(config.max_steps // len(train_dataset)):
        epoch_start = time.time()
        total_loss = 0.0
        num_batches = 0
        
        for batch in train_dataset:
            # Prepare batch for pmap
            batch = {
                k: jnp.stack([v] * num_devices) if v.ndim == 2 else v
                for k, v in batch.items()
            }
            
            state, loss = pmap_step(state, batch)
            total_loss += float(loss[0])
            num_batches += 1
            
            if num_batches % 10 == 0:
                print(f"Epoch {epoch}, Batch {num_batches}, Loss: {loss[0]:.4f}")
        
        avg_loss = total_loss / num_batches
        epoch_time = time.time() - epoch_start
        
        # Validation
        state_unrep = unreplicate_state(state)
        val_loss = validate(model, state_unrep, val_dataset, diffusion)
        
        print(f"Epoch {epoch} completed in {epoch_time:.2f}s")
        print(f"Train Loss: {avg_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        # Save checkpoint if best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(state_unrep, checkpoint_dir, epoch)
        
        # Re-replicate state
        state = replicate_state(state_unrep, num_devices)
    
    return unreplicate_state(state)


def validate(
    model: SIFELDM,
    state: TrainState,
    dataset,
    diffusion: GaussianDiffusion
) -> float:
    """Compute validation loss."""
    total_loss = 0.0
    num_batches = 0
    
    for batch in dataset:
        key = jax.random.PRNGKey(0)
        loss = model.get_loss(state.params, batch, key, diffusion)
        total_loss += float(loss)
        num_batches += 1
    
    return total_loss / num_batches


def save_checkpoint(
    state: TrainState,
    directory: str,
    step: int,
    name: Optional[str] = None
) -> None:
    """Save a training checkpoint."""
    os.makedirs(directory, exist_ok=True)
    
    if name:
        path = os.path.join(directory, name)
    else:
        path = os.path.join(directory, f'checkpoint_{step}')
    
    # Save using flax serialization
    with open(path, 'wb') as f:
        f.write(flax.serialization.to_bytes(state))
    
    print(f"Saved checkpoint to {path}")


def load_checkpoint(
    path: str,
    state: TrainState
) -> TrainState:
    """Load a training checkpoint into an existing state."""
    with open(path, 'rb') as f:
        return flax.serialization.from_bytes(state, f.read())


# Export main classes and functions
__all__ = [
    'SIFELDMConfig',
    'SIFELDM',
    'TrainState',
    'create_model',
    'create_optimizer',
    'create_train_state',
    'train_step',
    'train_epoch',
    'train_tpu',
    'generate',
    'save_checkpoint',
    'load_checkpoint'
]
