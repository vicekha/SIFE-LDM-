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
    initialize_hierarchical_field, HierarchicalField,
    MultiScaleSIFEUNet2D
)
from .unet import ComplexUNet2D

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


class ImageEncoder(nn.Module):
    """
    Encodes RGB images into complex-valued latent fields.
    """
    features: int
    
    @nn.compact
    def __call__(self, x: Array) -> Array:
        """x: RGB image (B, H, W, 3)"""
        # Linear projection to complex feature space
        # Distribute RGB info into complex features
        from .unet import ComplexLinear
        h = ComplexLinear(self.features)(x.astype(jnp.complex64))
        return h


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
        action: Optional[Array] = None
    ) -> Array:
        """
        Forward pass: predict noise given noisy input and timestep.
        
        Args:
            x: Input complex field of shape (batch, seq_len, embed_dim)
            t: Timestep of shape (batch,)
            context: Optional conditioning context
            mask: Optional attention mask
            deterministic: Whether to use dropout
            abs_phase: Global phase conditioning
            action: Optional agent action
        
        Returns:
            Predicted noise in complex field
        """
        # Multi-scale architecture selection
        if self.config.is_image:
            model = MultiScaleSIFEUNet2D(
                features=tuple(
                    self.config.embed_dim * m
                    for m in self.config.multiscale.feature_multipliers
                ),
                num_heads=self.config.num_heads,
                dropout_rate=self.config.dropout_rate,
                output_features=self.config.embed_dim
            )
        else:
            model = MultiScaleSIFEUNet(
                features=tuple(
                    self.config.embed_dim * m
                    for m in self.config.multiscale.feature_multipliers
                ),
                num_heads=self.config.num_heads,
                dropout_rate=self.config.dropout_rate,
                output_features=self.config.embed_dim,
                num_experts=self.config.num_experts
            )
        
        return model(x, t, context, abs_phase, action, deterministic)
    
    def get_loss(
        self,
        params: dict,
        batch: Dict[str, Array],
        key: PRNGKey,
        diffusion: GaussianDiffusion
    ) -> Array:
        """Compute training loss."""
        x = batch['complex_embedding']
        batch_size = x.shape[0]
        
        # Sample timesteps
        t = jax.random.randint(
            key, (batch_size,), 0, diffusion.num_timesteps
        )
        
        # Sample noise
        noise = jax.random.normal(key, x.shape, dtype=jnp.float32)
        noise = noise + 1j * jax.random.normal(
            jax.random.split(key)[0], x.shape, dtype=jnp.float32
        )
        
        # Add noise
        x_t = diffusion.q_sample(x, t, key, noise)
        
        # Absolute phase for temporal conditioning
        abs_phase = self.config.sife.omega_0 * t.astype(jnp.float32)
        
        # Predict noise
        epsilon_pred = self.apply(
            params, x_t, t,
            context=batch.get('context'),
            deterministic=True,
            abs_phase=abs_phase,
            action=batch.get('action')
        )
        
        # Prediction Loss
        mse_loss = jnp.mean(jnp.abs(epsilon_pred - noise) ** 2)
        
        # Phase Neighbor Coupling Loss (Grammar Physics)
        # We enforce that the denoised phase field is locally smooth.
        # x_0_pred = (x_t - sqrt(1-alpha_t)*epsilon_pred) / sqrt(alpha_t)
        # For simplicity, we can regularize the predicted epsilon's phase 
        # as it approximates the structure of the target field.
        theta = jnp.angle(epsilon_pred)
        
        # Compute cosine similarity between adjacent phases: mean(1 - cos(theta_i - theta_{i+1}))
        phase_diff = theta[:, 1:, :] - theta[:, :-1, :]
        coupling_loss = jnp.mean(1 - jnp.cos(phase_diff))
        
        total_loss = mse_loss + self.config.phase_coupling_lambda * coupling_loss
        
        # Stability Regularization (Landscape Hessian)
        if self.config.stability_lambda > 0:
            from .field import compute_landscape_curvature
            # Estimate x_0 from epsilon_pred
            # (In a real diffusion model, alpha_t would be used here)
            # We use predicted epsilon directly as it captures the field structure
            curvature = compute_landscape_curvature(jnp.abs(epsilon_pred), self.config.sife)
            # Penalize shallow minima: high curature needed
            # Use negative log or similar to push curvature upwards
            stability_loss = jnp.mean(jnp.exp(-curvature))
            total_loss = total_loss + self.config.stability_lambda * stability_loss
        
        return total_loss


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
    
    if config.is_image:
        dummy_shape = (config.batch_size, config.image_size[0], config.image_size[1], config.embed_dim)
    else:
        dummy_shape = (config.batch_size, config.max_seq_len, config.embed_dim)
        
    dummy_x = jax.random.normal(key, dummy_shape)
    dummy_x = dummy_x + 1j * jax.random.normal(jax.random.split(key)[0], dummy_shape)
    dummy_t = jnp.zeros((config.batch_size,), dtype=jnp.int32)
    
    # Initialize
    params = model.init(key, dummy_x, dummy_t, deterministic=True)
    
    return model, params


def create_optimizer(
    config: SIFELDMConfig
) -> optax.GradientTransformation:
    """
    Create the optimizer with learning rate schedule.
    
    Uses AdamW with cosine decay and linear warmup.
    """
    # Learning rate schedule
    def lr_schedule(step):
        return cosine_lr_schedule(
            step,
            config.max_steps,
            config.warmup_steps,
            config.learning_rate
        )
    
    # Combined optimizer with clipping
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(
            learning_rate=lr_schedule,
            b1=0.9,
            b2=0.999,
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
    def loss_fn(params):
        return model.get_loss(params, batch, subkey, diffusion)
    
    loss, grads = value_and_grad(loss_fn)(state.params)
    
    # Apply gradients
    updates, opt_state = optimizer.update(grads, state.opt_state, params=state.params)
    params = optax.apply_updates(state.params, updates)
    
    # Update state
    new_state = TrainState(
        step=state.step + 1,
        params=params,
        opt_state=opt_state,
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
