"""
SIFE-LDM: Structured Intelligence Field Latent Diffusion Model
==============================================================

A physics-based generative model that unifies diffusion models
with coherent dynamics of classical field theory.

Main Components:
- field: SIFE field equations and dynamics
- unet: Complex-valued U-Net architecture
- diffusion: Latent diffusion process with DDIM sampling
- tokenizer: NLP/coding tokenization with complex field embedding
- multiscale: Multi-scale nested architecture
- model: Complete SIFE-LDM model and training utilities

Example Usage:
    from sife import SIFELDM, SIFELDMConfig, create_train_state
    
    # Create configuration
    config = SIFELDMConfig()
    
    # Initialize model
    key = jax.random.PRNGKey(42)
    model, state, diffusion = create_train_state(config, key)
    
    # Generate samples
    samples = generate(model, state.params, diffusion, key, shape)

Author: SIFE-LDM Research Team
License: MIT
"""

__version__ = '0.1.0'

from .field import (
    SIFEConfig,
    SIFField,
    initialize_field,
    compute_hamiltonian,
    truth_potential,
    evolve_field,
    leapfrog_step
)

from .unet import (
    ComplexLinear,
    ComplexConv,
    ComplexConv1D,
    ComplexLayerNorm,
    ComplexModReLU,
    ComplexSelfAttention,
    ComplexTransformerBlock,
    ComplexUNet1D,
    SIFEUNet
)

from .diffusion import (
    DiffusionConfig,
    GaussianDiffusion,
    DDIMSampler,
    SIFEDiffusion,
    compute_loss,
    cosine_lr_schedule
)

from .tokenizer import (
    Vocabulary,
    ComplexFieldEmbedding,
    SIFETokenizer,
    DataPipeline,
    SIFEDataset
)

from .multiscale import (
    MultiScaleConfig,
    HierarchicalField,
    MultiScaleSIFEUNet,
    create_multiscale_config
)

from .model import (
    SIFELDMConfig,
    SIFELDM,
    TrainState,
    create_model,
    create_train_state,
    train_step,
    train_tpu,
    generate,
    save_checkpoint,
    load_checkpoint
)

__all__ = [
    # Field
    'SIFEConfig',
    'SIFField',
    'initialize_field',
    'compute_hamiltonian',
    'truth_potential',
    'evolve_field',
    'leapfrog_step',
    
    # U-Net
    'ComplexLinear',
    'ComplexConv',
    'ComplexConv1D',
    'ComplexLayerNorm',
    'ComplexModReLU',
    'ComplexSelfAttention',
    'ComplexTransformerBlock',
    'ComplexUNet1D',
    'SIFEUNet',
    
    # Diffusion
    'DiffusionConfig',
    'GaussianDiffusion',
    'DDIMSampler',
    'SIFEDiffusion',
    'compute_loss',
    'cosine_lr_schedule',
    
    # Tokenizer
    'Vocabulary',
    'ComplexFieldEmbedding',
    'SIFETokenizer',
    'DataPipeline',
    'SIFEDataset',
    
    # Multi-scale
    'MultiScaleConfig',
    'HierarchicalField',
    'MultiScaleSIFEUNet',
    'create_multiscale_config',
    
    # Model
    'SIFELDMConfig',
    'SIFELDM',
    'TrainState',
    'create_model',
    'create_train_state',
    'train_step',
    'train_tpu',
    'generate',
    'save_checkpoint',
    'load_checkpoint'
]
