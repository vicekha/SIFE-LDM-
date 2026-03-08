"""
Adaptive Norm-Distribution Interface (ANDI) Optimizer
Implemented as an Optax Gradient Transformation in JAX.

Based on the paper: "ANDI: Adaptive Norm-Distribution Interface" by Vladimer Khasia (2025)
"""

import math
import jax
import jax.numpy as jnp
import optax
from typing import Any, NamedTuple, Optional, Tuple, Callable


class ANDIState(NamedTuple):
    """Empty state for the ANDI preconditioner since it is stateless."""
    pass


def scale_by_andi(
    dim_threshold: int = 32,
    epsilon: float = 1e-8
) -> optax.GradientTransformation:
    """
    Applies the ANDI (Adaptive Norm-Distribution Interface) preconditioning step.
    
    ANDI equilibrates gradient matrices by normalizing each element by the sum 
    of its row and column norms. It interpolates between structural preconditioning 
    for large matrices and global energy normalization for vectors/small layers.
    
    Args:
        dim_threshold: Matrices with both dimensions >= this threshold undergo 
            structural equilibration. Otherwise, they get global energy normalization.
        epsilon: Small stability constant to prevent division by zero.
        
    Returns:
        An `optax.GradientTransformation` object.
    """
    
    def init_fn(params):
        # ANDI preconditioning is stateless (it only uses the current gradient)
        return jax.tree_util.tree_map(lambda _: ANDIState(), params)
        
    def update_fn(updates, state, params=None):
        del params # Unused
        
        def _andi_process_tensor(g: jnp.ndarray) -> jnp.ndarray:
            if not isinstance(g, jnp.ndarray):
                return g
                
            # Base variables
            nu = jnp.linalg.norm(g)
            tau = jnp.sqrt(nu**2 + 1.0)
            
            # Global normalization for scalars or empty arrays
            if g.ndim == 0 or g.size == 0:
                return g * (tau / (nu + epsilon))
                
            # Matricization for higher-order tensors (e.g., Conv2D weights: Cout, Cin, H, W)
            # Flatten all dimensions after the first one to form a matrix
            orig_shape = g.shape
            
            if g.ndim > 1:
                # Reshape to (d_1, d_2 * d_3 * ... * d_k)
                # IMPORTANT: Use Python math.prod (not jnp.prod) so the shape is a
                # static integer known at JIT-compile time, avoiding XLA reshape errors.
                d1 = orig_shape[0]
                d_rest = math.prod(orig_shape[1:])  # static int
                g_mat = g.reshape(d1, d_rest)
                
                m, n = g_mat.shape
                
                if m >= dim_threshold and n >= dim_threshold:
                    # Structural Equilibration (Rank-1 Additive Normalization)
                    # r: Row norms (m,)
                    # c: Column norms (n,)
                    r = jnp.linalg.norm(g_mat, axis=1, keepdims=True) + epsilon
                    c = jnp.linalg.norm(g_mat, axis=0, keepdims=True) + epsilon
                    
                    # Broadcasting Addition: D_ij = r_i + c_j
                    D = r + c
                    
                    # Element-wise equilibration
                    g_hat = g_mat / D
                    
                    # Rescale to target energy and restore shape
                    alpha = tau / (jnp.linalg.norm(g_hat) + epsilon)
                    g_final = (alpha * g_hat).reshape(orig_shape)
                    
                    return g_final
            
            # Global Normalization for vectors or matrices below threshold
            return g * (tau / (nu + epsilon))
            
        new_updates = jax.tree_util.tree_map(_andi_process_tensor, updates)
        return new_updates, state
        
    return optax.GradientTransformation(init_fn, update_fn)


def andi(
    learning_rate: optax.ScalarOrSchedule,
    b1: float = 0.9,
    weight_decay: float = 0.01,
    dim_threshold: int = 32,
    epsilon: float = 1e-8,
    mask: Optional[Any] = None,
) -> optax.GradientTransformation:
    """
    The full ANDI Optimizer.
    
    A principled implementation of the Adaptive Norm-Distribution Interface,
    incorporating decoupled weight decay and Nesterov momentum.
    
    Args:
        learning_rate: The learning rate or learning rate schedule.
        b1: The exponential decay rate for the Nesterov momentum buffer (default: 0.9).
        weight_decay: Strength of the weight decay regularization.
        dim_threshold: Minimum dimension size to apply structural equilibration.
        epsilon: Small stability constant.
        mask: Optional mask for decoupled weight decay.
        
    Returns:
        The `optax.GradientTransformation` for the full ANDI optimizer.
    """
    
    # 1. Decoupled Weight Decay
    # We apply this BEFORE the ANDI update as specified in the algorithm (Algorithm 1, Line 5-6)
    # Note: Optax applies all transformations to the gradient. 
    # Optax's add_decayed_weights handles Decoupled Weight Decay efficiently.
    tx_list = [optax.add_decayed_weights(weight_decay, mask)]
        
    # 2. ANDI Preconditioning
    tx_list.append(scale_by_andi(dim_threshold=dim_threshold, epsilon=epsilon))
    
    # 3. Nesterov Momentum
    tx_list.append(optax.trace(decay=b1, nesterov=True))
    
    # 4. Global Learning Rate Scaling
    tx_list.append(optax.scale_by_learning_rate(learning_rate))
    
    return optax.chain(*tx_list)
