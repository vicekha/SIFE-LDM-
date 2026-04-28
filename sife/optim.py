import optax
import jax
import jax.numpy as jnp

def scale_by_andi(tau: float = 1e-3, nu: float = 1e-4, epsilon: float = 1e-8) -> optax.GradientTransformation:
    def init_fn(params):
        return optax.EmptyState()

    def update_fn(updates, state, params=None):
        def precondition(g):
            # SAFETY: Force gradient to be real to prevent Optax ComplexWarning
            if jnp.iscomplexobj(g):
                g = g.real
                
            scale = jnp.minimum(tau / (nu + epsilon), 5.0)
            if g.ndim == 2:
                # Structural equilibration
                row_norm = jnp.linalg.norm(g, axis=1, keepdims=True) + epsilon
                col_norm = jnp.linalg.norm(g, axis=0, keepdims=True) + epsilon
                g = g / jnp.sqrt(row_norm * col_norm)
            return g * scale
            
        preconditioned_updates = jax.tree_util.tree_map(precondition, updates)
        return preconditioned_updates, state

    return optax.GradientTransformation(init_fn, update_fn)

def true_decoupled_weight_decay(weight_decay: float) -> optax.GradientTransformation:
    def init_fn(params):
        return optax.EmptyState()
    
    def update_fn(updates, state, params=None):
        if params is None:
            raise ValueError("Params must be provided to apply decoupled weight decay.")
        
        def apply_wd(g, p):
            return g + weight_decay * p
            
        updates = jax.tree_util.tree_map(apply_wd, updates, params)
        return updates, state
        
    return optax.GradientTransformation(init_fn, update_fn)

def andi(learning_rate: float, b1: float = 0.9, weight_decay: float = 1e-4) -> optax.GradientTransformation:
    return optax.chain(
        optax.clip_by_global_norm(1.0), # SAFETY: Hard cap on global norm to prevent NaN spikes
        scale_by_andi(),
        optax.trace(decay=b1, nesterov=True),
        true_decoupled_weight_decay(weight_decay),
        optax.scale(-learning_rate)  # Negative because optax adds the updates to params
    )
