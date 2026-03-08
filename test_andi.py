import os
import sys

# Ensure sife is importable
app_dir = os.path.dirname(os.path.abspath(__file__))
if app_dir not in sys.path:
    sys.path.insert(0, app_dir)

import jax
import jax.numpy as jnp
import optax
from sife.optim.andi import andi

print("Testing ANDI Optimizer with JAX Arrays")

# Create a random key
key = jax.random.PRNGKey(42)

# Create dummy parameters of various shapes
params = {
    'dense_layer': {
        'kernel': jax.random.normal(key, (64, 128)), # Matrix (should use structural ANDI)
        'bias': jax.random.normal(key, (128,)) # Vector (should use global ANDI)
    },
    'conv_layer': {
        'kernel': jax.random.normal(key, (32, 16, 3, 3)), # 4D Tensor (should matricize and use structural ANDI)
        'bias': jax.random.normal(key, (32,)) # Vector
    },
    'scalar': jnp.array(1.0) # Scalar
}

# Create dummy gradients (let's say they're same shape as params)
key, subkey = jax.random.split(key)
grads = jax.tree_util.tree_map(lambda p: jax.random.normal(subkey, jnp.shape(p)), params)

# Initialize optimizer
print("Initializing ANDI...")
optimizer = andi(learning_rate=0.01, b1=0.9, weight_decay=0.01, dim_threshold=32)
opt_state = optimizer.init(params)
print("State initialized successfully.")

# Apply updates
print("Applying updates...")
updates, new_opt_state = optimizer.update(grads, opt_state, params=params)
new_params = optax.apply_updates(params, updates)

# Check norms of updates
for layer, layer_updates in updates.items():
    if isinstance(layer_updates, dict):
        for param_name, update_val in layer_updates.items():
            print(f"Update norm for {layer}.{param_name} ({update_val.shape}): {jnp.linalg.norm(update_val):.4f}")
    else:
        print(f"Update norm for {layer} ({layer_updates.shape}): {jnp.linalg.norm(layer_updates):.4f}")

print("\nSUCCESS! ANDI Optimizer applied updates correctly.")
