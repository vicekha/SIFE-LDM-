import os, sys, jax, jax.numpy as jnp, numpy as np
os.environ["PYTHONPATH"] = "."
from sife.model import SIFELDMConfig, create_train_state
from sife.field import SIFEConfig
from sife.diffusion import DiffusionConfig
from sife.multiscale import create_multiscale_config

# Config matched to A100 test
sife_config = SIFEConfig()
diff_config = DiffusionConfig(num_timesteps=1000)
ms_config = create_multiscale_config(num_levels=3, base_features=32)
config = SIFELDMConfig(
    sife=sife_config, diffusion=diff_config, multiscale=ms_config,
    is_image=True, image_size=(32, 32),
    embed_dim=128, batch_size=4,
    learning_rate=2e-4, num_classes=100
)

key = jax.random.PRNGKey(42)
model, state, diffusion = create_train_state(config, key)

# Create dummy batch
batch_images = jax.random.uniform(key, (4, 32, 32, 3))
batch_labels = jnp.array([1, 2, 3, 4], dtype=jnp.int32)
batch = {
    'images': batch_images,
    'labels': batch_labels,
    'use_context_mask': jnp.array([True, True, True, True])
}

key, subkey = jax.random.split(key)
t = jax.random.randint(subkey, (4,), 0, 1000)

key, subkey = jax.random.split(key)
noise = jax.random.normal(subkey, (4, 32, 32, 128), dtype=jnp.float32)
noise = noise + 1j * jax.random.normal(jax.random.split(subkey)[0], (4, 32, 32, 128), dtype=jnp.float32)

# Predict noise and loss directly using get_loss
def loss_fn(p):
    return model.get_loss(p, batch, subkey, diffusion)

import json

images_encoded = model.apply(state.params, batch['images'], method=model.encode_images)
x_t = diffusion.q_sample(images_encoded, t, subkey, noise)

loss_val, grads = jax.value_and_grad(loss_fn)(state.params)

grad_norms = jax.tree_util.tree_map(lambda x: jnp.linalg.norm(x) if x is not None else 0.0, grads)
leaves = jax.tree_util.tree_leaves(grad_norms)
leaf_arr = jnp.array([l for l in leaves if l > 0.0])

first_leaf_p = jax.tree_util.tree_leaves(state.params)[0]
first_leaf_g = jax.tree_util.tree_leaves(grads)[0]

results = {
    "Loss computed via get_loss": float(loss_val),
    "Images Encoded Var": float(jnp.var(images_encoded)),
    "x_t Var": float(jnp.var(x_t)),
    "Noise Var": float(jnp.var(noise)),
    "Max grad norm": float(jnp.max(leaf_arr)) if len(leaf_arr) > 0 else 0.0,
    "Min grad norm": float(jnp.min(leaf_arr)) if len(leaf_arr) > 0 else 0.0,
    "Mean grad norm": float(jnp.mean(leaf_arr)) if len(leaf_arr) > 0 else 0.0,
    "Param dtype": str(first_leaf_p.dtype) if hasattr(first_leaf_p, 'dtype') else 'unknown',
    "Grad dtype": str(getattr(first_leaf_g, 'dtype', type(first_leaf_g)))
}

with open("diagnostic_results.json", "w") as f:
    json.dump(results, f, indent=2)
print("Done writing to diagnostic_results.json")


