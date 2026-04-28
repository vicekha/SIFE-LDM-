import jax
import jax.numpy as jnp
import optax
from flax.training import train_state
from sife.model import SIFELDM, SIFELDMConfig, get_loss
from sife.optim import andi
from sife.diffusion import GaussianDiffusion, MaskedDiffusion
import flax.linen as nn

class TrainState(train_state.TrainState):
    config: SIFELDMConfig

def create_train_state(config, rng):
    model = SIFELDM(config)
    
    class MultiInitModel(nn.Module):
        model: SIFELDM
        @nn.compact
        def __call__(self, img, t, tokens, mask):
            # Init Vision Enc/Dec
            x0_img = self.model.image_encoder(img)
            x0 = self.model.patch_encoder(x0_img)
            v = self.model(x0, t, mode='vision')
            hw = (img.shape[1], img.shape[2])
            img_out = self.model.image_decoder(v, hw)
            
            # Init Text Enc/Dec
            x_t = self.model.text_encoder(tokens, mask)
            t_out = self.model(x_t, t, mode='text')
            logits = self.model.symbol_decoder(t_out)
            return img_out, logits

    dummy_img = jnp.ones((1, config.image_size, config.image_size, config.channels))
    dummy_t = jnp.zeros((1,), dtype=jnp.int32)
    dummy_tokens = jnp.zeros((1, config.max_seq_len), dtype=jnp.int32)
    dummy_mask = jnp.zeros((1, config.max_seq_len), dtype=jnp.bool_)
    
    rng, init_rng = jax.random.split(rng)
    
    init_model = MultiInitModel(model=model)
    variables = init_model.init(init_rng, dummy_img, dummy_t, dummy_tokens, dummy_mask)
    params = variables['params']['model']
    
    tx = andi(learning_rate=1e-4, weight_decay=1e-4)
    
    return TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=tx,
        config=config
    )

def train_step_vision(state, batch, rng, config):
    def loss_fn(params):
        model = SIFELDM(config)
        diffusion = GaussianDiffusion()
        return get_loss(model, params, batch, rng, diffusion, config, mode='vision')
    
    loss, grads = jax.value_and_grad(loss_fn)(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss

def train_step_text(state, batch, rng, config):
    def loss_fn(params):
        model = SIFELDM(config)
        masked_diffusion = MaskedDiffusion()
        return get_loss(model, params, batch, rng, masked_diffusion, config, mode='text')
    
    loss, grads = jax.value_and_grad(loss_fn)(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss

train_step_vision_jit = jax.jit(train_step_vision, static_argnames=['config'])
train_step_text_jit = jax.jit(train_step_text, static_argnames=['config'])

def train_loop(state, vision_dataloader, text_dataloader, steps: int):
    rng = jax.random.PRNGKey(42)
    
    v_iter = iter(vision_dataloader)
    t_iter = iter(text_dataloader)
    
    for step in range(steps):
        rng, v_rng, t_rng = jax.random.split(rng, 3)
        
        try:
            v_batch = next(v_iter)
        except StopIteration:
            v_iter = iter(vision_dataloader)
            v_batch = next(v_iter)
            
        try:
            t_batch = next(t_iter)
        except StopIteration:
            t_iter = iter(text_dataloader)
            t_batch = next(t_iter)
            
        state, v_loss = train_step_vision_jit(state, v_batch, v_rng, config=state.config)
        state, t_loss = train_step_text_jit(state, t_batch, t_rng, config=state.config)
        
        if step % 100 == 0:
            print(f"Step {step} | Vision Loss: {v_loss:.4f} | Text Loss: {t_loss:.4f}")
            
    return state
