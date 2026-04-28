import os
import jax
import jax.numpy as jnp
from sife.model import SIFELDMConfig
from sife.tokenizer import Vocabulary, SIFETokenizer
from train import create_train_state, train_step_vision_jit, train_step_text_jit
from inference import generate_vision, generate_text

def main():
    print("Initializing Config...")
    config = SIFELDMConfig(
        vocab_size=100,
        max_seq_len=16,
        embed_dim=128,
        num_layers=2,
        num_heads=4,
        mlp_dim=256,
        patch_size=8,
        image_size=32,  # Small image for fast demo
        channels=3
    )

    print("Setting up Vocabulary...")
    vocab = Vocabulary()
    for char in "abcdefghijklmnopqrstuvwxyz ":
        vocab.add_token(char)
        
    tokenizer = SIFETokenizer(vocab, max_len=config.max_seq_len)

    print("Creating Train State...")
    rng = jax.random.PRNGKey(42)
    rng, init_rng = jax.random.split(rng)
    state = create_train_state(config, init_rng)

    print("Generating Dummy Data...")
    batch_size = 2
    # Vision Data
    dummy_images = jax.random.uniform(rng, (batch_size, config.image_size, config.image_size, config.channels))
    vision_batch = {'image': dummy_images}
    
    # Text Data
    texts = ["hello world", "sife model"]
    tokenized = tokenizer(texts)
    text_batch = {'tokens': tokenized['input_ids'], 'mask': tokenized['mask']}

    print("Running a few Training Steps...")
    for step in range(3):
        rng, v_rng, t_rng = jax.random.split(rng, 3)
        state, v_loss = train_step_vision_jit(state, vision_batch, v_rng, config=config)
        state, t_loss = train_step_text_jit(state, text_batch, t_rng, config=config)
        print(f"Step {step+1}: Vision Loss: {v_loss:.4f}, Text Loss: {t_loss:.4f}")

    print("\n--- Generating First Image ---")
    rng, gen_rng = jax.random.split(rng)
    # Apply_fn.__self__ refers to the model object in TrainState 
    # but we can just instantiate it to be safe if that throws an error, but let's try it
    
    # We must instantiate model because apply_fn is just a function
    # Wait, state.apply_fn.__self__ might not exist if it's not a bound method in JAX. 
    # Let's instantiate a model manually
    from sife.model import SIFELDM
    model = SIFELDM(config)
    
    gen_images = generate_vision(model, state.params, config, gen_rng, batch_size=1)
    print("Generated Image Shape:", gen_images.shape)
    print("Image Data Range: Min {:.4f}, Max {:.4f}".format(float(gen_images.min()), float(gen_images.max())))

    print("\n--- Generating First Text ---")
    rng, gen_rng = jax.random.split(rng)
    start_text = "he"
    start_ids = tokenizer([start_text], padding=False)['input_ids'][0]
    
    gen_tokens = generate_text(model, state.params, config, gen_rng, start_tokens=start_ids[None, :], max_new_tokens=5)
    print("Generated Tokens:", gen_tokens)
    print("Decoded Text:", vocab.decode(gen_tokens[0].tolist()))
    
if __name__ == "__main__":
    main()
