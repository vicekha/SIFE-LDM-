import jax
import jax.numpy as jnp
import optax
import flax.linen as nn
from datasets import load_dataset
from tokenizers import Tokenizer, models, trainers, pre_tokenizers
from sife.model import SIFELDM, SIFELDMConfig, contrastive_phase_loss
from sife.diffusion import GaussianDiffusion
import numpy as np
from PIL import Image

def train_phase_alignment():
    print("Loading COCO 2014 subset (10k images for POC)...")
    # Using a smaller split or subset for speed
    ds = load_dataset("ChristophSchuhmann/coco-2014", "coco_2014_captions", split="train", streaming=True).take(10000)
    
    print("Building BPE Tokenizer...")
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    trainer = trainers.BpeTrainer(special_tokens=["<pad>", "<unk>", "<bos>", "<eos>", "<mask>"], vocab_size=4000)
    
    # Collect 1000 captions for training tokenizer
    iter_ds = iter(ds)
    sample_captions = [next(iter_ds)["caption"] for _ in range(1000)]
    tokenizer.train_from_iterator(sample_captions, trainer=trainer)
    
    config = SIFELDMConfig(
        vocab_size=tokenizer.get_vocab_size(),
        max_seq_len=16,
        embed_dim=128,
        num_layers=4,
        num_heads=4,
        mlp_dim=256,
        patch_size=8,
        image_size=64
    )
    
    rng = jax.random.PRNGKey(42)
    model = SIFELDM(config)
    
    # Initialize
    dummy_img = jnp.ones((1, 64, 64, 3))
    dummy_t = jnp.zeros((1,), dtype=jnp.int32)
    dummy_tokens = jnp.zeros((1, 16), dtype=jnp.int32)
    dummy_mask = jnp.zeros((1, 16), dtype=jnp.bool_)
    
    rng, init_rng = jax.random.split(rng)
    # Re-using the logic from my debug script for initialization
    class MultiInitModel(nn.Module):
        model: SIFELDM
        @nn.compact
        def __call__(self, img, t, tokens, mask):
            x0_img = self.model.image_encoder(img)
            x0 = self.model.patch_encoder(x0_img)
            self.model(x0, t, mode='vision')
            self.model.text_encoder(tokens, mask)
            return None

    init_model = MultiInitModel(model=model)
    variables = init_model.init(init_rng, dummy_img, dummy_t, dummy_tokens, dummy_mask)
    params = variables['params']['model']
    
    optimizer = optax.adam(5e-5)
    opt_state = optimizer.init(params)
    
    @jax.jit
    def train_step(params, opt_state, images, tokens, masks):
        def loss_fn(p):
            # 1. Encode Image to Global Phase
            x0_img_raw = model.apply({'params': p}, images, method=model.encode_image)
            x0_img = model.apply({'params': p}, x0_img_raw, method=model.encode_patch)
            
            # 2. Encode Text to Global Phase
            x0_txt = model.apply({'params': p}, tokens, mask=masks, method=model.encode_text)
            
            # 3. Contrastive Alignment Loss (Phase 1)
            loss = contrastive_phase_loss(x0_txt, x0_img)
            return loss
            
        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

    print("Starting Phase 1 Alignment...")
    iter_ds = iter(ds)
    for step in range(500):
        batch_imgs = []
        batch_tokens = []
        batch_masks = []
        
        # Simple batching from stream
        for _ in range(8): # Small batch size for Colab CPU/GPU
            try:
                sample = next(iter_ds)
                # Process image
                img = sample['image'].convert("RGB").resize((64, 64))
                img_arr = np.array(img) / 255.0
                batch_imgs.append(img_arr)
                
                # Process text
                enc = tokenizer.encode(sample['caption'])
                tokens = enc.ids[:16]
                if len(tokens) < 16:
                    tokens = tokens + [0] * (16 - len(tokens))
                mask = [False] * len(enc.ids[:16]) + [True] * (16 - len(enc.ids[:16]))
                batch_tokens.append(tokens)
                batch_masks.append(mask)
            except StopIteration:
                break
                
        if not batch_imgs: break
        
        imgs = jnp.array(batch_imgs)
        tokens = jnp.array(batch_tokens)
        masks = jnp.array(batch_masks)
        
        params, opt_state, loss = train_step(params, opt_state, imgs, tokens, masks)
        
        if step % 50 == 0:
            print(f"Step {step} | Alignment Loss: {loss:.4f}")

    print("Phase 1 Alignment complete. Semantic Phase Manifold initialized.")

if __name__ == "__main__":
    train_phase_alignment()
