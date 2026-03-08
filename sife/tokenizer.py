"""
SIFE-LDM: Tokenization and Data Pipeline
=========================================

Implements tokenization for NLP and coding tasks, with complex field
embedding that preserves the semantic structure of the SIFE framework.

The key innovation is embedding tokens into a complex field where:
- Amplitude represents token importance/salience
- Phase represents semantic relationships

Author: SIFE-LDM Research Team
License: MIT
"""

import jax
import jax.numpy as jnp
from jax import jit, vmap
from typing import Dict, List, Optional, Tuple, Any, Sequence, Union, Callable
from functools import partial
import re
import json
import math
from collections import Counter

# Type aliases
Array = jnp.ndarray
PRNGKey = jnp.ndarray


class Vocabulary:
    """
    Vocabulary for NLP and coding tasks.
    
    Supports:
    - Special tokens (PAD, UNK, BOS, EOS)
    - Subword tokenization via BPE
    - Special handling for code tokens
    """
    
    PAD_TOKEN = '<pad>'
    UNK_TOKEN = '<unk>'
    BOS_TOKEN = '<bos>'
    EOS_TOKEN = '<eos>'
    MASK_TOKEN = '<mask>'
    NEWLINE_TOKEN = '<newline>'
    INDENT_TOKEN = '<indent>'
    DEDENT_TOKEN = '<dedent>'
    
    def __init__(
        self,
        min_freq: int = 1,
        max_size: Optional[int] = None,
        special_tokens: Optional[List[str]] = None
    ):
        self.min_freq = min_freq
        self.max_size = max_size
        self.special_tokens = special_tokens or [
            self.PAD_TOKEN,
            self.UNK_TOKEN,
            self.BOS_TOKEN,
            self.EOS_TOKEN,
            self.MASK_TOKEN,
            self.NEWLINE_TOKEN,
            self.INDENT_TOKEN,
            self.DEDENT_TOKEN
        ]
        
        self.token_to_id: Dict[str, int] = {}
        self.id_to_token: Dict[int, str] = {}
        self.token_freq: Counter = Counter()
        
        # Initialize special tokens
        for i, token in enumerate(self.special_tokens):
            self.token_to_id[token] = i
            self.id_to_token[i] = token
        
        self.pad_id = self.token_to_id[self.PAD_TOKEN]
        self.unk_id = self.token_to_id[self.UNK_TOKEN]
        self.bos_id = self.token_to_id[self.BOS_TOKEN]
        self.eos_id = self.token_to_id[self.EOS_TOKEN]
        self.mask_id = self.token_to_id[self.MASK_TOKEN]
        
        self._frozen = False
    
    def add_token(self, token: str) -> int:
        """Add a token to the vocabulary."""
        if self._frozen:
            raise RuntimeError("Vocabulary is frozen")
        
        if token not in self.token_to_id:
            token_id = len(self.token_to_id)
            self.token_to_id[token] = token_id
            self.id_to_token[token_id] = token
        
        self.token_freq[token] += 1
        return self.token_to_id[token]
    
    def build_from_texts(
        self,
        texts: List[str],
        tokenizer: Optional[Callable] = None
    ) -> None:
        """
        Build vocabulary from a list of texts.
        
        Args:
            texts: List of text strings
            tokenizer: Optional tokenization function
        """
        if tokenizer is None:
            tokenizer = self._default_tokenize
        
        # Count tokens
        token_counts = Counter()
        for text in texts:
            tokens = tokenizer(text)
            token_counts.update(tokens)
        
        # Add tokens meeting frequency threshold
        for token, count in token_counts.most_common():
            if count >= self.min_freq:
                if self.max_size is None or len(self.token_to_id) < self.max_size:
                    self.add_token(token)
        
        self._frozen = True
    
    def build_from_code(
        self,
        code_samples: List[str],
        language: str = 'python'
    ) -> None:
        """
        Build vocabulary optimized for code.
        
        Args:
            code_samples: List of code strings
            language: Programming language
        """
        tokenizer = self._get_code_tokenizer(language)
        self.build_from_texts(code_samples, tokenizer)
    
    def _default_tokenize(self, text: str) -> List[str]:
        """Default tokenization: whitespace + punctuation."""
        # Split on whitespace
        tokens = []
        for word in text.split():
            # Split on punctuation
            parts = re.findall(r'\w+|[^\w\s]', word)
            tokens.extend(parts)
        return tokens
    
    def _get_code_tokenizer(self, language: str) -> Callable:
        """Get language-specific code tokenizer."""
        if language == 'python':
            return self._tokenize_python
        elif language in ['javascript', 'js']:
            return self._tokenize_javascript
        else:
            return self._default_tokenize
    
    def _tokenize_python(self, code: str) -> List[str]:
        """Tokenize Python code."""
        import tokenize
        import io
        
        tokens = []
        try:
            for tok in tokenize.generate_tokens(io.StringIO(code).readline):
                token_type = tokenize.tok_name[tok.type]
                token_value = tok.string
                
                # Handle special cases
                if token_type == 'NL':
                    tokens.append(self.NEWLINE_TOKEN)
                elif token_type == 'INDENT':
                    tokens.append(self.INDENT_TOKEN)
                elif token_type == 'DEDENT':
                    tokens.append(self.DEDENT_TOKEN)
                elif token_type == 'STRING':
                    # Keep string tokens as is but mark as string
                    tokens.append(f'STRING:{token_value}')
                elif token_type == 'NUMBER':
                    tokens.append(f'NUMBER:{token_value}')
                else:
                    tokens.append(token_value)
        except:
            # Fallback to default tokenization
            tokens = self._default_tokenize(code)
        
        return tokens
    
    def _tokenize_javascript(self, code: str) -> List[str]:
        """Tokenize JavaScript code."""
        # Simple JavaScript tokenization
        # In practice, use a proper JS parser
        tokens = []
        
        # Patterns for JS tokens
        patterns = [
            (r'\/\/.*', 'COMMENT'),
            (r'\/\*[\s\S]*?\*\/', 'COMMENT'),
            (r'"(?:[^"\\]|\\.)*"', 'STRING'),
            (r"'(?:[^'\\]|\\.)*'", 'STRING'),
            (r'`(?:[^`\\]|\\.)*`', 'TEMPLATE'),
            (r'\b\d+\.?\d*\b', 'NUMBER'),
            (r'\b(const|let|var|function|return|if|else|for|while|class|import|export|from|async|await)\b', 'KEYWORD'),
            (r'[a-zA-Z_]\w*', 'IDENTIFIER'),
            (r'[{}()\[\];,.]', 'PUNCTUATION'),
            (r'[+\-*/%=<>!&|^~?:]', 'OPERATOR'),
        ]
        
        pos = 0
        while pos < len(code):
            if code[pos].isspace():
                if code[pos] == '\n':
                    tokens.append(self.NEWLINE_TOKEN)
                pos += 1
                continue
            
            matched = False
            for pattern, token_type in patterns:
                match = re.match(pattern, code[pos:])
                if match:
                    value = match.group()
                    if token_type in ['KEYWORD', 'PUNCTUATION', 'OPERATOR']:
                        tokens.append(value)
                    else:
                        tokens.append(f'{token_type}:{value}')
                    pos += len(value)
                    matched = True
                    break
            
            if not matched:
                tokens.append(code[pos])
                pos += 1
        
        return tokens
    
    def encode(self, text: str, tokenizer: Optional[Callable] = None) -> List[int]:
        """Encode text to token IDs."""
        if tokenizer is None:
            tokenizer = self._default_tokenize
        
        tokens = tokenizer(text)
        return [self.token_to_id.get(t, self.unk_id) for t in tokens]
    
    def decode(self, ids: List[int], skip_special: bool = True) -> str:
        """Decode token IDs to text."""
        tokens = []
        for id in ids:
            if id in self.id_to_token:
                token = self.id_to_token[id]
                if skip_special and token in self.special_tokens:
                    continue
                tokens.append(token)
            else:
                tokens.append(self.UNK_TOKEN)
        
        return ' '.join(tokens)
    
    def __len__(self) -> int:
        return len(self.token_to_id)
    
    def save(self, path: str) -> None:
        """Save vocabulary to file."""
        data = {
            'token_to_id': self.token_to_id,
            'special_tokens': self.special_tokens,
            'min_freq': self.min_freq,
            'max_size': self.max_size
        }
        with open(path, 'w') as f:
            json.dump(data, f)
    
    @classmethod
    def load(cls, path: str) -> 'Vocabulary':
        """Load vocabulary from file."""
        with open(path, 'r') as f:
            data = json.load(f)
        
        vocab = cls(
            min_freq=data.get('min_freq', 1),
            max_size=data.get('max_size'),
            special_tokens=data.get('special_tokens')
        )
        
        vocab.token_to_id = data['token_to_id']
        vocab.id_to_token = {int(v): k for k, v in vocab.token_to_id.items()}
        vocab._frozen = True
        
        return vocab


class ComplexFieldEmbedding:
    """
    Complex field embedding for tokens.
    
    Each token is embedded into the complex field where:
    - Amplitude (real magnitude): token salience/importance
    - Phase (angle): semantic position in meaning space
    
    The phase embedding uses a learned rotation that positions
    semantically similar tokens close in phase space.
    """
    
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        key: PRNGKey,
        amplitude_init_scale: float = 0.1,
        phase_init: str = 'learned'  # 'learned', 'random', 'polar'
    ):
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        
        key1, key2, key3 = jax.random.split(key, 3)
        
        # Amplitude embeddings (positive values)
        # Scaled to near-unit variance for diffusion SNR compatibility.
        self.amplitude_embeddings = jnp.abs(
            jax.random.normal(key1, (vocab_size, embed_dim))
        ) * 1.0  # Increased from 0.1 to 1.0 to match vision latent scale
        
        # Phase embeddings
        if phase_init == 'learned':
            # Learnable phase, initialized uniformly
            self.phase_embeddings = 2 * jnp.pi * jax.random.uniform(
                key2, (vocab_size, embed_dim)
            )
        elif phase_init == 'random':
            # Random phases
            self.phase_embeddings = 2 * jnp.pi * jax.random.uniform(
                key2, (vocab_size, embed_dim)
            )
        elif phase_init == 'polar':
            # Polar initialization: position determines angle
            positions = jnp.arange(vocab_size)
            self.phase_embeddings = jnp.outer(
                positions, 2 * jnp.pi * jnp.arange(embed_dim) / vocab_size
            )
        else:
            raise ValueError(f"Unknown phase_init: {phase_init}")
        
        # Phase adjustment matrix (for semantic rotation)
        self.phase_rotation = jax.random.normal(key3, (embed_dim, embed_dim)) * 0.1
    
    def __call__(self, token_ids: Array) -> Array:
        """
        Get complex embeddings for tokens.
        
        Args:
            token_ids: Token IDs of shape (batch, seq_len)
        
        Returns:
            Complex embeddings of shape (batch, seq_len, embed_dim)
        """
        # Get amplitude and phase
        amplitude = self.amplitude_embeddings[token_ids]
        phase = self.phase_embeddings[token_ids]
        
        # Apply phase rotation
        phase = phase @ self.phase_rotation
        phase = phase % (2 * jnp.pi)  # Keep in [0, 2π)
        
        # Construct complex embedding
        return amplitude * jnp.exp(1j * phase)
    
    def get_phase_similarity(self, token_id1: int, token_id2: int) -> float:
        """
        Compute phase similarity between two tokens.
        
        Returns a value in [0, 1] where 1 means identical phase.
        """
        phase1 = self.phase_embeddings[token_id1]
        phase2 = self.phase_embeddings[token_id2]
        
        # Cosine similarity in phase space
        cos_sim = jnp.mean(jnp.cos(phase1 - phase2))
        return (cos_sim + 1) / 2  # Normalize to [0, 1]


class PositionalEmbedding:
    """
    Positional embedding for sequences.
    
    Uses sinusoidal embeddings for amplitude and learned
    embeddings for phase.
    """
    
    def __init__(
        self,
        max_seq_len: int,
        embed_dim: int,
        key: PRNGKey
    ):
        self.max_seq_len = max_seq_len
        self.embed_dim = embed_dim
        
        # Sinusoidal amplitude embeddings
        position = jnp.arange(max_seq_len)[:, None]
        div_term = jnp.exp(
            jnp.arange(0, embed_dim, 2) * (-math.log(10000.0) / embed_dim)
        )
        
        self.amplitude_pos = jnp.zeros((max_seq_len, embed_dim))
        self.amplitude_pos = self.amplitude_pos.at[:, 0::2].set(jnp.sin(position * div_term))
        self.amplitude_pos = self.amplitude_pos.at[:, 1::2].set(jnp.cos(position * div_term))
        # Remove absolute and +0.1 to maintain zero-mean/unit-variance characteristics
        # In complex field, we want the magnitude to be stable.
        eps = 1e-6
        self.amplitude_pos = self.amplitude_pos / jnp.sqrt(jnp.mean(self.amplitude_pos**2) + eps)
        
        # Learned phase embeddings
        key1, key2 = jax.random.split(key)
        self.phase_pos = 2 * jnp.pi * jax.random.uniform(key1, (max_seq_len, embed_dim))
    
    def __call__(self, seq_len: int) -> Array:
        """
        Get positional embeddings.
        
        Args:
            seq_len: Sequence length
        
        Returns:
            Complex positional embeddings of shape (seq_len, embed_dim)
        """
        amp = self.amplitude_pos[:seq_len]
        phase = self.phase_pos[:seq_len]
        
        return amp * jnp.exp(1j * phase)


class SIFETokenizer:
    """
    Complete tokenizer for SIFE-LDM with complex field embedding.
    
    This combines:
    - Vocabulary management
    - Token encoding/decoding
    - Complex field embedding
    - Positional embedding
    """
    
    def __init__(
        self,
        vocab: Vocabulary,
        embed_dim: int = 256,
        max_seq_len: int = 2048,
        key: PRNGKey = None
    ):
        self.vocab = vocab
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len
        
        if key is None:
            key = jax.random.PRNGKey(0)
        
        key1, key2 = jax.random.split(key)
        
        self.token_embedding = ComplexFieldEmbedding(
            vocab_size=len(vocab),
            embed_dim=embed_dim,
            key=key1
        )
        
        self.positional_embedding = PositionalEmbedding(
            max_seq_len=max_seq_len,
            embed_dim=embed_dim,
            key=key2
        )
    
    def encode(
        self,
        text: str,
        add_special_tokens: bool = True,
        max_length: Optional[int] = None,
        padding: bool = True,
        truncation: bool = True
    ) -> Dict[str, Array]:
        """
        Tokenize and embed text.
        
        Args:
            text: Input text
            add_special_tokens: Add BOS/EOS tokens
            max_length: Maximum sequence length
            padding: Pad to max_length
            truncation: Truncate if longer than max_length
        
        Returns:
            Dictionary with 'input_ids', 'complex_embedding', 'attention_mask'
        """
        if max_length is None:
            max_length = self.max_seq_len
        
        # Encode text
        ids = self.vocab.encode(text)
        
        if add_special_tokens:
            ids = [self.vocab.bos_id] + ids + [self.vocab.eos_id]
        
        # Truncate
        if truncation and len(ids) > max_length:
            ids = ids[:max_length - 1] + [self.vocab.eos_id]
        
        seq_len = len(ids)
        
        # Pad
        attention_mask = [1] * seq_len
        if padding:
            pad_length = max_length - seq_len
            ids = ids + [self.vocab.pad_id] * pad_length
            attention_mask = attention_mask + [0] * pad_length
        
        # Convert to arrays
        input_ids = jnp.array(ids, dtype=jnp.int32)
        attention_mask = jnp.array(attention_mask, dtype=jnp.float32)
        
        # Get complex embeddings
        token_emb = self.token_embedding(input_ids[jnp.newaxis, :])[0]
        pos_emb = self.positional_embedding(max_length)
        
        # Combine token and positional embeddings
        # In complex space: combine via multiplication
        # (amplitude multiplies, phase adds)
        complex_embedding = token_emb * pos_emb
        
        return {
            'input_ids': input_ids,
            'complex_embedding': complex_embedding,
            'attention_mask': attention_mask
        }
    
    def decode(
        self,
        ids: Array,
        skip_special_tokens: bool = True
    ) -> str:
        """Decode token IDs to text."""
        ids_list = ids.tolist() if hasattr(ids, 'tolist') else list(ids)
        return self.vocab.decode(ids_list, skip_special_tokens)
    
    def batch_encode(
        self,
        texts: List[str],
        **kwargs
    ) -> Dict[str, Array]:
        """Encode a batch of texts."""
        results = [self.encode(text, **kwargs) for text in texts]
        
        return {
            'input_ids': jnp.stack([r['input_ids'] for r in results]),
            'complex_embedding': jnp.stack([r['complex_embedding'] for r in results]),
            'attention_mask': jnp.stack([r['attention_mask'] for r in results])
        }


class DataPipeline:
    """
    Data pipeline for training SIFE-LDM on NLP and coding tasks.
    
    Handles:
    - Loading and preprocessing text/code data
    - Batching and shuffling
    - On-the-fly tokenization and embedding
    """
    
    def __init__(
        self,
        tokenizer: SIFETokenizer,
        batch_size: int = 32,
        shuffle: bool = True,
        drop_last: bool = True
    ):
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
    
    def load_text_file(self, path: str) -> List[str]:
        """Load text from a file, one sample per line."""
        with open(path, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f if line.strip()]
    
    def load_code_files(
        self,
        directory: str,
        extensions: List[str] = None,
        max_file_size: int = 100000
    ) -> List[str]:
        """Load code files from a directory."""
        import os
        import glob
        
        if extensions is None:
            extensions = ['.py', '.js', '.java', '.cpp', '.c', '.h', '.ts', '.go', '.rs']
        
        code_samples = []
        for ext in extensions:
            for filepath in glob.glob(os.path.join(directory, f'**/*{ext}'), recursive=True):
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        code = f.read()
                        if len(code) < max_file_size:
                            code_samples.append(code)
                except Exception as e:
                    print(f"Error reading {filepath}: {e}")
        
        return code_samples
    
    def create_dataset(
        self,
        texts: List[str],
        key: PRNGKey = None
    ) -> 'SIFEDataset':
        """Create a dataset from a list of texts."""
        return SIFEDataset(
            texts=texts,
            tokenizer=self.tokenizer,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            drop_last=self.drop_last,
            key=key
        )
    
    def create_code_dataset(
        self,
        directory: str,
        extensions: List[str] = None,
        key: PRNGKey = None
    ) -> 'SIFEDataset':
        """Create a dataset from code files."""
        code_samples = self.load_code_files(directory, extensions)
        return self.create_dataset(code_samples, key)


class SIFEDataset:
    """
    Dataset class for SIFE-LDM training.
    
    Provides an iterator interface for training loops.
    """
    
    def __init__(
        self,
        texts: List[str],
        tokenizer: SIFETokenizer,
        batch_size: int = 32,
        shuffle: bool = True,
        drop_last: bool = True,
        key: PRNGKey = None
    ):
        self.texts = texts
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.key = key if key is not None else jax.random.PRNGKey(0)
        
        self.num_samples = len(texts)
        self.num_batches = self.num_samples // batch_size if drop_last else (self.num_samples + batch_size - 1) // batch_size
    
    def __len__(self) -> int:
        return self.num_batches
    
    def __iter__(self):
        """Iterate over batches."""
        # Shuffle indices
        indices = jnp.arange(self.num_samples)
        if self.shuffle:
            self.key, subkey = jax.random.split(self.key)
            indices = jax.random.permutation(subkey, indices)
        
        # Yield batches
        for i in range(0, self.num_samples - self.batch_size + 1, self.batch_size):
            batch_indices = indices[i:i + self.batch_size]
            batch_texts = [self.texts[idx] for idx in batch_indices]
            
            yield self.tokenizer.batch_encode(batch_texts)
    
    def get_batch(self, indices: List[int]) -> Dict[str, Array]:
        """Get a specific batch by indices."""
        batch_texts = [self.texts[i] for i in indices]
        return self.tokenizer.batch_encode(batch_texts)


def create_training_data(
    tokenizer: SIFETokenizer,
    text_path: Optional[str] = None,
    code_dir: Optional[str] = None,
    batch_size: int = 32,
    key: PRNGKey = None
) -> Tuple[SIFEDataset, SIFEDataset]:
    """
    Create training and validation datasets.
    
    Args:
        tokenizer: SIFE tokenizer
        text_path: Path to text file
        code_dir: Path to code directory
        batch_size: Batch size
        key: Random key
    
    Returns:
        Tuple of (train_dataset, val_dataset)
    """
    texts = []
    
    if text_path:
        with open(text_path, 'r', encoding='utf-8') as f:
            texts.extend([line.strip() for line in f if line.strip()])
    
    if code_dir:
        import os
        import glob
        for ext in ['.py', '.js', '.java', '.cpp', '.ts']:
            for filepath in glob.glob(os.path.join(code_dir, f'**/*{ext}'), recursive=True):
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        texts.append(f.read())
                except:
                    pass
    
    # Split into train/val
    if key is None:
        key = jax.random.PRNGKey(0)
    
    key, subkey = jax.random.split(key)
    indices = jax.random.permutation(subkey, len(texts))
    
    split_point = int(0.95 * len(texts))
    train_indices = indices[:split_point]
    val_indices = indices[split_point:]
    
    train_texts = [texts[i] for i in train_indices]
    val_texts = [texts[i] for i in val_indices]
    
    key, subkey = jax.random.split(key)
    train_dataset = SIFEDataset(train_texts, tokenizer, batch_size, key=subkey)
    val_dataset = SIFEDataset(val_texts, tokenizer, batch_size, shuffle=False, key=key)
    
    return train_dataset, val_dataset


# Utility functions for common NLP tasks

def compute_perplexity(
    model: Callable,
    dataset: SIFEDataset,
    diffusion: 'GaussianDiffusion',
    key: PRNGKey
) -> float:
    """
    Compute perplexity on a dataset.
    """
    total_loss = 0.0
    total_samples = 0
    
    for batch in dataset:
        batch_size = batch['input_ids'].shape[0]
        key, subkey = jax.random.split(key)
        
        # Sample timesteps
        t = jax.random.randint(
            subkey, (batch_size,), 0, diffusion.num_timesteps
        )
        
        # Compute loss
        loss = compute_loss(
            model,
            batch['complex_embedding'],
            t,
            subkey,
            diffusion
        )
        
        total_loss += loss * batch_size
        total_samples += batch_size
    
    avg_loss = total_loss / total_samples
    perplexity = jnp.exp(avg_loss)
    
    return float(perplexity)


def generate_text(
    model: Callable,
    tokenizer: SIFETokenizer,
    prompt: str,
    diffusion: 'GaussianDiffusion',
    key: PRNGKey,
    num_steps: int = 50,
    max_length: int = 256,
    temperature: float = 1.0
) -> str:
    """
    Generate text from a prompt.
    """
    from .diffusion import DDIMSampler
    
    # Encode prompt
    prompt_data = tokenizer.encode(prompt, max_length=max_length)
    prompt_emb = prompt_data['complex_embedding']
    
    # Initialize from prompt + noise
    key, subkey = jax.random.split(key)
    noise_shape = (max_length - prompt_emb.shape[0], tokenizer.embed_dim)
    noise = jax.random.normal(subkey, noise_shape, dtype=jnp.float32)
    noise = noise + 1j * jax.random.normal(jax.random.split(subkey)[0], noise_shape)
    
    # Generate
    ddim = DDIMSampler(diffusion)
    generated = ddim.sample(
        model,
        noise_shape,
        key,
        num_steps=num_steps
    )
    
    # Combine prompt and generated
    full_embedding = jnp.concatenate([prompt_emb, generated], axis=0)
    
    # Decode (simplified - in practice would need learned decoder)
    # For now, return prompt as placeholder
    return prompt
