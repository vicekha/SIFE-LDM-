import json
import jax.numpy as jnp
from typing import List, Optional, Union

class Vocabulary:
    def __init__(self, token_to_id: Optional[dict] = None):
        self.token_to_id = token_to_id or {"<PAD>": 0, "<UNK>": 1, "<BOS>": 2, "<EOS>": 3, "<MASK>": 4}
        self.id_to_token = {v: k for k, v in self.token_to_id.items()}
        self.pad_id = self.token_to_id["<PAD>"]
        self.unk_id = self.token_to_id["<UNK>"]
        self.bos_id = self.token_to_id["<BOS>"]
        self.eos_id = self.token_to_id["<EOS>"]
        self.mask_id = self.token_to_id["<MASK>"]
        
    def __len__(self):
        return len(self.token_to_id)
        
    def add_token(self, token: str):
        if token not in self.token_to_id:
            idx = len(self.token_to_id)
            self.token_to_id[token] = idx
            self.id_to_token[idx] = token
            
    def build_from_texts(self, texts: List[str]):
        for text in texts:
            # Word-level tokenization
            for word in text.split():
                self.add_token(word)
            
    def encode(self, text: str) -> List[int]:
        # Use space-based tokenization
        tokens = text.split()
        return [self.token_to_id.get(t, self.unk_id) for t in tokens]
        
    def decode(self, ids: List[int]) -> str:
        # Ignore special tokens during simple decode
        specials = {self.pad_id, self.bos_id, self.eos_id, self.mask_id}
        return " ".join([self.id_to_token.get(i, "<UNK>") for i in ids if i not in specials])
        
    def save(self, filepath: str):
        with open(filepath, 'w') as f:
            json.dump(self.token_to_id, f)
            
    @classmethod
    def load(cls, filepath: str):
        with open(filepath, 'r') as f:
            token_to_id = json.load(f)
        return cls(token_to_id)

class SIFETokenizer:
    def __init__(self, vocab: Vocabulary, max_len: int = 1024):
        self.vocab = vocab
        self.max_len = max_len
        
    def __call__(self, text: Union[str, List[str]], padding: bool = True) -> dict:
        if isinstance(text, str):
            text = [text]
            
        batch_ids = []
        batch_masks = []
        
        for t in text:
            ids = [self.vocab.bos_id] + self.vocab.encode(t) + [self.vocab.eos_id]
            if len(ids) > self.max_len:
                ids = ids[:self.max_len]
                ids[-1] = self.vocab.eos_id
                
            mask = [False] * len(ids) # False means unmasked / valid
            
            if padding:
                pad_len = self.max_len - len(ids)
                ids = ids + [self.vocab.pad_id] * pad_len
                mask = mask + [True] * pad_len # True means masked / padded
                
            batch_ids.append(ids)
            batch_masks.append(mask)
            
        return {
            "input_ids": jnp.array(batch_ids, dtype=jnp.int32),
            "mask": jnp.array(batch_masks, dtype=jnp.bool_)
        }
