import re
import os

path = r'c:\Users\Dream\Downloads\sife-ldm-python (1)\home\z\my-project\download\sife-ldm\sife\unet.py'

with open(path, 'r') as f:
    content = f.read()

# 1. Update ComplexUpBlock signature
content = re.sub(
    r'class ComplexUpBlock\(nn\.Module\):.*?def __call__\(\s+self,\s+x: Array,\s+skip: Array,\s+t_emb: Array,\s+context: Optional\[Array\] = None,\s+abs_phase: Optional\[Array\] = None,',
    r'class ComplexUpBlock(nn.Module):\n    """\n    Upsampling block for the U-Net decoder path.\n    \n    Upsamples the sequence and concatenates with skip connection,\n    then applies residual blocks.\n    """\n    features: int\n    num_layers: int = 2\n    kernel_size: int = 3\n    attention: bool = False\n    dropout_rate: float = 0.1\n    num_experts: int = 0\n    \n    @nn.compact\n    def __call__(\n        self,\n        x: Array,\n        skip: Array,\n        t_emb: Array,\n        context: Optional[Array] = None,\n        abs_phase: Optional[Array] = None,\n        action_emb: Optional[Array] = None,',
    content,
    flags=re.DOTALL
)

# 2. Update ComplexResidualBlock calls in ComplexUpBlock
content = re.sub(
    r'x = ComplexResidualBlock\(\s+features=self\.features,\s+kernel_size=self\.kernel_size,\s+dropout_rate=self\.dropout_rate\s+\)\(x, t_emb, abs_phase, deterministic\)',
    r'x = ComplexResidualBlock(\n                features=self.features,\n                kernel_size=self.kernel_size,\n                dropout_rate=self.dropout_rate\n            )(x, t_emb, abs_phase, action_emb, deterministic)',
    content
)

# 3. Update ComplexUNet1D signature and calls
# (Assuming context for ComplexUNet1D)
content = re.sub(
    r'def __call__\(\s+self,\s+x: Array,\s+t: Array,\s+context: Optional\[Array\] = None,\s+abs_phase: Optional\[Array\] = None,',
    r'def __call__(\n        self,\n        x: Array,\n        t: Array,\n        context: Optional[Array] = None,\n        abs_phase: Optional[Array] = None,\n        action: Optional[Array] = None,',
    content
)

# 4. Inject action_emb in ComplexUNet1D
content = re.sub(
    r't_emb = ComplexTimeEmbedding\(self\.features\)\(t\)',
    r't_emb = ComplexTimeEmbedding(self.features)(t)\n        \n        # Action embedding\n        action_emb = None\n        if action is not None:\n            action_emb = ComplexTimeEmbedding(dim=self.features)(action)',
    content
)

# 5. Update DownBlock calls in ComplexUNet1D
content = re.sub(
    r'\)\(h, t_emb, context, abs_phase, mask, deterministic\)',
    r')(h, t_emb, context, abs_phase, action_emb, mask, deterministic)',
    content
)

# 6. Update UpBlock calls in ComplexUNet1D
content = re.sub(
    r'\)\(h, skips\[i\], t_emb, context, abs_phase, mask, deterministic\)',
    r')(h, skips[i], t_emb, context, abs_phase, action_emb, mask, deterministic)',
    content
)

with open(path, 'w') as f:
    f.write(content)

print("Patching complete.")
