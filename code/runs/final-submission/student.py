"""Student model v3: RoPE + RMSNorm + SwiGLU + GPT-2 scaled residual init.

Changes vs. the v2 RoPE-only run (val BPB 1.780 at 2400 steps):
  - LayerNorm -> RMSNorm (no mean-centering, no bias; LLaMA-style).
  - GELU MLP -> SwiGLU gated MLP (silu(gate) * up, then down-proj).
  - Residual projections (attn proj, mlp down) initialized with std/sqrt(2*depth).
  - Depth 4 -> 6 (set via config; width/heads/vocab/context unchanged).

RoPE half-split convention and weight tying (head.weight == token.weight)
are kept from v2. No learned position embedding.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F


def rope_cos_sin(seq_len, head_dim, base=10000.0):
    """Precompute cos/sin tables for RoPE, GPT-NeoX (half-split) convention."""
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
    t = torch.arange(seq_len, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)                 # [T, head_dim/2]
    emb = torch.cat([freqs, freqs], dim=-1)          # [T, head_dim]
    return emb.cos(), emb.sin()


def apply_rope(x, cos, sin):
    """Rotate the last dim of x [B, H, T, D] using precomputed cos/sin [T, D]."""
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    rotated = torch.cat([-x2, x1], dim=-1)
    cos = cos[None, None, :, :].to(x.dtype)
    sin = sin[None, None, :, :].to(x.dtype)
    return x * cos + rotated * sin


class RMSNorm(nn.Module):
    """Root-mean-square layer norm, computed in fp32 for stability."""

    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        norm = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * norm).type_as(x) * self.weight


class SwiGLU(nn.Module):
    """Gated MLP: down(silu(gate(x)) * up(x))."""

    def __init__(self, width, hidden):
        super().__init__()
        self.gate = nn.Linear(width, hidden, bias=False)
        self.up = nn.Linear(width, hidden, bias=False)
        self.down = nn.Linear(hidden, width, bias=False)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


class Block(nn.Module):
    def __init__(self, width=128, heads=4, dropout=0.1):
        super().__init__()
        self.heads = heads
        self.norm1 = RMSNorm(width)
        self.norm2 = RMSNorm(width)
        self.qkv = nn.Linear(width, 3 * width, bias=False)
        self.proj = nn.Linear(width, width, bias=False)
        # SwiGLU hidden ~= 8/3 * width, rounded to a multiple of 64.
        hidden = int(8 * width / 3)
        hidden = ((hidden + 63) // 64) * 64
        self.mlp = SwiGLU(width, hidden)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, cos, sin):
        batch, length, width = x.shape
        q, k, v = self.qkv(self.norm1(x)).view(
            batch, length, 3, self.heads, width // self.heads).permute(2, 0, 3, 1, 4)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)
        attended = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.drop(self.proj(attended.transpose(1, 2).reshape(batch, length, width)))
        return x + self.drop(self.mlp(self.norm2(x)))


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = dict(config)
        self.context = config['context']
        width = config['width']
        self.heads = config['heads']
        self.head_dim = width // config['heads']
        self.token = nn.Embedding(config['vocab'], width)
        dropout = config.get('dropout', 0.1)
        if not 0 <= dropout < 1:
            raise ValueError('dropout must be in [0, 1).')
        self.blocks = nn.ModuleList([Block(width, config['heads'], dropout) for _ in range(config['depth'])])
        self.norm = RMSNorm(width)
        self.head = nn.Linear(width, config['vocab'], bias=False)
        self.apply(self._init)
        # GPT-2 scaled init: shrink the residual-stream projections so deep
        # layers start near-identity and the activations don't blow up.
        for name, p in self.named_parameters():
            if name.endswith('proj.weight') or name.endswith('mlp.down.weight'):
                nn.init.normal_(p, std=0.02 / math.sqrt(2 * config['depth']))
        self.head.weight = self.token.weight
        cos, sin = rope_cos_sin(self.context, self.head_dim)
        self.register_buffer('rope_cos', cos)
        self.register_buffer('rope_sin', sin)

    @staticmethod
    def _init(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, std=0.02)
            if getattr(module, 'bias', None) is not None:
                nn.init.zeros_(module.bias)

    def features(self, ids):
        x = self.token(ids)
        length = ids.shape[1]
        cos, sin = self.rope_cos[:length], self.rope_sin[:length]
        for block in self.blocks:
            x = block(x, cos, sin)
        return self.norm(x)

    def forward(self, ids):
        """Training interface: unnormalized next-token logits [batch, time, vocab]."""
        return self.head(self.features(ids))

    def predict_log_probs(self, ids):
        """Evaluation interface: normalized log probabilities, strictly causal."""
        return F.log_softmax(self(ids).float(), dim=-1)


def build_model(config):
    return GPT(config)
