"""Student model: the baseline GPT with switchable position encoding.

Diagnosed limitation
--------------------
The baseline learns one independent vector per window slot (nn.Embedding of
size context x width). Nothing in that design says slots 40 and 41 are
neighbours, so relative position -- what language mostly depends on -- has to
be learned indirectly from absolute labels.

Change
------
config["position"] selects how position is encoded:
  "learned" (default): identical to model.py -- learned absolute embeddings
                       added to the token embeddings.
  "rope":              rotary position embeddings. Queries and keys are rotated
                       by an angle proportional to their position before the
                       attention dot product, so attention scores depend only
                       on the distance between tokens. No learned position
                       parameters.

With "learned", modules are created in the same order as model.py, so the same
seed gives the same initial weights: this is the ablation control.

Further switches, each defaulting to the baseline behaviour:
  config["norm"]: "layernorm" (default) or "rmsnorm" -- RMSNorm rescales by the
                  root-mean-square only, with no mean subtraction and no bias.
  config["mlp"]:  "gelu" (default) or "swiglu" -- a gated feed-forward layer;
                  hidden size 344 keeps parameters and compute within 1% of GELU.
  config["dropout"]: 0.0 (default) or a probability such as 0.1 -- during
                  training only, randomly zeroes that fraction of each residual
                  branch's output (attention and feed-forward) so the model
                  cannot rely on any single feature. Scoring always runs with
                  dropout switched off (see predict_log_probs).
With every switch at its default, this file reproduces model.py exactly.
"""
import torch
from torch import nn
from torch.nn import functional as F


def rope_tables(context, head_dim, base=10000.0):
    """Cosine and sine tables of shape [context, head_dim // 2].

    Each pair of dimensions rotates at its own speed: pair i advances by
    base ** (-2i / head_dim) radians per position. Early pairs spin fast
    (short-range distances), later pairs spin slowly (long-range).
    """
    speeds = base ** (-torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
    angles = torch.arange(context, dtype=torch.float32)[:, None] * speeds[None, :]
    return angles.cos(), angles.sin()


def rotate(x, cos, sin):
    """Rotate each (first-half, second-half) pair of x by its position's angle.

    x: [batch, heads, time, head_dim]; cos, sin: [time, head_dim // 2].
    Uses only each position's own index, so causality is unaffected.
    """
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1)


def make_norm(kind, width):
    if kind == 'layernorm':
        return nn.LayerNorm(width)
    if kind == 'rmsnorm':
        return nn.RMSNorm(width, eps=1e-5)
    raise ValueError(f"config['norm'] must be 'layernorm' or 'rmsnorm', not {kind!r}")


class SwiGLU(nn.Module):
    """Gated feed-forward: down(silu(gate(x)) * up(x)).

    The gate branch passes through a smooth on/off curve (SiLU) and multiplies
    the up branch element by element, so each hidden feature is scaled by an
    input-dependent amount instead of a fixed curve.
    """
    def __init__(self, width, hidden):
        super().__init__()
        self.gate, self.up = nn.Linear(width, hidden), nn.Linear(width, hidden)
        self.down = nn.Linear(hidden, width)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


class Block(nn.Module):
    def __init__(self, width=128, heads=4, norm='layernorm', mlp='gelu', dropout=0.0):
        super().__init__()
        self.heads = heads
        self.dropout = dropout
        self.norm1, self.norm2 = make_norm(norm, width), make_norm(norm, width)
        self.qkv, self.proj = nn.Linear(width, 3 * width), nn.Linear(width, width)
        if mlp == 'gelu':
            self.mlp = nn.Sequential(nn.Linear(width, 4 * width), nn.GELU(), nn.Linear(4 * width, width))
        elif mlp == 'swiglu':
            # 2/3 of 4*width, rounded to a multiple of 8: 344 for width 128.
            self.mlp = SwiGLU(width, 8 * round(4 * width * 2 / 3 / 8))
        else:
            raise ValueError(f"config['mlp'] must be 'gelu' or 'swiglu', not {mlp!r}")

    def forward(self, x, rope=None):
        batch, length, width = x.shape
        q, k, v = self.qkv(self.norm1(x)).view(batch, length, 3, self.heads, width // self.heads).permute(2, 0, 3, 1, 4)
        if rope is not None:
            cos, sin = rope
            q, k = rotate(q, cos, sin), rotate(k, cos, sin)
        # Each position attends only to itself and earlier input tokens.
        attended = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.drop(self.proj(attended.transpose(1, 2).reshape(batch, length, width)))
        return x + self.drop(self.mlp(self.norm2(x)))

    def drop(self, branch):
        # Skipped entirely at p=0, so default runs stay bit-identical to before.
        return F.dropout(branch, self.dropout, self.training) if self.dropout > 0 else branch


class StudentGPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = dict(config)
        self.context = config['context']
        self.position = config.get('position', 'learned')
        if self.position not in ('learned', 'rope'):
            raise ValueError(f"config['position'] must be 'learned' or 'rope', not {self.position!r}")
        width, heads = config['width'], config['heads']
        if width % heads or (width // heads) % 2:
            raise ValueError('width / heads must be an even integer for rotary embeddings')
        self.token = nn.Embedding(config['vocab'], width)
        if self.position == 'learned':
            self.pos = nn.Embedding(self.context, width)
        else:
            cos, sin = rope_tables(self.context, width // heads, config.get('rope_base', 10000.0))
            # Fixed tables, recomputed on construction; not stored in checkpoints.
            self.register_buffer('rope_cos', cos, persistent=False)
            self.register_buffer('rope_sin', sin, persistent=False)
        norm, mlp = config.get('norm', 'layernorm'), config.get('mlp', 'gelu')
        dropout = float(config.get('dropout', 0.0))
        if not 0.0 <= dropout < 1.0:
            raise ValueError("config['dropout'] must be in [0, 1)")
        self.blocks = nn.ModuleList([Block(width, heads, norm, mlp, dropout) for _ in range(config['depth'])])
        self.norm = make_norm(norm, width)
        self.head = nn.Linear(width, config['vocab'], bias=False)
        self.apply(self.initialize)
        self.head.weight = self.token.weight

    @staticmethod
    def initialize(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, std=.02)
            if getattr(module, 'bias', None) is not None:
                nn.init.zeros_(module.bias)

    def features(self, ids):
        length = ids.shape[1]
        x = self.token(ids)
        rope = None
        if self.position == 'learned':
            x = x + self.pos(torch.arange(length, device=ids.device))
        else:
            rope = (self.rope_cos[:length], self.rope_sin[:length])
        for block in self.blocks:
            x = block(x, rope)
        return self.norm(x)

    def forward(self, ids):
        """Training interface: unnormalized next-token logits [batch, time, vocab]."""
        return self.head(self.features(ids))

    def predict_log_probs(self, ids):
        """Evaluation interface: normalized log probabilities. Stateless per call.

        Always scores with dropout off, whatever mode the caller left the model
        in, then restores that mode -- so mid-training validation and final
        scoring are deterministic.
        """
        was_training = self.training
        self.eval()
        try:
            return F.log_softmax(self(ids).float(), dim=-1)
        finally:
            self.train(was_training)


def build_model(config):
    return StudentGPT(config)
