"""Neural components shared by the baseline (T0) and the candidate backbone.

Design notes (all recorded in the run manifest):

* Pre-RMSNorm blocks, rotary positions over the full head dimension, SwiGLU
  feed-forward at ~8/3 width rounded to a multiple of 8, residual connections,
  no dropout, no learned absolute-position rows.
* The optional **memory** module is chunkwise linear attention (an associative
  fast-weight state): keys/values are written along the sequence and read by the
  query hidden state. It is exact as a linear-attention scan: intra-chunk via a
  masked QK product, cross-chunk via the carried state matrix. Bypass means the
  module is never constructed, so it has zero parameters and zero executed work.
* The optional **reasoning** module is one extra block applied `loops` times to
  the final hidden state with per-loop input injection. Bypass means the module
  is never constructed; `loops` work is not executed.
* A one-loop refinement is *not* an identity, so reasoning-off cells are
  architecturally distinct networks, not a `loops=1` setting.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, width: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        var = x.pow(2).mean(dim=-1, keepdim=True)
        return x * torch.rsqrt(var + self.eps) * self.weight


def rope_tables(seq_len: int, head_dim: int, base: float = 10_000.0, device=None, dtype=torch.float32):
    if head_dim % 2 != 0:
        raise ValueError("rotary dimension must be even")
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim))
    t = torch.arange(seq_len, device=device, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)
    cos = torch.cos(freqs).to(dtype)
    sin = torch.sin(freqs).to(dtype)
    return cos, sin


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # x: [B, H, L, Dh]
    x1, x2 = x[..., 0::2], x[..., 1::2]
    c = cos[None, None, :, :]
    s = sin[None, None, :, :]
    out = torch.empty_like(x)
    out[..., 0::2] = x1 * c - x2 * s
    out[..., 1::2] = x1 * s + x2 * c
    return out


class Attention(nn.Module):
    """Causal multi-head self-attention with rotary positions."""

    def __init__(self, width: int, n_heads: int) -> None:
        super().__init__()
        if width % n_heads != 0:
            raise ValueError(f"width {width} not divisible by heads {n_heads}")
        if (width // n_heads) % 2 != 0:
            raise ValueError(f"head dim {width // n_heads} must be even for rotary")
        self.width = width
        self.n_heads = n_heads
        self.head_dim = width // n_heads
        self.qkv = nn.Linear(width, 3 * width, bias=False)
        self.out = nn.Linear(width, width, bias=False)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, pad_mask: torch.Tensor | None):
        B, L, _ = x.shape
        q, k, v = self.qkv(x).split(self.width, dim=2)
        shape = (B, L, self.n_heads, self.head_dim)
        q = q.view(shape).transpose(1, 2)
        k = k.view(shape).transpose(1, 2)
        v = v.view(shape).transpose(1, 2)
        q = apply_rope(q, cos[:L], sin[:L])
        k = apply_rope(k, cos[:L], sin[:L])
        if pad_mask is not None and not bool(pad_mask.all()):
            attn_mask = torch.ones(L, L, dtype=torch.bool, device=x.device).tril()
            attn_mask = attn_mask[None, None] & pad_mask[:, None, None, :]
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        else:
            out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = out.transpose(1, 2).reshape(B, L, self.width)
        return self.out(out)


class SwiGLU(nn.Module):
    def __init__(self, width: int, hidden: int) -> None:
        super().__init__()
        self.gate = nn.Linear(width, hidden, bias=False)
        self.up = nn.Linear(width, hidden, bias=False)
        self.down = nn.Linear(hidden, width, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))


class LinearMemory(nn.Module):
    """Chunkwise linear-attention associative memory (fast-weight state).

    Written state: S in R^{H, Dh, Dh}. For a chunk of C tokens the read is
    ``o_t = q_t^T (S_prev + sum_{j<=t} k_j v_j^T)`` computed exactly as
    ``q_t^T S_prev + sum_{j<=t} (q_t . k_j) v_j``, so the module is a real
    causal recurrence but parallel inside a chunk.
    """

    def __init__(self, width: int, n_heads: int, chunk: int = 16) -> None:
        super().__init__()
        self.width = width
        self.n_heads = n_heads
        self.head_dim = width // n_heads
        self.chunk = chunk
        self.q_proj = nn.Linear(width, width, bias=False)
        self.k_proj = nn.Linear(width, width, bias=False)
        self.v_proj = nn.Linear(width, width, bias=False)
        self.out = nn.Linear(width, width, bias=False)

    def forward(self, x: torch.Tensor, pad_mask: torch.Tensor | None):
        B, L, _ = x.shape
        H, Dh, C = self.n_heads, self.head_dim, self.chunk
        q = self.q_proj(x).view(B, L, H, Dh).transpose(1, 2)
        k = self.k_proj(x).view(B, L, H, Dh).transpose(1, 2)
        v = self.v_proj(x).view(B, L, H, Dh).transpose(1, 2)
        if pad_mask is not None:
            keep = pad_mask[:, None, :, None].to(x.dtype)  # zero out padded writes
            k = k * keep
            v = v * keep
        state = x.new_zeros(B, H, Dh, Dh)
        key_mass = x.new_zeros(B, H, Dh)
        outs = []
        n_chunks = math.ceil(L / C)
        for ci in range(n_chunks):
            s, e = ci * C, min(L, (ci + 1) * C)
            qc, kc, vc = q[:, :, s:e], k[:, :, s:e], v[:, :, s:e]
            c = e - s
            valid = torch.ones(c, c, dtype=torch.bool, device=x.device).tril()
            if pad_mask is not None:
                valid = valid[None, None] & pad_mask[:, None, None, s:e].expand(B, H, c, c)
            w = torch.matmul(qc, kc.transpose(-1, -2)).masked_fill(~valid, 0.0)
            num = torch.matmul(w, vc) + torch.matmul(qc, state)
            den = (w.sum(dim=-1) + torch.einsum("bhcd,bhd->bhc", qc, key_mass)).abs().clamp_min(1e-6)
            outs.append(num / den.unsqueeze(-1))
            state = state + torch.matmul(vc.transpose(-1, -2), kc)
            key_mass = key_mass + kc.sum(dim=2)
        out = torch.cat(outs, dim=2).transpose(1, 2).reshape(B, L, self.width)
        return self.out(out)


class Block(nn.Module):
    def __init__(self, width: int, n_heads: int, use_memory: bool) -> None:
        super().__init__()
        self.norm1 = RMSNorm(width)
        self.attn = Attention(width, n_heads)
        self.use_memory = use_memory
        if use_memory:
            self.norm_mem = RMSNorm(width)
            self.memory = LinearMemory(width, n_heads)
            # residual scaling for the added branch: stabilises early training
            # without changing what the branch can represent
            self.mem_scale = nn.Parameter(torch.full((1,), 0.1))
        self.norm2 = RMSNorm(width)
        self.ffn = SwiGLU(width, ffn_hidden(width))

    def forward(self, x, cos, sin, pad_mask):
        x = x + self.attn(self.norm1(x), cos, sin, pad_mask)
        if self.use_memory:
            x = x + self.mem_scale * self.memory(self.norm_mem(x), pad_mask)
        x = x + self.ffn(self.norm2(x))
        return x


def ffn_hidden(width: int) -> int:
    """8/3 * width rounded to the next multiple of 8 (spec section 5)."""
    raw = 8.0 / 3.0 * width
    return int(math.ceil(raw / 8.0) * 8)


def n_heads_for(width: int, requested: int = 4) -> int:
    """Four heads where width permits, head dimension even."""
    heads = requested
    while heads > 1 and (width % heads != 0 or (width // heads) % 2 != 0):
        heads -= 1
    if width % heads != 0 or (width // heads) % 2 != 0:
        raise ValueError(f"width {width} does not admit an even head dimension")
    return heads


class SequenceModel(nn.Module):
    """Shared trunk: token embedding -> blocks -> [memory/reasoning] -> readout."""

    def __init__(
        self,
        *,
        vocab: int,
        n_classes: int,
        width: int,
        depth: int,
        n_heads: int = 4,
        use_memory: bool = False,
        use_reasoning: bool = False,
        reasoning_loops: int = 2,
    ) -> None:
        super().__init__()
        self.vocab = vocab
        self.width = width
        self.depth = depth
        self.n_heads = n_heads_for(width, n_heads)
        self.use_memory = use_memory
        self.use_reasoning = use_reasoning
        self.reasoning_loops = reasoning_loops

        self.tok_emb = nn.Embedding(vocab, width)
        self.blocks = nn.ModuleList(
            [Block(width, self.n_heads, use_memory) for _ in range(depth)]
        )
        self.norm_f = RMSNorm(width)
        if use_reasoning:
            self.reason_norm = RMSNorm(width)
            self.reason_block = Block(width, self.n_heads, use_memory=False)
            self.reason_gate = nn.Linear(width, width, bias=False)
        self.head = nn.Linear(width, n_classes, bias=False)
        self._cos, self._sin = rope_tables(4096, width // self.n_heads)
        self.apply(self._init)

    @staticmethod
    def _init(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, tokens: torch.Tensor, pad_mask: torch.Tensor | None = None) -> torch.Tensor:
        """Return logits for every position: [B, L, n_classes]."""
        B, L = tokens.shape
        if pad_mask is None:
            pad_mask = torch.ones(B, L, dtype=torch.bool, device=tokens.device)
        x = self.tok_emb(tokens)
        cos = self._cos.to(x.device, x.dtype)
        sin = self._sin.to(x.device, x.dtype)
        for blk in self.blocks:
            x = blk(x, cos, sin, pad_mask)
        if self.use_reasoning:
            base = x
            for _ in range(self.reasoning_loops):
                x = x + self.reason_gate(self.reason_block(self.reason_norm(x) + base, cos, sin, pad_mask))
        x = self.norm_f(x)
        return self.head(x)

    def trainable_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def total_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
