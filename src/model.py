"""Tiny decoder-only transformer with Per-Layer Embeddings (PLE).

Adapted from slvDev/esp32-ai. The architecture is identical to the reference:
a small dense core runs in SRAM while a large embedding table lives in flash
(Per-Layer Embeddings from Google Gemma). For the distributed version, the
model is split across 3 ESP32-S3 boards at export time, not at training time.
"""

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Config:
    arm: str = "ple"
    vocab_size: int = 32768
    d_model: int = 96
    n_layers: int = 6
    n_heads: int = 4
    ffn_hidden: int = 66
    seq_len: int = 512
    ple_dim: int = 128
    rope_theta: float = 10000.0

    @property
    def head_dim(self):
        return self.d_model // self.n_heads

    @property
    def uses_per_layer(self):
        return self.arm in ("ple", "ple_notable")

    @property
    def table_width(self):
        return self.n_layers * self.ple_dim


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return self.weight * x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)


def build_rope(seq_len, head_dim, theta, device):
    inv = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    t = torch.arange(seq_len, device=device).float()
    freqs = torch.outer(t, inv)
    return torch.cos(freqs), torch.sin(freqs)


def apply_rope(x, cos, sin):
    x1, x2 = x.chunk(2, dim=-1)
    cos = cos[None, None, : x.shape[2], :]
    sin = sin[None, None, : x.shape[2], :]
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)


class Attention(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)

    def forward(self, x, cos, sin):
        B, T, C = x.shape
        H, Dh = self.cfg.n_heads, self.cfg.head_dim
        q, k, v = self.qkv(x).split(C, dim=2)
        q = q.view(B, T, H, Dh).transpose(1, 2)
        k = k.view(B, T, H, Dh).transpose(1, 2)
        v = v.view(B, T, H, Dh).transpose(1, 2)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        o = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.proj(o.transpose(1, 2).contiguous().view(B, T, C))


class SwiGLU(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.gate = nn.Linear(cfg.d_model, cfg.ffn_hidden, bias=False)
        self.up = nn.Linear(cfg.d_model, cfg.ffn_hidden, bias=False)
        self.down = nn.Linear(cfg.ffn_hidden, cfg.d_model, bias=False)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


class Block(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.attn_norm = RMSNorm(cfg.d_model)
        self.attn = Attention(cfg)
        self.ffn_norm = RMSNorm(cfg.d_model)
        self.ffn = SwiGLU(cfg)
        if cfg.uses_per_layer:
            self.ple_gate = nn.Linear(cfg.d_model, cfg.ple_dim, bias=False)
            self.ple_proj = nn.Linear(cfg.ple_dim, cfg.d_model, bias=False)
            self.ple_norm = RMSNorm(cfg.d_model)

    def forward(self, x, cos, sin, ple=None):
        x = x + self.attn(self.attn_norm(x), cos, sin)
        x = x + self.ffn(self.ffn_norm(x))
        if ple is not None:
            g = F.gelu(self.ple_gate(x))
            x = x + self.ple_norm(self.ple_proj(g * ple))
        return x


class TinyLM(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg

        if cfg.arm == "fatembed":
            self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.table_width)
            self.emb_down = nn.Linear(cfg.table_width, cfg.d_model, bias=False)
            self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        else:
            self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
            self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
            self.head.weight = self.tok_emb.weight  # tied

        if cfg.uses_per_layer:
            self.ple_model_proj = nn.Linear(cfg.d_model, cfg.n_layers * cfg.ple_dim, bias=False)
            self.ple_proj_norm = RMSNorm(cfg.ple_dim)
        if cfg.arm == "ple":
            self.ple_table = nn.Embedding(cfg.vocab_size, cfg.n_layers * cfg.ple_dim)

        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.out_norm = RMSNorm(cfg.d_model)

        self.apply(self._init)
        for n, p in self.named_parameters():
            if n.endswith("proj.weight") or n.endswith("down.weight"):
                nn.init.normal_(p, std=0.02 / math.sqrt(2 * cfg.n_layers))
        if cfg.arm == "fatembed":
            nn.init.normal_(self.emb_down.weight, std=cfg.table_width**-0.5)
        for block in self.blocks:
            if cfg.uses_per_layer:
                nn.init.zeros_(block.ple_norm.weight)

        cos, sin = build_rope(cfg.seq_len, cfg.head_dim, cfg.rope_theta, "cpu")
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def _init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, idx, targets=None):
        cfg = self.cfg
        x = self.tok_emb(idx)
        if cfg.arm == "fatembed":
            x = self.emb_down(x)

        ple = None
        if cfg.uses_per_layer:
            B, T = idx.shape
            ple = self.ple_model_proj(x) * (cfg.d_model**-0.5)
            ple = self.ple_proj_norm(ple.view(B, T, cfg.n_layers, cfg.ple_dim))
            if cfg.arm == "ple":
                table = self.ple_table(idx).view(B, T, cfg.n_layers, cfg.ple_dim)
                ple = (ple + table * (cfg.ple_dim**0.5)) * (2**-0.5)

        for i, block in enumerate(self.blocks):
            x = block(x, self.cos, self.sin, None if ple is None else ple[:, :, i])

        x = self.out_norm(x)
        logits = self.head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, cfg.vocab_size), targets.reshape(-1), ignore_index=-1
            )
        return logits, loss

    def param_budget(self):
        table = 0
        if self.cfg.arm == "ple":
            table += self.ple_table.weight.numel()
        if self.cfg.arm == "fatembed":
            table += self.tok_emb.weight.numel()
        stream = self.head.weight.numel()
        seen, total = set(), 0
        for p in self.parameters():
            if id(p) in seen:
                continue
            seen.add(id(p))
            total += p.numel()
        return {"core": total - table - stream, "stream": stream,
                "table": table, "total": total}

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=0.8, top_k=40):
        for _ in range(max_new_tokens):
            idx_c = idx[:, -self.cfg.seq_len :]
            logits, _ = self(idx_c)
            logits = logits[:, -1, :] / temperature
            if top_k:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("inf")
            probs = F.softmax(logits, dim=-1)
            idx = torch.cat([idx, torch.multinomial(probs, 1)], dim=1)
        return idx


def make_model(arm, target_core, base: Config = None, verbose=True, fixed_ffn=None):
    cfg = Config(**{**(base.__dict__ if base else {}), "arm": arm})
    if fixed_ffn is not None:
        cfg.ffn_hidden = fixed_ffn
        model = TinyLM(cfg)
        if verbose:
            b = model.param_budget()
            print(f"[{arm}] d_model={cfg.d_model} layers={cfg.n_layers} "
                  f"ffn={cfg.ffn_hidden} ple_dim={cfg.ple_dim} core={b['core']:,} "
                  f"stream={b['stream']:,} table={b['table']:,} total={b['total']:,}")
        return model
    table_budget = cfg.vocab_size * cfg.n_layers * cfg.ple_dim
    if arm == "bigcore":
        best = None
        for d in range(cfg.d_model, 8 * cfg.d_model, cfg.n_heads):
            trial = Config(**{**cfg.__dict__, "arm": "baseline", "d_model": d, "ffn_hidden": 2 * d})
            if TinyLM(trial).param_budget()["core"] <= target_core + table_budget:
                best = trial
            else:
                break
        cfg = best
    else:
        lo, hi = 1, 64 * cfg.d_model
        while lo < hi:
            mid = (lo + hi + 1) // 2
            cfg.ffn_hidden = mid
            if TinyLM(cfg).param_budget()["core"] <= target_core:
                lo = mid
            else:
                hi = mid - 1
        cfg.ffn_hidden = lo

    model = TinyLM(cfg)
    if verbose:
        b = model.param_budget()
        print(
            f"[{arm}] d_model={cfg.d_model} layers={cfg.n_layers} ffn={cfg.ffn_hidden} "
            f"core={b['core']:,} stream={b['stream']:,} table={b['table']:,} "
            f"total={b['total']:,}"
        )
    return model
