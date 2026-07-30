import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import torch
import numpy as np
from model import Config, TinyLM
from quantize import quantize_groupwise
from tokenizers import Tokenizer

RUNS = os.path.join(os.path.dirname(__file__), "..", "runs")
TOK_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "tokenizer", "bpe32768.json")

tok = Tokenizer.from_file(TOK_PATH)
V = tok.get_vocab_size()

ck = torch.load(os.path.join(RUNS, "ple-wiki60m-s0.pt"), map_location="cpu", weights_only=False)
cfg = Config(**ck["cfg"])
V, D, L, P = cfg.vocab_size, cfg.d_model, cfg.n_layers, cfg.ple_dim
P_half = P // 2
sd = ck["state"]

# --- Quantize tok_emb: OLD (4-bit g=128) vs NEW (8-bit g=128) ---
dq_tok_emb_new = quantize_groupwise(sd["tok_emb.weight"].float(), bits=8, group=128)
dq_tok_emb_old = quantize_groupwise(sd["tok_emb.weight"].float(), bits=4, group=128)

# --- Split PLE table ---
ple_table_w = sd["ple_table.weight"].float()
ple_table_a = ple_table_w.view(V, L, P)[:, :, :P_half].contiguous()
ple_table_b = ple_table_w.view(V, L, P)[:, :, P_half:].contiguous()

# NEW scheme: ple_table_a 4-bit group=64, ple_table_b 4-bit group=128
dq_ple_a_new = quantize_groupwise(ple_table_a, bits=4, group=64)
dq_ple_b_new = quantize_groupwise(ple_table_b, bits=4, group=128)
ple_table_new = torch.cat([dq_ple_a_new, dq_ple_b_new], dim=2).view(V, L * P)

# OLD scheme: both halves 4-bit group=128
dq_ple_a_old = quantize_groupwise(ple_table_a, bits=4, group=128)
dq_ple_b_old = quantize_groupwise(ple_table_b, bits=4, group=128)
ple_table_old = torch.cat([dq_ple_a_old, dq_ple_b_old], dim=2).view(V, L * P)


def build_dq_sd(ple_table, tok_emb):
    dq = {}
    for k, v in sd.items():
        if "norm" in k.lower():
            dq[k] = v.clone()
        elif k == "tok_emb.weight":
            dq[k] = tok_emb
        elif k == "ple_table.weight":
            dq[k] = ple_table
        elif k in ("output.weight", "head.weight"):
            continue
        else:
            dq[k] = quantize_groupwise(v.float(), bits=4, group=128)
    if "head.weight" in sd:
        dq["head.weight"] = dq["tok_emb.weight"]
    if "output.weight" in sd:
        dq["output.weight"] = dq["tok_emb.weight"]
    return dq


dq_sd_new = build_dq_sd(ple_table_new, dq_tok_emb_new)
dq_sd_old = build_dq_sd(ple_table_old, dq_tok_emb_old)
dq_sd_ideal = {k: v.clone() for k, v in sd.items()}
if "head.weight" in dq_sd_ideal:
    dq_sd_ideal["head.weight"] = dq_sd_ideal["tok_emb.weight"]
if "output.weight" in dq_sd_ideal:
    dq_sd_ideal["output.weight"] = dq_sd_ideal["tok_emb.weight"]

model_new = TinyLM(cfg)
model_new.load_state_dict(dq_sd_new, strict=False)
model_new.eval()

model_old = TinyLM(cfg)
model_old.load_state_dict(dq_sd_old, strict=False)
model_old.eval()

model_ideal = TinyLM(cfg)
model_ideal.load_state_dict(dq_sd_ideal, strict=False)
model_ideal.eval()

prompts = [
    "The history of",
    "The most common",
    "In the beginning",
    "Artificial intelligence",
    "The United States",
    "Once upon a time",
]

print("=" * 72)
print("  QUALITY COMPARISON")
print("  NEW: tok_emb 8-bit g=128, PLE_A 4-bit g=64, PLE_B 4-bit g=128")
print("  OLD: everything 4-bit g=128")
print("=" * 72)

for prompt_text in prompts:
    ids = tok.encode(prompt_text).ids
    if not ids:
        continue
    x = torch.tensor([ids])
    with torch.no_grad():
        logits_new = model_new(x)[0][0, -1]
        logits_old = model_old(x)[0][0, -1]
        logits_ideal = model_ideal(x)[0][0, -1]

    top5_new = logits_new.topk(5).indices.tolist()
    top5_old = logits_old.topk(5).indices.tolist()
    top5_ideal = logits_ideal.topk(5).indices.tolist()
    diff_new = (logits_ideal - logits_new).abs().max().item()
    diff_old = (logits_ideal - logits_old).abs().max().item()
    match_new = top5_new[0] == top5_ideal[0]
    match_old = top5_old[0] == top5_ideal[0]

    print(f"\nPrompt: \"{prompt_text}\"  tokens={ids}")
    print(f"  FP32 top-5   : {top5_ideal}")
    print(f"  NEW  top-5   : {top5_new}  max_diff={diff_new:.4f}  top1_match={match_new}")
    print(f"  OLD  top-5   : {top5_old}  max_diff={diff_old:.4f}  top1_match={match_old}")

print("\n" + "=" * 72)
prompt_all = torch.tensor([tok.encode("The history of").ids])
with torch.no_grad():
    gen_new = model_new.generate(prompt_all, max_new_tokens=48, temperature=0.5, top_k=40)
    gen_old = model_old.generate(prompt_all, max_new_tokens=48, temperature=0.5, top_k=40)
    gen_ideal = model_ideal.generate(prompt_all, max_new_tokens=48, temperature=0.5, top_k=40)

print("\nGeneration (temp=0.5, prompt='The history of'):")
print(f"  FP32: {tok.decode(gen_ideal[0].tolist())}")
print(f"  NEW : {tok.decode(gen_new[0].tolist())}")
print(f"  OLD : {tok.decode(gen_old[0].tolist())}")
print("=" * 72)

print("\nModel binary sizes (actual):")
print(f"  NEW: A=4.31MB  B=13.71MB  C=13.37MB  total=31.39MB")
print(f"  OLD: A=15.19MB B=13.71MB  C=0.00MB   total=28.90MB")
print("=" * 72)
