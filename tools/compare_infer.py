import os, sys, struct
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import torch
import numpy as np
from model import Config, TinyLM
from quantize import quantize_groupwise

RUNS = os.path.join(os.path.dirname(__file__), "..", "runs")
OUT = os.path.join(os.path.dirname(__file__), "..", "firmware", "model")

# Load the checkpoint
ck = torch.load(os.path.join(RUNS, "ple-wiki60m-s0.pt"), map_location="cpu", weights_only=False)
cfg = Config(**ck["cfg"])

# Full-precision model
model_fp = TinyLM(cfg)
model_fp.load_state_dict(ck["state"])
model_fp.eval()

# Quantized (dequantized) model - same as what firmware should produce
V, D, L, P = cfg.vocab_size, cfg.d_model, cfg.n_layers, cfg.ple_dim
P_half = P // 2
sd = ck["state"]
GROUP = 128

def quant_dequant(w):
    return quantize_groupwise(w.float(), bits=4, group=128)

dq_sd = {}
for k, v in sd.items():
    if "norm" in k.lower() or "weight" not in k:
        dq_sd[k] = v.clone()
    elif k == "tok_emb.weight" or k == "out_norm.weight":
        dq_sd[k] = v.clone()
    else:
        dq_sd[k] = quant_dequant(v)

model_q = TinyLM(cfg)
model_q.load_state_dict(dq_sd, strict=False)
model_q.eval()

prompt = torch.tensor([[5053, 6309, 5024, 13854]])

with torch.no_grad():
    logits_fp, _ = model_fp(prompt)
    logits_q, _ = model_q(prompt)

last_fp = logits_fp[0, -1]
last_q = logits_q[0, -1]

top5_fp = last_fp.topk(5).indices.tolist()
top5_q = last_q.topk(5).indices.tolist()
print(f"FP32  top5: {top5_fp}")
print(f"Quant top5: {top5_q}")

# Max diff
diff = (last_fp - last_q).abs().max().item()
print(f"Max logits diff: {diff:.6f}")

# Check if same top-1
print(f"FP32  top1: {top5_fp[0]}")
print(f"Quant top1: {top5_q[0]}")
print(f"Match: {top5_fp[0] == top5_q[0]}")
