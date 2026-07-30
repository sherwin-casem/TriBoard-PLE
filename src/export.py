import os
import struct
import sys

import numpy as np
import torch

from model import Config, TinyLM
from quantize import quantize_groupwise

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, "..", "runs")
OUT = os.path.join(HERE, "..", "firmware", "model")
MAGIC = 0x504C4532
GROUP = 128


def quant_pack(w, group=GROUP):
    w = w.float()
    out_shape = w.shape
    x = w.reshape(-1, out_shape[-1])
    rows, cols = x.shape
    n_groups = (cols + group - 1) // group
    q = torch.zeros(rows, cols)
    dq = torch.zeros(rows, cols)
    scales = torch.zeros(rows, n_groups)
    for gi in range(n_groups):
        a, b = gi * group, min((gi + 1) * group, cols)
        seg = x[:, a:b]
        sc = (seg.abs().amax(dim=1, keepdim=True) / 7).clamp_min(1e-8)
        sc = sc.half().float()
        scales[:, gi] = sc.squeeze(1)
        qi = torch.clamp(torch.round(seg / sc), -7, 7)
        q[:, a:b] = qi
        dq[:, a:b] = qi * sc
    dq = dq.reshape(out_shape)

    codes = (q.to(torch.int16) + 8).to(torch.uint8).numpy()
    row_bytes = (cols + 1) // 2
    packed = np.zeros((rows, row_bytes), dtype=np.uint8)
    lo = codes[:, 0::2]
    hi = codes[:, 1::2]
    packed[:, : lo.shape[1]] = lo
    packed[:, : hi.shape[1]] |= (hi << 4)
    scales16 = scales.numpy().astype(np.float16)
    return packed.reshape(-1), scales16.reshape(-1), dq


def quant_pack_8bit(w, group=GROUP):
    w = w.float()
    out_shape = w.shape
    x = w.reshape(-1, out_shape[-1])
    rows, cols = x.shape
    n_groups = (cols + group - 1) // group
    dq = torch.zeros(rows, cols)
    codes = np.zeros((rows, cols), dtype=np.uint8)
    scales = np.zeros((rows, n_groups), dtype=np.float16)
    for gi in range(n_groups):
        a, b = gi * group, min((gi + 1) * group, cols)
        seg = x[:, a:b]
        sc = (seg.abs().amax(dim=1, keepdim=True) / 127).clamp_min(1e-8)
        sc = sc.half().float()
        scales[:, gi] = sc.squeeze(1).numpy()
        qi = torch.clamp(torch.round(seg / sc), -127, 127)
        dq[:, a:b] = qi * sc
        codes[:, a:b] = (qi + 128).to(torch.uint8).numpy()
    dq = dq.reshape(out_shape)
    return codes.reshape(-1), scales.reshape(-1), dq


def write_tensor_quant(f, w, group=GROUP):
    packed, scales, dq = quant_pack(w, group=group)
    f.write(struct.pack("<i", group))
    f.write(packed.tobytes())
    f.write(scales.tobytes())
    return dq


def write_tensor_quant_8bit(f, w, group=GROUP):
    codes, scales, dq = quant_pack_8bit(w, group=group)
    f.write(struct.pack("<i", group))
    f.write(codes.tobytes())
    f.write(scales.tobytes())
    return dq


def write_tensor_fp32(f, w):
    arr = w.contiguous().numpy().astype(np.float32)
    f.write(arr.tobytes())
    return arr


def main():
    tag = sys.argv[1] if len(sys.argv) > 1 else "ple-wiki60m-s0"
    os.makedirs(OUT, exist_ok=True)
    ck = torch.load(os.path.join(RUNS, f"{tag}.pt"), map_location="cpu", weights_only=False)
    cfg = Config(**ck["cfg"])
    model = TinyLM(cfg)
    model.load_state_dict(ck["state"])
    model.eval()
    sd = model.state_dict()

    D = cfg.d_model
    L = cfg.n_layers
    P = cfg.ple_dim
    F = cfg.ffn_hidden
    V = cfg.vocab_size

    # Split PLE table, ple_model_proj, and ple_proj_norm into two halves
    P_half = P // 2
    ple_model_w = sd["ple_model_proj.weight"].view(L * P, D)
    ple_model_a = ple_model_w.view(L, P, D)[:, :P_half, :].contiguous().view(L * P_half, D)
    ple_model_b = ple_model_w.view(L, P, D)[:, P_half:, :].contiguous().view(L * P_half, D)
    norm_w = sd["ple_proj_norm.weight"]
    norm_a = norm_w[:P_half].contiguous()
    norm_b = norm_w[P_half:].contiguous()
    ple_table_w = sd["ple_table.weight"]
    ple_table_a = ple_table_w.view(V, L, P)[:, :, :P_half].contiguous()
    ple_table_b = ple_table_w.view(V, L, P)[:, :, P_half:].contiguous()

    # --- Board A: tok_emb (8-bit) + PLE_A (proj, norm) + out_norm ---
    path_a = os.path.join(OUT, "board_a.bin")
    with open(path_a, "wb") as f:
        f.write(struct.pack("<I", MAGIC))
        for v in [V, D, L, cfg.n_heads, F, P, P_half, cfg.seq_len, GROUP]:
            f.write(struct.pack("<i", v))
        f.write(struct.pack("<f", cfg.rope_theta))
        dq_tok_emb = write_tensor_quant_8bit(f, sd["tok_emb.weight"])
        dq_ple_proj_a = write_tensor_quant(f, ple_model_a)
        write_tensor_fp32(f, norm_a)
        write_tensor_fp32(f, sd["out_norm.weight"])
    print(f"Board A: {os.path.getsize(path_a)/1e6:.2f} MB  (tok_emb 8-bit + PLE_proj + out_norm)")

    # --- Board B: Core + PLE_B (proj, norm, table) ---
    path_b = os.path.join(OUT, "board_b.bin")
    with open(path_b, "wb") as f:
        f.write(struct.pack("<I", MAGIC))
        for v in [V, D, L, cfg.n_heads, F, P, P_half, cfg.seq_len, GROUP]:
            f.write(struct.pack("<i", v))
        f.write(struct.pack("<f", cfg.rope_theta))
        for i in range(L):
            p = f"blocks.{i}."
            write_tensor_fp32(f, sd[p + "attn_norm.weight"])
            write_tensor_quant(f, sd[p + "attn.qkv.weight"])
            write_tensor_quant(f, sd[p + "attn.proj.weight"])
            write_tensor_fp32(f, sd[p + "ffn_norm.weight"])
            write_tensor_quant(f, sd[p + "ffn.gate.weight"])
            write_tensor_quant(f, sd[p + "ffn.up.weight"])
            write_tensor_quant(f, sd[p + "ffn.down.weight"])
            write_tensor_quant(f, sd[p + "ple_gate.weight"])
            write_tensor_quant(f, sd[p + "ple_proj.weight"])
            write_tensor_fp32(f, sd[p + "ple_norm.weight"])
        dq_ple_proj_b = write_tensor_quant(f, ple_model_b)
        write_tensor_fp32(f, norm_b)
        dq_ple_b = write_tensor_quant(f, ple_table_b)
    print(f"Board B: {os.path.getsize(path_b)/1e6:.2f} MB  (core + PLE_B)")

    # --- Board C: PLE table A (4-bit, group=64) ---
    path_c = os.path.join(OUT, "board_c.bin")
    with open(path_c, "wb") as f:
        f.write(struct.pack("<I", MAGIC))
        for v in [V, D, L, cfg.n_heads, F, P, P_half, cfg.seq_len, 64]:
            f.write(struct.pack("<i", v))
        f.write(struct.pack("<f", cfg.rope_theta))
        dq_ple_a = write_tensor_quant(f, ple_table_a, group=64)
    print(f"Board C: {os.path.getsize(path_c)/1e6:.2f} MB  (ple_table_a group=64)")

    # --- Golden logits for verification ---
    dq_sd = {k: v.clone() for k, v in sd.items()}
    dq_sd["tok_emb.weight"] = dq_tok_emb
    dq_sd["ple_model_proj.weight"] = torch.cat([dq_ple_proj_a, dq_ple_proj_b], dim=0)
    dq_sd["ple_proj_norm.weight"] = torch.cat([norm_a, norm_b])
    dq_sd["ple_table.weight"] = torch.cat([dq_ple_a, dq_ple_b], dim=2).view(V, L * P)
    if "head.weight" in dq_sd:
        dq_sd["head.weight"] = dq_sd["tok_emb.weight"]

    gold = TinyLM(cfg)
    gold.load_state_dict(dq_sd)
    gold.eval()
    prompt = [1, 500, 1000, 200, 42, 777, 13, 99]
    ids = torch.tensor([prompt])
    with torch.no_grad():
        logits, _ = gold(ids)
    last = logits[0, -1].numpy().astype(np.float32)
    np.savez(os.path.join(OUT, "golden.npz"),
             prompt=np.array(prompt, dtype=np.int32), logits=last)
    with open(os.path.join(OUT, "golden.txt"), "w") as gf:
        gf.write(f"{len(prompt)}\n")
        gf.write(" ".join(str(t) for t in prompt) + "\n")
        gf.write("\n".join(f"{v:.6f}" for v in last) + "\n")
    top5 = last.argsort()[-5:][::-1]
    print(f"\ngolden: prompt={prompt}")
    print(f"golden: last-pos top5 = {top5.tolist()}")
    total_mb = (os.path.getsize(path_a) + os.path.getsize(path_b) + os.path.getsize(path_c)) / 1e6
    print(f"Total flash needed: {total_mb:.2f} MB")


if __name__ == "__main__":
    main()
