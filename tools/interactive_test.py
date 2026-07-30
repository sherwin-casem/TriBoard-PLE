import os, sys, torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from model import Config, TinyLM
from quantize import quantize_groupwise
from tokenizers import Tokenizer

tok = Tokenizer.from_file(os.path.join(os.path.dirname(__file__), "..", "data", "tokenizer", "bpe32768.json"))
ck = torch.load(os.path.join(os.path.dirname(__file__), "..", "runs", "ple-wiki60m-s0.pt"), map_location="cpu", weights_only=False)
cfg = Config(**ck["cfg"])
V, D, L, P = cfg.vocab_size, cfg.d_model, cfg.n_layers, cfg.ple_dim
P_half = P // 2
sd = ck["state"]

print("Loading model with NEW quant (tok_emb 8-bit, PLE_A 4-bit g=64)...")
dq_tok = quantize_groupwise(sd["tok_emb.weight"].float(), bits=8, group=128)
ple_a = sd["ple_table.weight"].float().view(V, L, P)[:, :, :P_half].contiguous()
ple_b = sd["ple_table.weight"].float().view(V, L, P)[:, :, P_half:].contiguous()
dq_ple_a = quantize_groupwise(ple_a, bits=4, group=64)
dq_ple_b = quantize_groupwise(ple_b, bits=4, group=128)
dq_sd = {}
for k, v in sd.items():
    if "norm" in k.lower(): dq_sd[k] = v.clone()
    elif k == "tok_emb.weight": dq_sd[k] = dq_tok
    elif k == "ple_table.weight": dq_sd[k] = torch.cat([dq_ple_a, dq_ple_b], dim=2).view(V, L * P)
    elif k in ("output.weight", "head.weight"): continue
    else: dq_sd[k] = quantize_groupwise(v.float(), bits=4, group=128)
if "head.weight" in sd: dq_sd["head.weight"] = dq_sd["tok_emb.weight"]

model = TinyLM(cfg)
model.load_state_dict(dq_sd, strict=False)
model.eval()
print("Ready.\n")

print("=" * 60)
print("Model: 56M PLE TinyLM | WikiText-103 | 2M tokens training")
print(f"Config: tok_emb 8-bit g=128 | PLE_A 4-bit g=64 | PLE_B 4-bit g=128")
print("=" * 60)
print("Este modelo solo vio 2M tokens de Wikipedia -- no tiene")
print("conocimiento especifico sobre personas, lugares o eventos.")
print("Genera texto que suena a Wikipedia pero inventa.")
print("=" * 60)

while True:
    try:
        prompt = input("\nPrompt> ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\nBye!")
        break
    if not prompt:
        continue
    if prompt.lower() in ("q", "quit", "exit"):
        break

    temp_str = input("temp (0.3-1.0, default 0.5)> ").strip()
    temp = float(temp_str) if temp_str else 0.5
    max_str = input("max tokens (default 48)> ").strip()
    max_tok = int(max_str) if max_str else 48

    ids = tok.encode(prompt).ids
    x = torch.tensor([ids])
    with torch.no_grad():
        logits = model(x)[0][0, -1]
    top5 = logits.topk(5).indices.tolist()

    print(f"\n--- tokens: {len(ids)}  ids={ids}")
    print(f"top-5 next:", [tok.decode([t]) for t in top5])

    with torch.no_grad():
        gen = model.generate(x, max_new_tokens=max_tok, temperature=temp, top_k=40)
    output = tok.decode(gen[0].tolist())
    print(f"--- output ({len(gen[0])} tokens) ---")
    print(output)
    print("---")
