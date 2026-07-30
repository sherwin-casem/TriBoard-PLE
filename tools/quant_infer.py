import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import torch, re
from model import Config, TinyLM
from quantize import quantize_groupwise

RUNS = os.path.join(os.path.dirname(__file__), "..", "runs")
ck = torch.load(os.path.join(RUNS, "ple-wiki60m-s0.pt"), map_location="cpu", weights_only=False)
cfg = Config(**ck["cfg"])
sd = ck["state"]

# Quantize all large weights (same as export.py)
GROUP = 128
def quant_dequant(w):
    w = w.float()
    out_shape = w.shape
    x = w.reshape(-1, out_shape[-1])
    out, cols = x.shape
    n_groups = (cols + GROUP - 1) // GROUP
    q = torch.zeros(out, cols)
    dq = torch.zeros(out, cols)
    for gi in range(n_groups):
        a, b = gi * GROUP, min((gi + 1) * GROUP, cols)
        seg = x[:, a:b]
        sc = (seg.abs().amax(dim=1, keepdim=True) / 7).clamp_min(1e-8)
        sc = sc.half().float()
        qi = torch.clamp(torch.round(seg / sc), -7, 7)
        q[:, a:b] = qi
        dq[:, a:b] = qi * sc
    return dq.reshape(out_shape)

dq_sd = {}
for k, v in sd.items():
    if "norm" in k.lower():
        dq_sd[k] = v.clone()
    elif k == "out_norm.weight":
        dq_sd[k] = v.clone()
    elif k == "tok_emb.weight":
        dq_sd[k] = v.clone()
    elif k == "head.weight":
        dq_sd[k] = v.clone()
    else:
        dq_sd[k] = quant_dequant(v)

model = TinyLM(cfg)
model.load_state_dict(dq_sd, strict=False)
model.eval()

# Load vocab
path = os.path.join(os.path.dirname(__file__), "..", "firmware", "board_a_embeddings", "vocab.h")
with open(path, "rb") as f:
    raw = f.read()
m = re.search(rb"#define VOCAB_N\s+(\d+)", raw)
VOCAB_N = int(m.group(1))
m = re.search(rb"static const unsigned char VOCAB_BLOB\[(\d+)\] = \{(.*?)\};", raw, re.DOTALL)
VOCAB_BLOB = bytes([int(n) for n in re.findall(rb"\d+", m.group(2))])
m = re.search(rb"static const int VOCAB_OFF\[(\d+)\] = \{(.*?)\};", raw, re.DOTALL)
VOCAB_OFF = [int(n) for n in re.findall(rb"-?\d+", m.group(2))]

def decode(tid):
    if tid < 0 or tid >= VOCAB_N: return f"[{tid}]"
    s, e = VOCAB_OFF[tid], VOCAB_OFF[tid+1]
    return VOCAB_BLOB[s:e].decode("utf-8", errors="replace")

def sample(logits, temp=0.5, top_k=40):
    logits = logits / temp
    if top_k > 0:
        kth = torch.topk(logits, top_k).values.min()
        logits[logits < kth] = float("-inf")
    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, 1).item()

prompt_text = "The history of Argentina"
prompt_bytes = prompt_text.encode("utf-8")
prompt_tokens = []
i = 0
while i < len(prompt_bytes):
    best_id, best_len = -1, 0
    for t in range(VOCAB_N):
        s, e = VOCAB_OFF[t], VOCAB_OFF[t+1]
        tl = e - s
        if tl > 0 and i + tl <= len(prompt_bytes) and VOCAB_BLOB[s:e] == prompt_bytes[i:i+tl]:
            if tl > best_len: best_len, best_id = tl, t
    if best_id >= 0: prompt_tokens.append(best_id); i += best_len
    else: i += 1

print(f"Tokens: {prompt_tokens}")
ids = torch.tensor([prompt_tokens])
gen_text = ""

for g in range(64):
    with torch.no_grad():
        logits, _ = model(ids[:, -128:])
    tid = sample(logits[0, -1], 0.5)
    text = decode(tid)
    if len(gen_text) > 0 and len(text) > 0 and text[0].isalpha() and gen_text[-1].isalpha():
        gen_text += " "
    gen_text += text
    print(f"  gen_{g}={tid} '{text}'")
    ids = torch.cat([ids, torch.tensor([[tid]])], dim=1)
    if tid == 0: break

print(f"\n=== QUANTIZED LOCAL ===")
print(gen_text)
