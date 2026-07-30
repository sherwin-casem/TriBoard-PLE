import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import torch
from model import Config, TinyLM

CHECKPOINT = os.path.join(os.path.dirname(__file__), "..", "runs", "ple-wiki60m-s0.pt")

VOCAB = None
VOCAB_OFF = None
VOCAB_N = 32768

def load_vocab():
    global VOCAB_BLOB, VOCAB_OFF, VOCAB_N
    import re
    path = os.path.join(os.path.dirname(__file__), "..", "firmware", "board_a_embeddings", "vocab.h")
    with open(path, "rb") as f:
        raw = f.read()

    m = re.search(rb"#define VOCAB_N\s+(\d+)", raw)
    VOCAB_N = int(m.group(1))

    m = re.search(rb"static const unsigned char VOCAB_BLOB\[(\d+)\] = \{(.*?)\};", raw, re.DOTALL)
    blob_data = m.group(2)
    numbers = [int(n) for n in re.findall(rb"\d+", blob_data)]
    VOCAB_BLOB = bytes(numbers)

    m = re.search(rb"static const int VOCAB_OFF\[(\d+)\] = \{(.*?)\};", raw, re.DOTALL)
    off_data = m.group(2)
    VOCAB_OFF = [int(n) for n in re.findall(rb"-?\d+", off_data)]

def decode_token(tid):
    if tid < 0 or tid >= VOCAB_N:
        return f"[{tid}]"
    s = VOCAB_OFF[tid]
    e = VOCAB_OFF[tid+1]
    return VOCAB_BLOB[s:e].decode("utf-8", errors="replace")

def sample(logits, temperature=0.5, top_k=40):
    logits = logits / temperature
    if top_k > 0:
        kth = torch.topk(logits, top_k).values.min()
        logits[logits < kth] = float("-inf")
    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, 1).item()

def generate(prompt_text, temperature=0.5, max_tokens=64):
    load_vocab()
    print(f"Loading checkpoint: {CHECKPOINT}")
    ck = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    cfg = Config(**ck["cfg"])
    model = TinyLM(cfg)
    model.load_state_dict(ck["state"])
    model.eval()
    print(f"Model: V={cfg.vocab_size} D={cfg.d_model} L={cfg.n_layers} Ph={cfg.ple_dim}")

    # Tokenize prompt manually (simple lookup)
    prompt_bytes = prompt_text.encode("utf-8")
    prompt_tokens = []
    i = 0
    while i < len(prompt_bytes):
        best_id = -1
        best_len = 0
        for t in range(VOCAB_N):
            s = VOCAB_OFF[t]
            e = VOCAB_OFF[t+1]
            tl = e - s
            if tl > 0 and i + tl <= len(prompt_bytes):
                if VOCAB_BLOB[s:e] == prompt_bytes[i:i+tl]:
                    if tl > best_len:
                        best_len = tl
                        best_id = t
        if best_id >= 0:
            prompt_tokens.append(best_id)
            i += best_len
        else:
            i += 1

    print(f"Prompt: '{prompt_text}'")
    print(f"Tokens ({len(prompt_tokens)}): {prompt_tokens}")
    if len(prompt_tokens) == 0:
        prompt_tokens = [1]

    ids = torch.tensor([prompt_tokens])
    max_seq = cfg.seq_len or 256

    generated_text = ""
    for g in range(max_tokens):
        with torch.no_grad():
            logits, _ = model(ids[:, -max_seq:])
        next_logits = logits[0, -1]
        tid = sample(next_logits, temperature)
        text = decode_token(tid)

        # Space insertion (same heuristic as firmware)
        if len(generated_text) > 0 and len(text) > 0 and text[0].isalpha():
            last_c = generated_text[-1]
            if last_c.isalpha() or last_c.isdigit():
                generated_text += " "

        generated_text += text
        print(f"  gen_{g}={tid} '{text}'")

        ids = torch.cat([ids, torch.tensor([[tid]])], dim=1)

        if tid == 0:
            break

    print(f"\n=== GENERATED TEXT ===")
    print(generated_text)
    print(f"=== END ===")

if __name__ == "__main__":
    import sys
    prompt = sys.argv[1] if len(sys.argv) > 1 else "The history of"
    temp = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5
    generate(prompt, temperature=temp)
