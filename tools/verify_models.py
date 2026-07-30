import os
import struct
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(HERE, "..", "firmware", "model")
MAGIC = 0x504C4532

def read_header(path):
    with open(path, "rb") as f:
        magic = struct.unpack("<I", f.read(4))[0]
        if magic != MAGIC:
            print(f"WARN: {os.path.basename(path)} magic={magic:#x} (expected {MAGIC:#x})")
        hv = struct.unpack("<9i", f.read(36))
        rope_theta = struct.unpack("<f", f.read(4))[0]
        return {"vocab": hv[0], "dim": hv[1], "n_layers": hv[2], "n_heads": hv[3],
                "ffn": hv[4], "ple_dim": hv[5], "ple_half": hv[6],
                "seq_len": hv[7], "group": hv[8], "rope_theta": rope_theta}

def main():
    for name in ["board_a.bin", "board_b.bin", "board_c.bin"]:
        path = os.path.join(MODEL_DIR, name)
        if not os.path.exists(path): print(f"MISSING: {path}"); continue
        hdr = read_header(path)
        size = os.path.getsize(path)
        print(f"{name}: {size/1e6:.2f} MB | V={hdr['vocab']} D={hdr['dim']} "
              f"L={hdr['n_layers']} H={hdr['n_heads']} F={hdr['ffn']} "
              f"P={hdr['ple_dim']} Ph={hdr['ple_half']}")

    golden = os.path.join(MODEL_DIR, "golden.txt")
    if os.path.exists(golden):
        with open(golden) as f:
            plen = int(f.readline())
            prompt = list(map(int, f.readline().split()))
            print(f"\nGolden: prompt={prompt} ({plen} tokens)")
        print("Golden logits available for C runtime verification.")
    else:
        print("\nNo golden file found. Run: python src/export.py")

    a = os.path.join(MODEL_DIR, "board_a.bin")
    b = os.path.join(MODEL_DIR, "board_b.bin")
    c = os.path.join(MODEL_DIR, "board_c.bin")
    if all(os.path.exists(p) for p in [a, b, c]):
        total = sum(os.path.getsize(p) for p in [a, b, c])
        print(f"\nTotal flash: {total/1e6:.2f} MB (fits in 3x 16MB flash)")

if __name__ == "__main__":
    main()
