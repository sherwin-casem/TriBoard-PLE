"""Dataset loading for TinyStories and WikiText-103.

TinyStories is fetched via the HuggingFace datasets library.
WikiText-103 is fetched the same way. Both are tokenized with a BPE tokenizer
trained on the corpus and saved as memory-mapped .bin files (uint16 tokens),
matching the format expected by train.py.
"""

import os
import struct
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from tqdm import tqdm

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE.parent / "data"
TINYSTORIES_DIR = DATA_DIR / "tinystories"
WIKITEXT_DIR = DATA_DIR / "wikitext103"
TOKENIZER_DIR = DATA_DIR / "tokenizer"


def download_tinystories():
    """Download TinyStories from HuggingFace and save as raw text."""
    from datasets import load_dataset

    TINYSTORIES_DIR.mkdir(parents=True, exist_ok=True)
    out_train = TINYSTORIES_DIR / "train.txt"
    out_val = TINYSTORIES_DIR / "val.txt"

    if out_train.exists() and out_val.exists():
        print(f"TinyStories already downloaded at {TINYSTORIES_DIR}")
        return

    print("Downloading TinyStories...")
    ds = load_dataset("roneneldan/TinyStories", split="train")
    print(f"Writing {len(ds)} train stories...")
    with open(out_train, "w", encoding="utf-8") as f:
        for row in tqdm(ds, desc="train"):
            f.write(row["text"] + "\n")

    ds_val = load_dataset("roneneldan/TinyStories", split="validation")
    print(f"Writing {len(ds_val)} val stories...")
    with open(out_val, "w", encoding="utf-8") as f:
        for row in tqdm(ds_val, desc="val"):
            f.write(row["text"] + "\n")

    print(f"TinyStories saved to {TINYSTORIES_DIR}")


def download_wikitext103():
    """Download WikiText-103 from HuggingFace and save as raw text."""
    from datasets import load_dataset

    WIKITEXT_DIR.mkdir(parents=True, exist_ok=True)
    out_train = WIKITEXT_DIR / "train.txt"
    out_val = WIKITEXT_DIR / "val.txt"

    if out_train.exists() and out_val.exists():
        print(f"WikiText-103 already downloaded at {WIKITEXT_DIR}")
        return

    print("Downloading WikiText-103...")
    ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="train")
    print(f"Writing {len(ds)} train articles...")
    with open(out_train, "w", encoding="utf-8") as f:
        for row in tqdm(ds, desc="train"):
            text = row["text"].strip()
            if text:
                f.write(text + "\n")

    ds_val = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="validation")
    print(f"Writing {len(ds_val)} val articles...")
    with open(out_val, "w", encoding="utf-8") as f:
        for row in tqdm(ds_val, desc="val"):
            text = row["text"].strip()
            if text:
                f.write(text + "\n")

    print(f"WikiText-103 saved to {WIKITEXT_DIR}")


def train_tokenizer(corpus_dir: Path, vocab_size: int = 32768) -> Tokenizer:
    """Train a BPE tokenizer on a corpus directory and save it."""
    TOKENIZER_DIR.mkdir(parents=True, exist_ok=True)
    tok_path = TOKENIZER_DIR / f"bpe{vocab_size}.json"

    if tok_path.exists():
        print(f"Tokenizer already exists at {tok_path}")
        return Tokenizer.from_file(str(tok_path))

    print(f"Training BPE tokenizer (vocab_size={vocab_size})...")
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["[UNK]", "[PAD]", "[BOS]", "[EOS]"],
        min_frequency=2,
    )

    files = list(corpus_dir.glob("*.txt"))
    tokenizer.train([str(f) for f in files], trainer)
    tokenizer.save(str(tok_path))
    print(f"Tokenizer saved to {tok_path}")
    return tokenizer


def tokenize_and_save(tokenizer: Tokenizer, text_file: Path, out_file: Path):
    """Tokenize a text file and save as uint16 binary (mmap-able)."""
    if out_file.exists():
        print(f"  {out_file.name} already exists, skipping")
        return

    print(f"  Tokenizing {text_file.name}...")
    tokens = []
    with open(text_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            encoded = tokenizer.encode(line)
            tokens.extend(encoded.ids)

    arr = np.array(tokens, dtype=np.uint16)
    arr.tofile(str(out_file))
    print(f"  Saved {len(tokens):,} tokens ({len(arr) * 2 / 1e6:.1f} MB) to {out_file}")


def prepare_dataset(name: str = "tinystories", vocab_size: int = 32768):
    """Full pipeline: download, train tokenizer, tokenize to binary."""
    print(f"=== Preparing {name} (vocab_size={vocab_size}) ===\n")

    if name == "tinystories":
        download_tinystories()
        corpus_dir = TINYSTORIES_DIR
    elif name == "wikitext103":
        download_wikitext103()
        corpus_dir = WIKITEXT_DIR
    else:
        raise ValueError(f"Unknown dataset: {name}")

    tokenizer = train_tokenizer(corpus_dir, vocab_size)

    bin_dir = DATA_DIR
    for split in ["train", "val"]:
        txt = corpus_dir / f"{split}.txt"
        suffix = "" if vocab_size == 32768 else f"_v{vocab_size}"
        out = bin_dir / f"{split}{suffix}.bin"
        tokenize_and_save(tokenizer, txt, out)

    print("\n=== Dataset ready ===")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="tinystories", choices=["tinystories", "wikitext103"])
    ap.add_argument("--vocab-size", type=int, default=32768)
    args = ap.parse_args()
    prepare_dataset(args.dataset, args.vocab_size)
