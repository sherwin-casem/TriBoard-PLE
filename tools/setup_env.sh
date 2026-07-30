#!/bin/bash
# Install all Python dependencies for the training pipeline.
set -e

echo "=== Installing Python dependencies ==="
pip install torch numpy tokenizers datasets tqdm

echo ""
echo "=== Dependencies installed ==="
echo "Quick test:"
python -c "import torch; print(f'PyTorch {torch.__version__}, MPS: {torch.backends.mps.is_available()}')"
echo ""
echo "Next steps:"
echo "  1. Prepare dataset:  python src/dataset.py --dataset tinystories"
echo "  2. Train model:      python src/train.py --arm ple --steps 4000 --tag deploy"
echo "  3. Quantize:         python src/quantize.py --tag deploy"
echo "  4. Export:           python src/export.py ple-deploy-s0"
echo "  5. Generate vocab:   python src/gen_assets.py"
