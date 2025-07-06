#!/bin/bash
# Setup script for Phi-3-mini training on Apple Silicon

echo "Setting up Phi-3-mini training environment for Apple Silicon..."

# Create virtual environment
uv init
uv venv
source .venv/bin/activate

# Install requirements
uv pip install --upgrade pip

# Install MLX and related packages
uv add mlx
uv add mlx-lm

# Install other dependencies
uv add numpy
uv add transformers
uv add tokenizers
uv add safetensors
uv add datasets

# Create directories
mkdir -p models
mkdir -p logs

echo "Setup complete!"
echo ""
echo "To activate the environment: source phi3_env/bin/activate"
echo "To start training: python phi3_mlx_training.py --data_file postgres_training_data.json"
echo ""
echo "Training configuration:"
echo "- Model: Phi-3-mini-4k-instruct (3.8B parameters)"
echo "- Method: LoRA fine-tuning (r=16, alpha=32)"
echo "- Expected VRAM usage: ~8-12GB"
echo "- Expected training time: ~2-4 hours for 1000 steps"