#!/bin/bash
# Linux Setup Script for Traffic Scene Graph Project

set -e

echo "=== Starting Environment Setup (Linux) ==="

# 1. Update and install basic dependencies
# Handle root environment (common on AutoDL)
if command -v sudo > /dev/null 2>&1; then
    SUDO="sudo"
else
    SUDO=""
fi

$SUDO apt-get update && $SUDO apt-get install -y \
    python3-pip \
    python3-venv \
    git \
    libgl1-mesa-glx \
    libglib2.0-0

# 2. Setup Virtual Environment (using venv)
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

source venv/bin/activate

# 3. Install Python dependencies
echo "Installing requirements..."
# Use Tsinghua mirror for faster and more stable access in China
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
pip install --upgrade pip

# Install PyTorch first (needed for PyG)
echo "Installing PyTorch..."
pip install torch==2.1.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

pip install -r requirements.txt

# 4. Install PyTorch Geometric Dependencies (Important for your project)
echo "Installing PyTorch Geometric dependencies..."
# Determine torch version for correct whl
TORCH_VER=$(python -c "import torch; print(torch.__version__.split('+')[0])")
CUDA_VER=$(python -c "import torch; if torch.cuda.is_available(): print('cu' + torch.version.cuda.replace('.', '')) else: print('cpu')")

pip install torch-scatter torch-sparse torch-cluster torch-spline-conv \
    -f https://data.pyg.org/whl/torch-${TORCH_VER}+${CUDA_VER}.html

pip install torch-geometric

# 5. Install the project in editable mode
pip install -e .

# 4. (Optional) Docker Setup - if the user prefers Docker
# docker-compose up --build -d

echo "=== Setup Complete! ==="
echo "To activate the environment: source venv/bin/activate"
