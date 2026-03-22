#!/bin/bash
# Linux Setup Script for Traffic Scene Graph Project

set -e

echo "=== Starting Environment Setup (Linux) ==="

# 1. Update and install basic dependencies
sudo apt-get update && sudo apt-get install -y \
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
pip install --upgrade pip
pip install -r requirements.txt

# 4. (Optional) Docker Setup - if the user prefers Docker
# docker-compose up --build -d

echo "=== Setup Complete! ==="
echo "To activate the environment: source venv/bin/activate"
