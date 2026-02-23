#!/bin/bash
# Watermark Eraser - One-Step Installation Script with AI Detection
# This script installs the watermark-eraser skill to your local environment

set -e  # Exit on error

echo "🚀 Installing Watermark Eraser Skill with AI Detection..."
echo ""

# Define installation directory
INSTALL_DIR="${HOME}/skills/watermark-eraser"

# Check if directory already exists
if [ -d "$INSTALL_DIR" ]; then
    echo "⚠️  Directory $INSTALL_DIR already exists."
    read -p "Do you want to overwrite it? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "❌ Installation cancelled."
        exit 1
    fi
    rm -rf "$INSTALL_DIR"
fi

# Create parent directory
mkdir -p "$(dirname "$INSTALL_DIR")"

# Clone repository
echo "📥 Cloning repository..."
git clone https://github.com/moose-lab/watermark-eraser.git "$INSTALL_DIR"

# Navigate to installation directory
cd "$INSTALL_DIR"

# Install Python dependencies
echo ""
echo "📦 Installing Python dependencies..."
pip install -q -r requirements.txt

# Download AI detection model
echo ""
echo "🤖 Downloading AI watermark detection model..."
MODEL_DIR="${HOME}/models"
MODEL_FILE="$MODEL_DIR/best_watermark_model_mit_b5_best.pth"

mkdir -p "$MODEL_DIR"

if [ ! -f "$MODEL_FILE" ]; then
    echo "   Downloading from Hugging Face (this may take a minute)..."
    curl -L -o "$MODEL_FILE" "https://huggingface.co/Sanster/watermark-segmentation/resolve/main/best_watermark_model_mit_b5_best.pth"
    echo "   ✓ Model downloaded successfully"
else
    echo "   ✓ Model already exists"
fi

# Test installation
echo ""
echo "🧪 Testing installation..."
python3 -c "import cv2, torch, simple_lama_inpainting, segmentation_models_pytorch; print('✓ All dependencies installed successfully')"

echo ""
echo "✅ Installation complete!"
echo ""
echo "📍 Installation directory: $INSTALL_DIR"
echo "📍 AI model location: $MODEL_FILE"
echo ""
echo "🎯 Quick Start:"
echo ""
echo "   # AI Detection (Recommended - works with any watermark)"
echo "   python $INSTALL_DIR/src/batch_processor_v2.py \\"
echo "     --input /path/to/images/ \\"
echo "     --output /path/to/results/ \\"
echo "     --detection ai \\"
echo "     --batch"
echo ""
echo "   # Fixed Detection (Faster, for specific watermarks)"
echo "   python $INSTALL_DIR/src/batch_processor_v2.py \\"
echo "     --input /path/to/images/ \\"
echo "     --output /path/to/results/ \\"
echo "     --detection fixed \\"
echo "     --batch"
echo ""
echo "📖 Usage with AI Agents (Claude, GPT, etc.):"
echo "   \"Use the watermark-eraser skill with AI detection to remove watermarks from my images\""
echo ""
echo "🎉 Ready to remove watermarks intelligently!"

