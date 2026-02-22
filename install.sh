#!/bin/bash
# Watermark Eraser - One-Step Installation Script
# This script installs the watermark-eraser skill to your local environment

set -e  # Exit on error

echo "🚀 Installing Watermark Eraser Skill..."
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

# Test installation
echo ""
echo "🧪 Testing installation..."
python3 -c "import cv2, torch, simple_lama_inpainting; print('✓ All dependencies installed successfully')"

# Create a test script
cat > "$INSTALL_DIR/test.sh" << 'EOF'
#!/bin/bash
# Quick test script for watermark-eraser skill

echo "🧪 Testing watermark-eraser skill..."

# Create test directory
TEST_DIR="/tmp/watermark_test_$$"
mkdir -p "$TEST_DIR"/{input,output}

# Create a test image with watermark
python3 << 'PYTHON'
import cv2
import numpy as np
import sys

img = np.ones((400, 600, 3), dtype=np.uint8) * 200
wm_h, wm_w = 30, 150
wm_y, wm_x = 360, 440
img[wm_y:wm_y+wm_h, wm_x:wm_x+wm_w] = [80, 80, 80]
cv2.putText(img, 'Watermark', (wm_x+10, wm_y+20), 
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
cv2.imwrite('/tmp/watermark_test_$$/input/test.png', img)
print('✓ Test image created')
PYTHON

# Run the skill
echo "🚀 Running watermark-eraser..."
python3 "$HOME/skills/watermark-eraser/src/batch_processor.py" \
  --input "$TEST_DIR/input/" \
  --output "$TEST_DIR/output/" \
  --batch > /dev/null 2>&1

# Check result
if [ -f "$TEST_DIR/output/test.png" ]; then
    echo "✅ Test passed! Skill is working correctly."
    echo "   Test files: $TEST_DIR"
else
    echo "❌ Test failed! Output file not found."
    exit 1
fi

# Cleanup
rm -rf "$TEST_DIR"
echo "🎉 Watermark-eraser skill is ready to use!"
EOF

chmod +x "$INSTALL_DIR/test.sh"

echo ""
echo "✅ Installation complete!"
echo ""
echo "📍 Installation directory: $INSTALL_DIR"
echo ""
echo "🧪 To test the installation, run:"
echo "   bash $INSTALL_DIR/test.sh"
echo ""
echo "📖 Usage with AI Agents:"
echo "   Tell your agent: \"Use the local watermark-remover skill to process images\""
echo ""
echo "📖 Manual usage:"
echo "   python $INSTALL_DIR/src/batch_processor.py --input /input/dir/ --output /output/dir/ --batch"
echo ""
echo "🎉 Ready to remove watermarks!"
