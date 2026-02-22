# Local Deployment Guide

This guide explains how to deploy the **watermark-eraser** skill to your local environment, making it ready for AI agents to use directly.

---

## 🎯 Goal

After following this guide, you will have:
- The watermark-eraser skill installed at `~/skills/watermark-eraser/`
- All dependencies (OpenCV, PyTorch, LaMa) installed and ready
- A working skill that AI agents can call with simple instructions

---

## 🚀 Quick Installation (Recommended)

### One-Line Install

Run this command to automatically install everything:

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/moose-lab/watermark-eraser/main/install.sh)"
```

This will:
1. Clone the repository to `~/skills/watermark-eraser/`
2. Install all Python dependencies
3. Verify the installation
4. Create a test script

### Test the Installation

After installation, run:

```bash
bash ~/skills/watermark-eraser/test.sh
```

If you see `✅ Test passed!`, the skill is ready to use.

---

## 📋 Manual Installation

If you prefer to install manually or need more control:

### Step 1: Clone the Repository

```bash
# Create the skills directory
mkdir -p ~/skills

# Clone to the correct location
git clone https://github.com/moose-lab/watermark-eraser.git ~/skills/watermark-eraser
```

### Step 2: Install Dependencies

```bash
cd ~/skills/watermark-eraser
pip install -r requirements.txt
```

**Dependencies installed:**
- `opencv-python` - Image processing
- `torch` - Deep learning framework
- `simple-lama-inpainting` - LaMa inpainting model
- `numpy` - Numerical operations

### Step 3: Verify Installation

```bash
python3 -c "import cv2, torch, simple_lama_inpainting; print('✓ All dependencies installed')"
```

If this prints `✓ All dependencies installed`, you're ready to go.

---

## 🤖 Using with AI Agents

Once installed, you can instruct your AI agent to use the skill with simple natural language commands.

### Example Instructions

**Basic batch processing:**
> "Use the local `watermark-remover` skill to process all images in `/data/images/` and save them to `/data/clean/`."

**With GPU acceleration:**
> "Use the `watermark-remover` skill with GPU acceleration to process the images in `/input/` and save to `/output/`."

**With quality assurance:**
> "Use `watermark-remover` to process images and save the detection masks for quality checking."

### What the Agent Will Do

The agent will automatically run:

```bash
python ~/skills/watermark-eraser/src/batch_processor.py \
  --input /data/images/ \
  --output /data/clean/ \
  --batch
```

And deliver:
- Processed images (watermark-free)
- `processing_report.json` (detailed statistics)
- `output_files.txt` (list of output files)

---

## 🛠️ Manual Usage

If you want to run the skill manually (without an agent):

### Basic Command

```bash
python ~/skills/watermark-eraser/src/batch_processor.py \
  --input /path/to/images/ \
  --output /path/to/results/ \
  --batch
```

### Command-Line Options

| Option | Description | Example |
|--------|-------------|---------|
| `--input` | Input directory or single image | `/data/images/` |
| `--output` | Output directory or single image | `/data/clean/` |
| `--batch` | Enable batch processing mode | (flag) |
| `--device` | Processing device (auto/cuda/mps/cpu) | `--device cuda` |
| `--save-masks` | Save detection masks | (flag) |
| `--visualize` | Save visualization images | (flag) |

### Examples

**Process a single image:**
```bash
python ~/skills/watermark-eraser/src/batch_processor.py \
  --input photo.jpg \
  --output photo_clean.jpg
```

**Batch process with GPU:**
```bash
python ~/skills/watermark-eraser/src/batch_processor.py \
  --input ./images/ \
  --output ./clean/ \
  --batch \
  --device cuda
```

**Quality assurance workflow:**
```bash
python ~/skills/watermark-eraser/src/batch_processor.py \
  --input ./images/ \
  --output ./qa_results/ \
  --batch \
  --save-masks \
  --visualize
```

---

## 📊 Output Format

### Directory Structure

After processing, your output directory will contain:

```
output/
├── image1.png              # Processed images
├── image2.png
├── ...
├── processing_report.json  # Detailed statistics
└── output_files.txt        # List of all output files
```

### processing_report.json

```json
{
  "timestamp": "2026-02-22T12:00:00",
  "input_dir": "/data/images/",
  "output_dir": "/data/clean/",
  "total_images": 28,
  "successful": 28,
  "failed": 0,
  "detector_method": "V3 Complete Detector",
  "average_coverage": 3.21,
  "average_time": 5.8,
  "results": [
    {
      "input": "/data/images/1.png",
      "output": "/data/clean/1.png",
      "success": true,
      "mask_coverage": 3.21,
      "processing_time": 5.8
    }
  ]
}
```

### output_files.txt

```
/data/clean/1.png
/data/clean/2.png
/data/clean/3.png
...
```

This file list can be used for:
- Uploading to cloud storage
- Further image processing
- Creating archives
- Integration with other tools

---

## 🔧 Troubleshooting

### Issue: "Module not found" error

**Solution:**
```bash
cd ~/skills/watermark-eraser
pip install -r requirements.txt
```

### Issue: LaMa model download fails

**Solution:**
```bash
# Manually download the model
mkdir -p ~/.cache/lama
cd ~/.cache/lama
wget https://github.com/enesmsahin/simple-lama-inpainting/releases/download/v0.1.0/big-lama.pt
```

### Issue: GPU out of memory

**Solution:**
Use CPU mode instead:
```bash
python ~/skills/watermark-eraser/src/batch_processor.py \
  --input /data/images/ \
  --output /data/clean/ \
  --batch \
  --device cpu
```

### Issue: Detection not accurate

**Solution:**
1. Save visualization to check detection:
   ```bash
   python ~/skills/watermark-eraser/src/batch_processor.py \
     --input /data/images/ \
     --output /data/clean/ \
     --batch \
     --visualize
   ```
2. Check the `*_detection.png` files to see what was detected
3. If needed, adjust detection parameters in `src/complete_watermark_detector.py`

---

## 🎓 Advanced Configuration

### Customizing Detection Parameters

If you need to process different types of watermarks, you can modify the detector:

```bash
# Edit the detector configuration
nano ~/skills/watermark-eraser/src/complete_watermark_detector.py
```

Key parameters:
- `search_height_ratio`: Height of search area (default: 0.15 = bottom 15%)
- `search_width_ratio`: Width of search area (default: 0.50 = right 50%)
- `bg_color_lower/upper`: Color range for watermark detection
- `min_width/max_width`: Expected watermark width range
- `min_height/max_height`: Expected watermark height range

### Integration with Other Tools

**Example: Process and upload to S3**

```python
import subprocess
import boto3
from pathlib import Path

# Process images
subprocess.run([
    'python', '~/skills/watermark-eraser/src/batch_processor.py',
    '--input', '/data/images/',
    '--output', '/data/clean/',
    '--batch'
])

# Upload to S3
s3 = boto3.client('s3')
output_files = Path('/data/clean/output_files.txt').read_text().splitlines()

for file_path in output_files:
    s3.upload_file(file_path, 'my-bucket', Path(file_path).name)
```

---

## 📚 Additional Resources

- **[Architecture](ARCHITECTURE.md)**: Technical details of the V3 detector and LaMa inpainting
- **[Development History](DEVELOPMENT.md)**: How the 100% coverage detector was built
- **[Examples](../examples/)**: Python scripts for basic and advanced usage
- **[GitHub Repository](https://github.com/moose-lab/watermark-eraser)**: Source code and issues

---

## ✅ Verification Checklist

After installation, verify that:

- [ ] Repository is cloned to `~/skills/watermark-eraser/`
- [ ] All dependencies are installed (`python3 -c "import cv2, torch, simple_lama_inpainting"`)
- [ ] Test script passes (`bash ~/skills/watermark-eraser/test.sh`)
- [ ] AI agent can find and use the skill
- [ ] Batch processing works on sample images

---

## 🎉 You're Ready!

Your watermark-eraser skill is now deployed and ready for AI agents to use. Simply instruct your agent to use the skill, and it will handle the rest automatically.

**Example instruction:**
> "Use the local watermark-remover skill to process all images in `/my/images/` and save to `/my/clean_images/`."

The agent will take care of everything from detection to inpainting to delivering the results!
