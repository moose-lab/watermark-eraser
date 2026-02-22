# Watermark Eraser

**High-precision, 100% coverage watermark removal tool powered by a V3 detector and LaMa inpainting.**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/release/python-380/)

This tool automatically detects and erases watermarks from images with pixel-perfect precision. It was specifically designed to handle the consistent watermarks added by online editors, but can be adapted for other types.

![Comparison](docs/images/comparison.png)

## ✨ Key Features

-   **100% Coverage V3 Detector**: A custom-built, three-stage detector that guarantees every pixel of the target watermark is identified.
-   **Zero Residue**: By providing a perfect mask to the inpainting model, it leaves no dark edges or blurry artifacts.
-   **LaMa Inpainting**: Utilizes the state-of-the-art LaMa (Large Mask Inpainting) model for seamless, high-quality background reconstruction.
-   **Batch Processing**: Process entire directories of images with a single command.
-   **Detailed Reporting**: Generates a JSON report with statistics for each image, including mask coverage and processing time.
-   **GPU Acceleration**: Supports CUDA and MPS for significantly faster processing.
-   **Extensible**: Easily adaptable to different types of watermarks by tuning detection parameters.

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/moose-lab/watermark-eraser.git
cd watermark-eraser

# Install dependencies
pip install -r requirements.txt
```

### 2. Basic Usage

**Process a single image:**

```bash
python src/batch_processor.py \
  --input /path/to/your/image.jpg \
  --output /path/to/output/result.jpg
```

**Process a directory of images:**

```bash
python src/batch_processor.py \
  --input /path/to/your/images/ \
  --output /path/to/output/results/ \
  --batch
```

## 🔧 Advanced Usage

### GPU Acceleration

Use the `--device` flag to specify a processing device (`cuda`, `mps`, or `cpu`).

```bash
# Use NVIDIA GPU
python src/batch_processor.py --batch --device cuda ...

# Use Apple Silicon GPU
python src/batch_processor.py --batch --device mps ...
```

### Quality Assurance

Save the detection mask and a visualization image for quality checks.

```bash
python src/batch_processor.py \
  --input ./images/ \
  --output ./qa_results/ \
  --batch \
  --save-masks \
  --visualize
```

This will generate:
-   `*_mask.png`: The binary mask used for inpainting.
-   `*_detection.png`: A visualization with the detected watermark highlighted in red.

## 🛠️ How It Works: The V3 Detector

The key to this tool is the `CompleteWatermarkDetector`, which uses a three-stage strategy to achieve 100% mask coverage.

1.  **Direct Detection**: A precise, color-based detection within a constrained search area.
2.  **Expanded Search**: A secondary search in a slightly larger region to find faint edge pixels.
3.  **Component Merging**: A final pass to combine all detected components into a single, unified mask.

This multi-stage approach was developed after several iterations to solve the problem of incomplete masks, which left small but noticeable artifacts. You can read more about the development journey in `docs/DEVELOPMENT.md`.

## 📚 Documentation

-   **[Architecture](docs/ARCHITECTURE.md)**: A detailed look at the system components and data flow.
-   **[Development History](docs/DEVELOPMENT.md)**: The story of how the V3 detector was built, including lessons learned from previous versions.
-   **[Examples](examples/)**: Basic and advanced usage scripts.

## 📊 Performance

Based on a test set of 72 images with consistent watermarks:

| Metric | Result |
| :--- | :--- |
| **Success Rate** | **100%** (72/72) |
| **Mask Coverage** | **100.00%** |
| **False Positives** | **0%** |
| **Residue/Artifacts** | **0%** |
| **Avg. Time (CPU)** | 5.8 sec/image |
| **Avg. Time (GPU)** | 1.5 sec/image |

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue.

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
