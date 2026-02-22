# Watermark Eraser

**High-precision, 100% coverage watermark removal tool powered by a V3 detector and LaMa inpainting.**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/release/python-380/)

This tool automatically detects and erases watermarks from images with pixel-perfect precision. It can be used as a standalone Python project or as a **Manus Skill** for direct integration with Code Agents.

![Comparison](docs/images/comparison.png)

---

## 🚀 **Usage with Code Agents (Manus Skill)**

**This is the recommended way to use the tool for automated, large-scale processing.** A Code Agent can directly use this repository as a skill to process images.

### How It Works

1.  **Provide the GitHub URL**: Give the agent the URL to this repository: `https://github.com/moose-lab/watermark-eraser.git`
2.  **Assign the Task**: Instruct the agent to process a directory of images.

### Example Prompt for a Code Agent

> "Use the skill from `https://github.com/moose-lab/watermark-eraser.git` to remove the watermarks from all images in the `/path/to/my/images` directory. Save the clean images to `/path/to/my/output`."

### Agent Workflow

The agent will automatically perform the following steps:

1.  **Clone the Repository**: `git clone https://github.com/moose-lab/watermark-eraser.git`
2.  **Install Dependencies**: `pip install -r requirements.txt`
3.  **Execute Batch Processing**: Run the `batch_processor.py` script with the specified input and output directories.

```bash
# Agent will run this command
python watermark-eraser/src/batch_processor.py \
  --input /path/to/my/images/ \
  --output /path/to/my/output/ \
  --batch
```

4.  **Deliver Results**: The agent will provide a link to the output directory containing:
    -   Clean, watermark-free images.
    -   A detailed `processing_report.json`.
    -   An `output_files.txt` list for easy integration.

---

## ✨ Key Features

-   **100% Coverage V3 Detector**: A custom-built, three-stage detector that guarantees every pixel of the target watermark is identified.
-   **Zero Residue**: By providing a perfect mask to the inpainting model, it leaves no dark edges or blurry artifacts.
-   **LaMa Inpainting**: Utilizes the state-of-the-art LaMa (Large Mask Inpainting) model for seamless, high-quality background reconstruction.
-   **Batch Processing**: Process entire directories of images with a single command.
-   **Detailed Reporting**: Generates a JSON report with statistics for each image, including mask coverage and processing time.
-   **GPU Acceleration**: Supports CUDA and MPS for significantly faster processing.

## 🛠️ Standalone Usage

For local development or integration into non-agent workflows.

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

## 🔧 Advanced Options

### GPU Acceleration

Use the `--device` flag to specify a processing device (`cuda`, `mps`, or `cpu`).

```bash
# Use NVIDIA GPU
python src/batch_processor.py --batch --device cuda ...
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

## 📚 Documentation

-   **[Architecture](docs/ARCHITECTURE.md)**: A detailed look at the system components and data flow.
-   **[Development History](docs/DEVELOPMENT.md)**: The story of how the V3 detector was built.
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

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue.

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
