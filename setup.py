"""
Setup script for Watermark Remover
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text(encoding="utf-8") if readme_file.exists() else ""

setup(
    name="watermark-remover",
    version="3.0.0",
    author="Manus AI",
    author_email="contact@manus.im",
    description="High-precision watermark removal tool with 100% coverage V3 detector and LaMa inpainting",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/manus-ai/watermark-remover",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Image Processing",
        "Topic :: Multimedia :: Graphics",
    ],
    python_requires=">=3.8",
    install_requires=[
        "opencv-python>=4.8.0",
        "numpy>=1.24.0",
        "Pillow>=10.0.0",
        "torch>=2.0.0",
        "simple-lama-inpainting>=0.1.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
        ],
        "viz": [
            "matplotlib>=3.7.0",
            "scikit-image>=0.21.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "watermark-remover=batch_processor:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
