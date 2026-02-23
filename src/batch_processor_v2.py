"""
Batch Watermark Removal Processor with AI Detection

This script provides batch processing capabilities for watermark removal,
supporting both AI-powered detection and traditional fixed detection methods.
"""

import cv2
import numpy as np
from pathlib import Path
import json
import argparse
from typing import List, Dict, Optional
from tqdm import tqdm

from ai_watermark_detector import AIWatermarkDetector
from lama_inpainter import LamaInpainter


class BatchWatermarkRemover:
    """
    Batch processor for watermark removal with AI detection.
    """
    
    def __init__(
        self,
        detection_method: str = "ai",
        device: str = "auto",
        save_masks: bool = False,
        save_visualizations: bool = False
    ):
        """
        Initialize batch processor.
        
        Args:
            detection_method: Detection method ('ai' or 'fixed')
            device: Device to use ('auto', 'cuda', 'mps', 'cpu')
            save_masks: Whether to save detection masks
            save_visualizations: Whether to save visualization images
        """
        self.detection_method = detection_method
        self.device = device
        self.save_masks = save_masks
        self.save_visualizations = save_visualizations
        
        # Initialize detector
        if detection_method == "ai":
            print(f"Initializing AI watermark detector...")
            self.detector = AIWatermarkDetector(device=device)
        else:
            print(f"Using fixed detection method (V3)...")
            from complete_watermark_detector import CompleteWatermarkDetector
            self.detector = CompleteWatermarkDetector()
        
        # Initialize inpainter
        print(f"Initializing LaMa inpainter...")
        self.inpainter = LamaInpainter(device=device)
        
        print(f"✓ Batch processor ready ({detection_method} detection)")
    
    def process_image(
        self,
        image_path: Path,
        output_dir: Path
    ) -> Dict:
        """
        Process a single image.
        
        Args:
            image_path: Path to input image
            output_dir: Directory to save outputs
        
        Returns:
            Processing statistics dictionary
        """
        # Load image
        image = cv2.imread(str(image_path))
        if image is None:
            return {"status": "error", "message": "Failed to load image"}
        
        # Detect watermark
        if self.detection_method == "ai":
            mask, viz = self.detector.detect(image, return_visualization=self.save_visualizations)
            coverage = self.detector.get_coverage_percentage(mask)
        else:
            mask = self.detector.detect(image)
            coverage = (np.sum(mask > 0) / (mask.shape[0] * mask.shape[1])) * 100
            viz = None
        
        # Remove watermark
        result_pil = self.inpainter.inpaint(image, mask)
        result = cv2.cvtColor(np.array(result_pil), cv2.COLOR_RGB2BGR)
        
        # Save outputs
        base_name = image_path.stem
        
        # Save cleaned image
        output_path = output_dir / f"{base_name}.png"
        cv2.imwrite(str(output_path), result)
        
        # Save mask if requested
        if self.save_masks:
            mask_path = output_dir / f"{base_name}_mask.png"
            cv2.imwrite(str(mask_path), mask)
        
        # Save visualization if requested
        if self.save_visualizations and viz is not None:
            viz_path = output_dir / f"{base_name}_detection.png"
            cv2.imwrite(str(viz_path), viz)
        
        return {
            "status": "success",
            "input_file": str(image_path),
            "output_file": str(output_path),
            "coverage_percent": round(coverage, 2),
            "detection_method": self.detection_method
        }
    
    def process_batch(
        self,
        input_dir: Path,
        output_dir: Path
    ) -> Dict:
        """
        Process all images in a directory.
        
        Args:
            input_dir: Directory containing input images
            output_dir: Directory to save outputs
        
        Returns:
            Processing report dictionary
        """
        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Find all images
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
        image_files = [
            f for f in input_dir.iterdir()
            if f.suffix.lower() in image_extensions
        ]
        
        if not image_files:
            print(f"No images found in {input_dir}")
            return {"status": "error", "message": "No images found"}
        
        print(f"Found {len(image_files)} images to process")
        
        # Process images
        results = []
        output_files = []
        
        for image_path in tqdm(image_files, desc="Processing images"):
            result = self.process_image(image_path, output_dir)
            results.append(result)
            
            if result["status"] == "success":
                output_files.append(result["output_file"])
        
        # Generate report
        successful = [r for r in results if r["status"] == "success"]
        failed = [r for r in results if r["status"] == "error"]
        
        report = {
            "total_images": len(image_files),
            "successful": len(successful),
            "failed": len(failed),
            "success_rate": round(len(successful) / len(image_files) * 100, 2),
            "detection_method": self.detection_method,
            "device": self.device,
            "average_coverage": round(
                np.mean([r["coverage_percent"] for r in successful]), 2
            ) if successful else 0,
            "results": results
        }
        
        # Save report
        report_path = output_dir / "processing_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Save output file list
        output_list_path = output_dir / "output_files.txt"
        with open(output_list_path, 'w') as f:
            f.write('\n'.join(output_files))
        
        print(f"\n✓ Processing complete!")
        print(f"  Total: {report['total_images']}")
        print(f"  Successful: {report['successful']}")
        print(f"  Failed: {report['failed']}")
        print(f"  Success rate: {report['success_rate']}%")
        print(f"  Average coverage: {report['average_coverage']}%")
        print(f"\n✓ Report saved to: {report_path}")
        print(f"✓ Output list saved to: {output_list_path}")
        
        return report


def main():
    parser = argparse.ArgumentParser(
        description="Batch watermark removal with AI detection"
    )
    
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input image file or directory"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output directory"
    )
    
    parser.add_argument(
        "--detection",
        type=str,
        choices=["ai", "fixed"],
        default="ai",
        help="Detection method: 'ai' (default) or 'fixed'"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        choices=["auto", "cuda", "mps", "cpu"],
        default="auto",
        help="Device to use (default: auto)"
    )
    
    parser.add_argument(
        "--save-masks",
        action="store_true",
        help="Save detection masks"
    )
    
    parser.add_argument(
        "--save-visualizations",
        action="store_true",
        help="Save detection visualizations"
    )
    
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Process entire directory (default: single file)"
    )
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    output_path = Path(args.output)
    
    # Initialize processor
    processor = BatchWatermarkRemover(
        detection_method=args.detection,
        device=args.device,
        save_masks=args.save_masks,
        save_visualizations=args.save_visualizations
    )
    
    # Process
    if args.batch or input_path.is_dir():
        # Batch processing
        processor.process_batch(input_path, output_path)
    else:
        # Single file processing
        output_path.mkdir(parents=True, exist_ok=True)
        result = processor.process_image(input_path, output_path)
        
        if result["status"] == "success":
            print(f"\n✓ Processing complete!")
            print(f"  Input: {result['input_file']}")
            print(f"  Output: {result['output_file']}")
            print(f"  Coverage: {result['coverage_percent']}%")
            print(f"  Method: {result['detection_method']}")
        else:
            print(f"\n✗ Processing failed: {result['message']}")


if __name__ == "__main__":
    main()
