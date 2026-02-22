"""
Batch Watermark Removal Processor
Combines detection and inpainting for batch processing
"""

import cv2
import numpy as np
from PIL import Image
from pathlib import Path
from typing import Union, Optional, List, Dict
import json
from datetime import datetime

from complete_watermark_detector import CompleteWatermarkDetector as WatermarkDetector
from lama_inpainter import LamaInpainter


class BatchWatermarkRemover:
    """
    Batch watermark removal combining detection and inpainting
    """
    
    def __init__(self, 
                 device: str = 'auto'):
        """
        Initialize batch processor
        
        Args:
            device: Device for inpainting ('auto', 'cuda', 'cpu', 'mps')
        """
        self.detector = WatermarkDetector()
        self.inpainter = LamaInpainter(device=device)
        
        print(f"Batch Watermark Remover initialized")
        print(f"  Detection method: V3 Complete Detector (100% coverage)")
        print(f"  Inpainting device: {self.inpainter.device}")
    
    def process_single(self,
                      image_path: Union[str, Path],
                      output_path: Optional[Union[str, Path]] = None,
                      mask_path: Optional[Union[str, Path]] = None,
                      save_mask: bool = False,
                      visualize: bool = False) -> Dict:
        """
        Process single image
        
        Args:
            image_path: Path to input image
            output_path: Path to save output (auto-generated if None)
            mask_path: Path to manual mask (optional)
            save_mask: Whether to save detected mask
            visualize: Whether to save visualization of detection
            
        Returns:
            Dictionary with processing results
        """
        image_path = Path(image_path)
        
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        # Setup output path
        if output_path is None:
            output_path = image_path.parent / f"{image_path.stem}_nowatermark{image_path.suffix}"
        else:
            output_path = Path(output_path)
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        print(f"\nProcessing: {image_path.name}")
        
        # Step 1: Detect watermark
        print("  [1/3] Detecting watermark...")
        if mask_path:
            # Use manual mask if provided
            import cv2
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                raise ValueError(f"Cannot read mask: {mask_path}")
        else:
            # Use V3 detector
            mask = self.detector.detect(image_path)
        
        # Check if watermark detected
        mask_coverage = np.sum(mask > 0) / mask.size * 100
        print(f"  Detected watermark coverage: {mask_coverage:.2f}%")
        
        if mask_coverage < 0.1:
            print("  Warning: Very small or no watermark detected")
        
        # Step 2: Save mask if requested
        mask_save_path = None
        if save_mask:
            mask_save_path = output_path.parent / f"{image_path.stem}_mask.png"
            cv2.imwrite(str(mask_save_path), mask)
            print(f"  Saved mask to: {mask_save_path.name}")
        
        # Step 3: Visualize detection if requested
        viz_save_path = None
        if visualize:
            viz_save_path = output_path.parent / f"{image_path.stem}_detection.png"
            self.detector.visualize_detection(image_path, mask, viz_save_path)
            print(f"  Saved visualization to: {viz_save_path.name}")
        
        # Step 4: Inpaint
        print("  [2/3] Inpainting...")
        result = self.inpainter.inpaint(image_path, mask)
        
        # Step 5: Save result
        print("  [3/3] Saving result...")
        result.save(str(output_path))
        print(f"  ✓ Saved to: {output_path.name}")
        
        return {
            'input': str(image_path),
            'output': str(output_path),
            'mask': str(mask_save_path) if mask_save_path else None,
            'visualization': str(viz_save_path) if viz_save_path else None,
            'mask_coverage': mask_coverage,
            'success': True
        }
    
    def process_directory(self,
                         input_dir: Union[str, Path],
                         output_dir: Optional[Union[str, Path]] = None,
                         mask_dir: Optional[Union[str, Path]] = None,
                         save_masks: bool = False,
                         visualize: bool = False,
                         extensions: tuple = ('.jpg', '.jpeg', '.png', '.bmp', '.webp'),
                         save_report: bool = True) -> List[Dict]:
        """
        Process all images in directory
        
        Args:
            input_dir: Directory containing input images
            output_dir: Directory to save outputs (auto-generated if None)
            mask_dir: Directory containing manual masks (optional)
            save_masks: Whether to save detected masks
            visualize: Whether to save detection visualizations
            extensions: Image file extensions to process
            save_report: Whether to save processing report
            
        Returns:
            List of processing results for each image
        """
        input_dir = Path(input_dir)
        
        if not input_dir.exists():
            raise FileNotFoundError(f"Input directory not found: {input_dir}")
        
        # Setup output directory
        if output_dir is None:
            output_dir = input_dir.parent / f"{input_dir.name}_nowatermark"
        else:
            output_dir = Path(output_dir)
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Find all images
        image_files = []
        for ext in extensions:
            image_files.extend(input_dir.glob(f"*{ext}"))
            image_files.extend(input_dir.glob(f"*{ext.upper()}"))
        
        image_files = sorted(set(image_files))
        
        print(f"\nBatch Processing")
        print("=" * 60)
        print(f"Input directory: {input_dir}")
        print(f"Output directory: {output_dir}")
        print(f"Found {len(image_files)} images")
        print("=" * 60)
        
        if len(image_files) == 0:
            print("No images found!")
            return []
        
        # Process each image
        results = []
        
        for i, image_path in enumerate(image_files, 1):
            print(f"\n[{i}/{len(image_files)}]", end=" ")
            
            try:
                # Find corresponding mask if mask_dir provided
                mask_path = None
                if mask_dir:
                    mask_dir_path = Path(mask_dir)
                    potential_mask = mask_dir_path / f"{image_path.stem}_mask.png"
                    if potential_mask.exists():
                        mask_path = potential_mask
                
                # Setup output path
                output_path = output_dir / image_path.name
                
                # Process
                result = self.process_single(
                    image_path=image_path,
                    output_path=output_path,
                    mask_path=mask_path,
                    save_mask=save_masks,
                    visualize=visualize
                )
                
                results.append(result)
                
            except Exception as e:
                print(f"  ✗ Error: {e}")
                results.append({
                    'input': str(image_path),
                    'output': None,
                    'error': str(e),
                    'success': False
                })
        
        # Print summary
        print("\n" + "=" * 60)
        print("Processing Summary")
        print("=" * 60)
        successful = sum(1 for r in results if r['success'])
        print(f"Total: {len(results)}")
        print(f"Successful: {successful}")
        print(f"Failed: {len(results) - successful}")
        
        # Save report
        if save_report:
            report_path = output_dir / "processing_report.json"
            report = {
                'timestamp': datetime.now().isoformat(),
                'input_dir': str(input_dir),
                'output_dir': str(output_dir),
                'total_images': len(results),
                'successful': successful,
                'failed': len(results) - successful,
                'detector_method': 'V3 Complete Detector',
                'results': results
            }
            
            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            
            print(f"\nReport saved to: {report_path}")
        
        # Save output paths list
        output_list_path = output_dir / "output_files.txt"
        with open(output_list_path, 'w', encoding='utf-8') as f:
            for result in results:
                if result['success'] and result['output']:
                    f.write(result['output'] + '\n')
        
        print(f"Output list saved to: {output_list_path}")
        
        return results
    
    def process_with_custom_masks(self,
                                  image_mask_pairs: List[tuple],
                                  output_dir: Union[str, Path]) -> List[Dict]:
        """
        Process images with custom mask pairs
        
        Args:
            image_mask_pairs: List of (image_path, mask_path) tuples
            output_dir: Directory to save outputs
            
        Returns:
            List of processing results
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        results = []
        
        for i, (image_path, mask_path) in enumerate(image_mask_pairs, 1):
            print(f"\n[{i}/{len(image_mask_pairs)}]", end=" ")
            
            try:
                image_path = Path(image_path)
                output_path = output_dir / image_path.name
                
                result = self.process_single(
                    image_path=image_path,
                    output_path=output_path,
                    mask_path=mask_path
                )
                
                results.append(result)
                
            except Exception as e:
                print(f"  ✗ Error: {e}")
                results.append({
                    'input': str(image_path),
                    'output': None,
                    'error': str(e),
                    'success': False
                })
        
        return results


def main():
    """Command-line interface"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Batch Watermark Removal Tool',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process single image
  python batch_processor.py --input image.jpg --output result.jpg
  
  # Process directory
  python batch_processor.py --input ./images/ --output ./results/ --batch
  
  # Use manual masks
  python batch_processor.py --input ./images/ --mask-dir ./masks/ --batch
  
  # Save detection masks and visualizations
  python batch_processor.py --input ./images/ --batch --save-masks --visualize
        """
    )
    
    parser.add_argument('--input', required=True,
                       help='Input image or directory')
    parser.add_argument('--output', required=True,
                       help='Output image or directory')
    parser.add_argument('--batch', action='store_true',
                       help='Process directory in batch mode')
    # Note: --method removed in V3, always uses Complete Detector
    parser.add_argument('--mask', default=None,
                       help='Manual mask file (single image mode)')
    parser.add_argument('--mask-dir', default=None,
                       help='Directory containing masks (batch mode)')
    parser.add_argument('--save-masks', action='store_true',
                       help='Save detected masks')
    parser.add_argument('--visualize', action='store_true',
                       help='Save detection visualizations')
    parser.add_argument('--device', default='auto',
                       choices=['auto', 'cuda', 'cpu', 'mps'],
                       help='Device for inpainting')
    
    args = parser.parse_args()
    
    # Initialize processor
    processor = BatchWatermarkRemover(
        device=args.device
    )
    
    # Process
    if args.batch:
        processor.process_directory(
            input_dir=args.input,
            output_dir=args.output,
            mask_dir=args.mask_dir,
            save_masks=args.save_masks,
            visualize=args.visualize
        )
    else:
        processor.process_single(
            image_path=args.input,
            output_path=args.output,
            mask_path=args.mask,
            save_mask=args.save_masks,
            visualize=args.visualize
        )


if __name__ == '__main__':
    main()
