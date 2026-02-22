"""
Advanced Usage Example for Watermark Remover
Demonstrates custom detection parameters and batch processing workflows
"""

import sys
from pathlib import Path
import json

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from batch_processor import BatchWatermarkRemover
from complete_watermark_detector import CompleteWatermarkDetector
import cv2
import numpy as np


def example_custom_detector():
    """Use detector with custom parameters"""
    print("Example 1: Custom Detector Parameters")
    print("=" * 60)
    
    # Create custom detector
    detector = CompleteWatermarkDetector()
    
    # Customize detection parameters
    detector.search_height_ratio = 0.25  # Search larger area
    detector.search_width_ratio = 0.70
    detector.bg_color_lower = np.array([30, 30, 30])  # Wider color range
    detector.bg_color_upper = np.array([130, 130, 130])
    
    # Detect watermark
    image_path = "path/to/your/image.jpg"
    mask = detector.detect(image_path)
    
    coverage = np.sum(mask > 0) / mask.size * 100
    print(f"Detection coverage: {coverage:.2f}%")
    
    # Save mask
    cv2.imwrite("custom_mask.png", mask)
    print("Mask saved to: custom_mask.png")


def example_analyze_batch_results():
    """Analyze batch processing results"""
    print("\nExample 2: Analyze Batch Results")
    print("=" * 60)
    
    # Load processing report
    report_path = "path/to/output/processing_report.json"
    
    with open(report_path, 'r') as f:
        report = json.load(f)
    
    # Calculate statistics
    coverages = [r['mask_coverage'] for r in report['results'] if r['success']]
    
    if coverages:
        avg_coverage = sum(coverages) / len(coverages)
        min_coverage = min(coverages)
        max_coverage = max(coverages)
        
        print(f"Total images: {report['total_images']}")
        print(f"Successful: {report['successful']}")
        print(f"Average coverage: {avg_coverage:.2f}%")
        print(f"Coverage range: {min_coverage:.2f}% - {max_coverage:.2f}%")
        
        # Find images with unusual coverage
        unusual = [r for r in report['results'] 
                  if r['success'] and (r['mask_coverage'] < 2.0 or r['mask_coverage'] > 5.0)]
        
        if unusual:
            print(f"\nImages with unusual coverage ({len(unusual)}):")
            for r in unusual:
                print(f"  {Path(r['input']).name}: {r['mask_coverage']:.2f}%")


def example_pipeline_integration():
    """Example of integrating into a larger pipeline"""
    print("\nExample 3: Pipeline Integration")
    print("=" * 60)
    
    processor = BatchWatermarkRemover(device='cuda')  # Use GPU for speed
    
    # Process directory
    results = processor.process_directory(
        input_dir="input_images/",
        output_dir="output_images/",
        save_report=True
    )
    
    # Read output file list
    output_list_path = Path("output_images/output_files.txt")
    with open(output_list_path, 'r') as f:
        output_files = [line.strip() for line in f]
    
    print(f"Processed {len(output_files)} images")
    
    # Example: Upload to S3 or further processing
    for output_file in output_files:
        print(f"Ready for upload: {output_file}")
        # upload_to_s3(output_file)  # Your upload function
        # apply_further_processing(output_file)  # Your processing function


def example_quality_assurance():
    """Quality assurance workflow with visualization"""
    print("\nExample 4: Quality Assurance Workflow")
    print("=" * 60)
    
    processor = BatchWatermarkRemover(device='auto')
    
    # Process with full visualization
    results = processor.process_directory(
        input_dir="input_images/",
        output_dir="qa_output/",
        save_masks=True,
        visualize=True,
        save_report=True
    )
    
    print("Quality assurance files generated:")
    print("  - Clean images: qa_output/*.png")
    print("  - Detection masks: qa_output/*_mask.png")
    print("  - Visualizations: qa_output/*_detection.png")
    print("  - Report: qa_output/processing_report.json")
    
    # Identify images that need manual review
    review_needed = []
    for result in results:
        if result['success']:
            # Flag if coverage is unusually low or high
            if result['mask_coverage'] < 2.0 or result['mask_coverage'] > 5.0:
                review_needed.append(result)
    
    if review_needed:
        print(f"\n{len(review_needed)} images flagged for manual review:")
        for r in review_needed:
            print(f"  {Path(r['input']).name} - Coverage: {r['mask_coverage']:.2f}%")


def example_error_handling():
    """Robust error handling in batch processing"""
    print("\nExample 5: Error Handling")
    print("=" * 60)
    
    processor = BatchWatermarkRemover(device='auto')
    
    # Process with error handling
    input_dir = Path("input_images/")
    output_dir = Path("output_images/")
    
    if not input_dir.exists():
        print(f"Error: Input directory not found: {input_dir}")
        return
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        results = processor.process_directory(
            input_dir=input_dir,
            output_dir=output_dir,
            save_report=True
        )
        
        # Check for failures
        failures = [r for r in results if not r['success']]
        
        if failures:
            print(f"\n{len(failures)} images failed to process:")
            for r in failures:
                print(f"  {Path(r['input']).name}: {r.get('error', 'Unknown error')}")
        else:
            print("\nAll images processed successfully!")
            
    except Exception as e:
        print(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    print("Watermark Remover - Advanced Usage Examples")
    print("=" * 60)
    print("\nNote: Replace 'path/to/your/...' with actual paths")
    print()
    
    # Uncomment the example you want to run:
    # example_custom_detector()
    # example_analyze_batch_results()
    # example_pipeline_integration()
    # example_quality_assurance()
    # example_error_handling()
    
    print("\nTo run an example, uncomment the function call at the bottom of this file.")
