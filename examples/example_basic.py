"""
Basic Usage Example for Watermark Remover
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from batch_processor import BatchWatermarkRemover


def example_single_image():
    """Process a single image"""
    print("Example 1: Processing a single image")
    print("=" * 60)
    
    # Initialize processor
    processor = BatchWatermarkRemover(device='auto')
    
    # Process single image
    result = processor.process_single(
        image_path="path/to/your/image.jpg",
        output_path="path/to/output/result.jpg",
        save_mask=True,
        visualize=True
    )
    
    print(f"Success: {result['success']}")
    print(f"Output: {result['output']}")
    print(f"Coverage: {result['mask_coverage']:.2f}%")


def example_batch_processing():
    """Process a directory of images"""
    print("\nExample 2: Batch processing a directory")
    print("=" * 60)
    
    # Initialize processor
    processor = BatchWatermarkRemover(device='auto')
    
    # Process directory
    results = processor.process_directory(
        input_dir="path/to/your/images/",
        output_dir="path/to/output/results/",
        save_masks=True,
        visualize=True,
        save_report=True
    )
    
    # Print summary
    successful = sum(1 for r in results if r['success'])
    print(f"\nProcessed {len(results)} images")
    print(f"Successful: {successful}")
    print(f"Failed: {len(results) - successful}")


def example_with_custom_output():
    """Process with custom output naming"""
    print("\nExample 3: Custom output naming")
    print("=" * 60)
    
    processor = BatchWatermarkRemover(device='auto')
    
    # Process with custom output path
    input_path = Path("path/to/your/image.jpg")
    output_path = input_path.parent / f"{input_path.stem}_clean{input_path.suffix}"
    
    result = processor.process_single(
        image_path=input_path,
        output_path=output_path
    )
    
    print(f"Saved to: {result['output']}")


if __name__ == '__main__':
    print("Watermark Remover - Basic Usage Examples")
    print("=" * 60)
    print("\nNote: Replace 'path/to/your/...' with actual paths")
    print()
    
    # Uncomment the example you want to run:
    # example_single_image()
    # example_batch_processing()
    # example_with_custom_output()
    
    print("\nTo run an example, uncomment the function call at the bottom of this file.")
