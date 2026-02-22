"""
LaMa Inpainting Module
Wrapper for simple-lama-inpainting package
"""

import numpy as np
from PIL import Image
from pathlib import Path
from typing import Union, Optional
import warnings


class LamaInpainter:
    """
    LaMa-based image inpainting for watermark removal
    """
    
    def __init__(self, device: str = 'auto'):
        """
        Initialize LaMa inpainter
        
        Args:
            device: Device to use ('auto', 'cuda', 'cpu')
        """
        self.device = self._setup_device(device)
        self.model = None
        self._load_model()
    
    def _setup_device(self, device: str) -> str:
        """Setup computation device"""
        if device == 'auto':
            try:
                import torch
                if torch.cuda.is_available():
                    return 'cuda'
                elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                    return 'mps'
                else:
                    return 'cpu'
            except ImportError:
                return 'cpu'
        return device
    
    def _load_model(self):
        """Load LaMa model"""
        try:
            from simple_lama_inpainting import SimpleLama
            self.model = SimpleLama()
            print(f"LaMa model loaded successfully on {self.device}")
        except ImportError:
            warnings.warn(
                "simple-lama-inpainting not installed. "
                "Install with: pip install simple-lama-inpainting"
            )
            self.model = None
    
    def inpaint(self, 
                image: Union[np.ndarray, Image.Image, str, Path],
                mask: Union[np.ndarray, Image.Image, str, Path]) -> Image.Image:
        """
        Inpaint image using mask
        
        Args:
            image: Input image (numpy array, PIL Image, or path)
            mask: Binary mask (numpy array, PIL Image, or path)
                  White (255) regions will be inpainted
        
        Returns:
            Inpainted PIL Image
        """
        if self.model is None:
            raise RuntimeError(
                "LaMa model not loaded. "
                "Please install: pip install simple-lama-inpainting"
            )
        
        # Convert inputs to PIL Images
        image_pil = self._to_pil_image(image)
        mask_pil = self._to_pil_mask(mask)
        
        # Ensure mask is same size as image
        if image_pil.size != mask_pil.size:
            mask_pil = mask_pil.resize(image_pil.size, Image.LANCZOS)
        
        # Perform inpainting
        result = self.model(image_pil, mask_pil)
        
        return result
    
    def inpaint_file(self,
                     image_path: Union[str, Path],
                     mask_path: Union[str, Path],
                     output_path: Union[str, Path]) -> Image.Image:
        """
        Inpaint image from files and save result
        
        Args:
            image_path: Path to input image
            mask_path: Path to mask image
            output_path: Path to save output
            
        Returns:
            Inpainted PIL Image
        """
        result = self.inpaint(image_path, mask_path)
        result.save(str(output_path))
        print(f"Saved inpainted image to {output_path}")
        return result
    
    def _to_pil_image(self, 
                      image: Union[np.ndarray, Image.Image, str, Path]) -> Image.Image:
        """Convert various image formats to PIL Image"""
        if isinstance(image, (str, Path)):
            return Image.open(image).convert('RGB')
        elif isinstance(image, np.ndarray):
            # Handle different numpy array formats
            if image.dtype == np.float32 or image.dtype == np.float64:
                image = (image * 255).astype(np.uint8)
            if len(image.shape) == 2:  # Grayscale
                return Image.fromarray(image).convert('RGB')
            elif len(image.shape) == 3:
                if image.shape[2] == 3:  # RGB
                    return Image.fromarray(image, 'RGB')
                elif image.shape[2] == 4:  # RGBA
                    return Image.fromarray(image, 'RGBA').convert('RGB')
        elif isinstance(image, Image.Image):
            return image.convert('RGB')
        
        raise ValueError(f"Unsupported image type: {type(image)}")
    
    def _to_pil_mask(self,
                     mask: Union[np.ndarray, Image.Image, str, Path]) -> Image.Image:
        """Convert various mask formats to PIL Image (L mode)"""
        if isinstance(mask, (str, Path)):
            return Image.open(mask).convert('L')
        elif isinstance(mask, np.ndarray):
            # Ensure binary mask with values 0 or 255
            if mask.dtype == np.bool_:
                mask = mask.astype(np.uint8) * 255
            elif mask.max() <= 1.0:
                mask = (mask * 255).astype(np.uint8)
            
            # Ensure 2D array
            if len(mask.shape) == 3:
                mask = mask[:, :, 0]
            
            return Image.fromarray(mask, 'L')
        elif isinstance(image, Image.Image):
            return mask.convert('L')
        
        raise ValueError(f"Unsupported mask type: {type(mask)}")
    
    def batch_inpaint(self,
                      image_paths: list,
                      mask_paths: list,
                      output_dir: Union[str, Path],
                      save_format: str = 'png') -> list:
        """
        Batch inpaint multiple images
        
        Args:
            image_paths: List of input image paths
            mask_paths: List of mask paths (same order as images)
            output_dir: Directory to save outputs
            save_format: Output format ('png', 'jpg', etc.)
            
        Returns:
            List of output paths
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if len(image_paths) != len(mask_paths):
            raise ValueError("Number of images and masks must match")
        
        output_paths = []
        
        for i, (img_path, mask_path) in enumerate(zip(image_paths, mask_paths)):
            img_path = Path(img_path)
            output_path = output_dir / f"{img_path.stem}_inpainted.{save_format}"
            
            print(f"Processing {i+1}/{len(image_paths)}: {img_path.name}")
            
            try:
                self.inpaint_file(img_path, mask_path, output_path)
                output_paths.append(str(output_path))
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
                continue
        
        print(f"\nCompleted: {len(output_paths)}/{len(image_paths)} images")
        return output_paths


def test_inpainter():
    """Test LaMa inpainter"""
    print("LaMa Inpainter Test")
    print("=" * 50)
    
    try:
        inpainter = LamaInpainter()
        print("✓ Inpainter initialized successfully")
        print(f"✓ Using device: {inpainter.device}")
        print("\nUsage example:")
        print("  inpainter = LamaInpainter()")
        print("  result = inpainter.inpaint('image.jpg', 'mask.png')")
        print("  result.save('output.jpg')")
    except Exception as e:
        print(f"✗ Error: {e}")
        print("\nPlease install dependencies:")
        print("  pip install simple-lama-inpainting")


if __name__ == '__main__':
    test_inpainter()
