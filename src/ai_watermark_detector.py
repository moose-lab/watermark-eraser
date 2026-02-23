"""
AI-powered Watermark Detector using Deep Learning Segmentation

This detector uses a pre-trained U-Net model to automatically identify watermarks
in images, regardless of their position, color, or shape. It provides pixel-perfect
segmentation masks for accurate watermark removal.

Based on: https://github.com/Diffusion-Dynamics/watermark-segmentation
"""

import cv2
import numpy as np
import torch
import segmentation_models_pytorch as smp
from pathlib import Path
from typing import Tuple, Optional


class AIWatermarkDetector:
    """
    AI-powered watermark detector using deep learning segmentation.
    
    This detector can identify watermarks of any type (logo, text, pattern)
    at any location in the image, without requiring manual configuration.
    """
    
    def __init__(
        self,
        model_path: str = "/home/ubuntu/models/best_watermark_model_mit_b5_best.pth",
        device: str = "auto",
        threshold: float = 0.5,
        min_area: int = 100
    ):
        """
        Initialize the AI watermark detector.
        
        Args:
            model_path: Path to the pre-trained model weights
            device: Device to run inference on ('auto', 'cuda', 'mps', 'cpu')
            threshold: Probability threshold for mask binarization (0-1)
            min_area: Minimum watermark area in pixels to filter noise
        """
        self.model_path = model_path
        self.threshold = threshold
        self.min_area = min_area
        
        # Determine device
        if device == "auto":
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)
        
        # Load model
        self.model = self._load_model()
        self.model.eval()
        
        print(f"✓ AI Watermark Detector initialized on {self.device}")
    
    def _load_model(self) -> torch.nn.Module:
        """Load the pre-trained segmentation model."""
        # Create model architecture (U-Net with MiT-B5 encoder)
        model = smp.Unet(
            encoder_name="mit_b5",
            encoder_weights=None,  # We'll load our own weights
            in_channels=3,
            classes=1,
            activation=None  # We'll apply sigmoid manually
        )
        
        # Load pre-trained weights
        if Path(self.model_path).exists():
            checkpoint = torch.load(self.model_path, map_location=self.device)
            
            # Handle different checkpoint formats
            if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
            
            # Remove 'model.' prefix if present (from PyTorch Lightning)
            state_dict = {k.replace('model.', ''): v for k, v in state_dict.items()}
            
            # Filter out non-model keys (like 'mean', 'std' used for normalization)
            model_keys = set(model.state_dict().keys())
            filtered_state_dict = {k: v for k, v in state_dict.items() if k in model_keys}
            
            model.load_state_dict(filtered_state_dict)
            print(f"✓ Loaded pre-trained weights from {self.model_path}")
        else:
            raise FileNotFoundError(f"Model weights not found at {self.model_path}")
        
        return model.to(self.device)
    
    def detect(
        self,
        image: np.ndarray,
        return_visualization: bool = False
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Detect watermarks in the image and return a binary mask.
        
        Args:
            image: Input image in BGR format (H, W, 3)
            return_visualization: If True, return visualization image
        
        Returns:
            mask: Binary mask (H, W) where 255 = watermark, 0 = background
            visualization: Optional visualization image showing detected regions
        """
        original_h, original_w = image.shape[:2]
        
        # Preprocess image
        input_tensor = self._preprocess(image)
        
        # Run inference
        with torch.no_grad():
            output = self.model(input_tensor)
            prob_mask = torch.sigmoid(output).squeeze().cpu().numpy()
        
        # Resize back to original size
        prob_mask = cv2.resize(prob_mask, (original_w, original_h))
        
        # Binarize
        binary_mask = (prob_mask > self.threshold).astype(np.uint8) * 255
        
        # Post-process: remove small noise
        binary_mask = self._postprocess(binary_mask)
        
        # Create visualization if requested
        visualization = None
        if return_visualization:
            visualization = self._create_visualization(image, binary_mask)
        
        return binary_mask, visualization
    
    def _preprocess(self, image: np.ndarray) -> torch.Tensor:
        """
        Preprocess image for model input.
        
        Args:
            image: Input image in BGR format
        
        Returns:
            Preprocessed tensor (1, 3, H, W)
        """
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize to model input size (keep aspect ratio)
        # The model was trained on 512x512, but can handle different sizes
        h, w = image_rgb.shape[:2]
        target_size = 512
        
        if max(h, w) > target_size:
            scale = target_size / max(h, w)
            new_h, new_w = int(h * scale), int(w * scale)
            image_rgb = cv2.resize(image_rgb, (new_w, new_h))
        
        # Normalize to [0, 1]
        image_rgb = image_rgb.astype(np.float32) / 255.0
        
        # Convert to tensor (C, H, W)
        tensor = torch.from_numpy(image_rgb).permute(2, 0, 1)
        
        # Add batch dimension (1, C, H, W)
        tensor = tensor.unsqueeze(0)
        
        return tensor.to(self.device)
    
    def _postprocess(self, mask: np.ndarray) -> np.ndarray:
        """
        Post-process the binary mask to remove noise and refine edges.
        
        Args:
            mask: Binary mask (H, W)
        
        Returns:
            Refined binary mask
        """
        # Remove small connected components (noise)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        
        refined_mask = np.zeros_like(mask)
        for i in range(1, num_labels):  # Skip background (label 0)
            area = stats[i, cv2.CC_STAT_AREA]
            if area >= self.min_area:
                refined_mask[labels == i] = 255
        
        # Morphological operations to refine edges
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        
        # Close small holes
        refined_mask = cv2.morphologyEx(refined_mask, cv2.MORPH_CLOSE, kernel)
        
        # Dilate slightly to ensure complete coverage
        kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        refined_mask = cv2.dilate(refined_mask, kernel_dilate, iterations=2)
        
        # Smooth edges
        refined_mask = cv2.GaussianBlur(refined_mask, (5, 5), 0)
        refined_mask = (refined_mask > 127).astype(np.uint8) * 255
        
        return refined_mask
    
    def _create_visualization(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        Create a visualization showing detected watermark regions.
        
        Args:
            image: Original image (BGR)
            mask: Binary mask
        
        Returns:
            Visualization image with watermark highlighted
        """
        # Create colored overlay
        overlay = image.copy()
        overlay[mask > 0] = [0, 0, 255]  # Red for watermark
        
        # Blend with original
        visualization = cv2.addWeighted(image, 0.7, overlay, 0.3, 0)
        
        # Draw contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(visualization, contours, -1, (0, 255, 0), 2)
        
        return visualization
    
    def get_coverage_percentage(self, mask: np.ndarray) -> float:
        """
        Calculate the percentage of image covered by watermark.
        
        Args:
            mask: Binary mask
        
        Returns:
            Coverage percentage (0-100)
        """
        total_pixels = mask.shape[0] * mask.shape[1]
        watermark_pixels = np.sum(mask > 0)
        return (watermark_pixels / total_pixels) * 100


# Convenience function for quick usage
def detect_watermark(
    image: np.ndarray,
    model_path: str = "/home/ubuntu/models/best_watermark_model_mit_b5_best.pth",
    device: str = "auto",
    return_visualization: bool = False
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Convenience function to detect watermark in a single image.
    
    Args:
        image: Input image in BGR format
        model_path: Path to pre-trained model weights
        device: Device to run on ('auto', 'cuda', 'mps', 'cpu')
        return_visualization: Whether to return visualization
    
    Returns:
        mask: Binary mask of watermark regions
        visualization: Optional visualization image
    """
    detector = AIWatermarkDetector(model_path=model_path, device=device)
    return detector.detect(image, return_visualization=return_visualization)


if __name__ == "__main__":
    # Test the detector
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python ai_watermark_detector.py <image_path>")
        sys.exit(1)
    
    image_path = sys.argv[1]
    image = cv2.imread(image_path)
    
    if image is None:
        print(f"Error: Could not load image from {image_path}")
        sys.exit(1)
    
    print(f"Processing {image_path}...")
    
    detector = AIWatermarkDetector()
    mask, visualization = detector.detect(image, return_visualization=True)
    
    coverage = detector.get_coverage_percentage(mask)
    print(f"Watermark coverage: {coverage:.2f}%")
    
    # Save results
    output_dir = Path(image_path).parent
    base_name = Path(image_path).stem
    
    mask_path = output_dir / f"{base_name}_ai_mask.png"
    viz_path = output_dir / f"{base_name}_ai_viz.png"
    
    cv2.imwrite(str(mask_path), mask)
    cv2.imwrite(str(viz_path), visualization)
    
    print(f"✓ Mask saved to: {mask_path}")
    print(f"✓ Visualization saved to: {viz_path}")
