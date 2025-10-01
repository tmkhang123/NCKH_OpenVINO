import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import openvino as ov
import openvino_genai as ov_genai
from typing import Tuple, Optional, List
import torch
import torchvision.transforms as transforms
from pathlib import Path

class StyleTransferProcessor:
    """Style Transfer module cho Image2Image generation"""
    
    def __init__(self, device: str = "CPU"):
        self.device = device
        self.style_models = {}
        
    def load_style_model(self, style_name: str, model_path: str):
        """Load style transfer model"""
        try:
            # Load OpenVINO model cho style transfer
            core = ov.Core()
            model = core.read_model(model_path)
            compiled_model = core.compile_model(model, device_name=self.device)
            self.style_models[style_name] = compiled_model
            print(f"✅ Loaded style model: {style_name}")
        except Exception as e:
            print(f"❌ Failed to load style model {style_name}: {e}")
    
    def apply_artistic_style(self, image: Image.Image, style: str) -> Image.Image:
        """Apply artistic style to image"""
        
        # Convert to numpy array
        img_array = np.array(image)
        
        if style == "Oil Painting":
            return self._oil_painting_effect(img_array)
        elif style == "Watercolor":
            return self._watercolor_effect(img_array)
        elif style == "Sketch":
            return self._sketch_effect(img_array)
        elif style == "Cartoon":
            return self._cartoon_effect(img_array)
        elif style == "Vintage":
            return self._vintage_effect(img_array)
        else:
            return image
    
    def _oil_painting_effect(self, img_array: np.ndarray) -> Image.Image:
        """Oil painting effect without xphoto"""
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        
        # Apply multiple bilateral filters for oil painting effect
        result = img_bgr.copy()
        for _ in range(3):
            result = cv2.bilateralFilter(result, 9, 100, 100)
        
        # Reduce color palette for more artistic look
        result = cv2.medianBlur(result, 7)
        
        # Convert back to RGB
        result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
        return Image.fromarray(result)
    
    def _watercolor_effect(self, img_array: np.ndarray) -> Image.Image:
        """Watercolor effect with soft edges and bleeding"""
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        
        # Light bilateral filter (không quá smooth)
        result = cv2.bilateralFilter(img_bgr, 5, 50, 50)
        
        # Strong Gaussian blur for watercolor bleeding
        result = cv2.GaussianBlur(result, (7, 7), 0)
        
        # Reduce intensity for lighter watercolor look
        result = cv2.convertScaleAbs(result, alpha=0.85, beta=10)
        
        # Boost saturation for vibrant watercolor
        hsv = cv2.cvtColor(result, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 1] = hsv[:, :, 1] * 1.3  # More saturation
        hsv[:, :, 2] = hsv[:, :, 2] * 1.1  # Slightly brighter
        hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)
        hsv[:, :, 2] = np.clip(hsv[:, :, 2], 0, 255)
        result = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
        
        # Median blur for watercolor texture
        result = cv2.medianBlur(result, 3)
        
        # Convert back to RGB
        result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
        return Image.fromarray(result)
    
    def _sketch_effect(self, img_array: np.ndarray) -> Image.Image:
        """Pencil sketch effect"""
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        
        # Invert the grayscale image
        inverted = 255 - gray
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(inverted, (21, 21), 0)
        
        # Blend the original and blurred images
        sketch = cv2.divide(gray, 255 - blurred, scale=256)
        
        # Convert back to 3-channel
        result = cv2.cvtColor(sketch, cv2.COLOR_GRAY2RGB)
        return Image.fromarray(result)
    
    def _cartoon_effect(self, img_array: np.ndarray) -> Image.Image:
        """Cartoon effect"""
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        
        # Apply bilateral filter
        filtered = cv2.bilateralFilter(img_bgr, 9, 75, 75)
        
        # Create edge mask
        gray = cv2.cvtColor(filtered, cv2.COLOR_BGR2GRAY)
        edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 9)
        
        # Apply edge mask
        edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        result = cv2.bitwise_and(filtered, edges)
        
        # Convert back to RGB
        result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
        return Image.fromarray(result)
    
    def _vintage_effect(self, img_array: np.ndarray) -> Image.Image:
        """Vintage/sepia effect"""
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        
        # Apply sepia tone
        sepia_kernel = np.array([[0.272, 0.534, 0.131],
                                 [0.349, 0.686, 0.168],
                                 [0.393, 0.769, 0.189]])
        
        result = cv2.transform(img_bgr, sepia_kernel)
        result = np.clip(result, 0, 255).astype(np.uint8)
        
        # Reduce brightness slightly for aged look
        result = cv2.convertScaleAbs(result, alpha=0.9, beta=-10)
        
        # Add slight vignette effect
        rows, cols = result.shape[:2]
        kernel_x = cv2.getGaussianKernel(cols, cols/2)
        kernel_y = cv2.getGaussianKernel(rows, rows/2)
        kernel = kernel_y * kernel_x.T
        mask = kernel / kernel.max()
        
        result = result.astype(np.float32)
        for i in range(3):
            result[:, :, i] = result[:, :, i] * mask
        result = np.clip(result, 0, 255).astype(np.uint8)
        
        # Convert back to RGB
        result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
        return Image.fromarray(result)

class ImageRestorationProcessor:
    """Image restoration module"""
    
    def __init__(self, device: str = "CPU"):
        self.device = device
        self.restoration_models = {}
    
    def load_restoration_model(self, model_name: str, model_path: str):
        """Load image restoration model"""
        try:
            core = ov.Core()
            model = core.read_model(model_path)
            compiled_model = core.compile_model(model, device_name=self.device)
            self.restoration_models[model_name] = compiled_model
            print(f"✅ Loaded restoration model: {model_name}")
        except Exception as e:
            print(f"❌ Failed to load restoration model {model_name}: {e}")
    
    def enhance_image(self, image: Image.Image, enhancement_type: str) -> Image.Image:
        """Enhance image quality"""
        
        if enhancement_type == "Super Resolution":
            return self._super_resolution(image)
        elif enhancement_type == "Denoise":
            return self._denoise(image)
        elif enhancement_type == "Deblur":
            return self._deblur(image)
        elif enhancement_type == "Color Correction":
            return self._color_correction(image)
        elif enhancement_type == "Contrast Enhancement":
            return self._contrast_enhancement(image)
        else:
            return image
    
    def _super_resolution(self, image: Image.Image) -> Image.Image:
        """Super resolution using OpenCV"""
        img_array = np.array(image)
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        
        # Use EDSR or ESPCN for super resolution
        # For now, use bicubic interpolation as fallback
        height, width = img_bgr.shape[:2]
        upscaled = cv2.resize(img_bgr, (width*2, height*2), interpolation=cv2.INTER_CUBIC)
        
        # Apply sharpening
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        sharpened = cv2.filter2D(upscaled, -1, kernel)
        
        result = cv2.cvtColor(sharpened, cv2.COLOR_BGR2RGB)
        return Image.fromarray(result)
    
    def _denoise(self, image: Image.Image) -> Image.Image:
        """Denoise image"""
        img_array = np.array(image)
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        
        # Non-local means denoising
        denoised = cv2.fastNlMeansDenoisingColored(img_bgr, None, 10, 10, 7, 21)
        
        result = cv2.cvtColor(denoised, cv2.COLOR_BGR2RGB)
        return Image.fromarray(result)
    
    def _deblur(self, image: Image.Image) -> Image.Image:
        """Deblur image"""
        img_array = np.array(image)
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        
        # Convert to grayscale for deblurring
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        
        # Estimate blur kernel
        kernel = cv2.getGaussianKernel(15, 1.0)
        kernel = np.outer(kernel, kernel)
        
        # Apply deconvolution
        deblurred = cv2.filter2D(gray, -1, kernel)
        
        # Convert back to 3-channel
        result = cv2.cvtColor(deblurred, cv2.COLOR_GRAY2RGB)
        return Image.fromarray(result)
    
    def _color_correction(self, image: Image.Image) -> Image.Image:
        """Color correction"""
        # Convert to LAB color space for better color correction
        img_array = np.array(image)
        lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)
        
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        lab[:,:,0] = clahe.apply(lab[:,:,0])
        
        # Convert back to RGB
        result = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        return Image.fromarray(result)
    
    def _contrast_enhancement(self, image: Image.Image) -> Image.Image:
        """Enhance contrast"""
        # Use PIL for contrast enhancement
        enhancer = ImageEnhance.Contrast(image)
        enhanced = enhancer.enhance(1.5)  # Increase contrast by 50%
        
        # Also enhance sharpness
        sharpness_enhancer = ImageEnhance.Sharpness(enhanced)
        result = sharpness_enhancer.enhance(1.2)
        
        return result

class AdvancedImageProcessor:
    """Combined processor for style transfer and restoration"""
    
    def __init__(self, device: str = "CPU"):
        self.device = device
        self.style_processor = StyleTransferProcessor(device)
        self.restoration_processor = ImageRestorationProcessor(device)
    
    def process_image(self, image: Image.Image, 
                     style: Optional[str] = None,
                     restoration: Optional[str] = None,
                     enhancement: Optional[str] = None) -> Image.Image:
        """Process image with style transfer and/or restoration"""
        
        result = image.copy()
        
        # Apply restoration first
        if restoration:
            result = self.restoration_processor.enhance_image(result, restoration)
        
        # Apply style transfer
        if style:
            result = self.style_processor.apply_artistic_style(result, style)
        
        # Apply additional enhancement
        if enhancement:
            result = self.restoration_processor.enhance_image(result, enhancement)
        
        return result
    
    def get_available_styles(self) -> List[str]:
        """Get available style options"""
        return ["Oil Painting", "Watercolor", "Sketch", "Cartoon", "Vintage"]
    
    def get_available_restorations(self) -> List[str]:
        """Get available restoration options"""
        return ["Super Resolution", "Denoise", "Deblur", "Color Correction", "Contrast Enhancement"]

# Example usage
if __name__ == "__main__":
    processor = AdvancedImageProcessor()
    
    # Load test image
    test_image = Image.open("input/download.png")
    
    # Apply style transfer
    styled_image = processor.process_image(
        test_image, 
        style="Oil Painting",
        restoration="Super Resolution"
    )
    
    styled_image.save("styled_output.png")
    print("✅ Style transfer completed!")

