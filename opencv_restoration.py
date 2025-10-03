"""
OpenCV Image Restoration Module
Pure OpenCV - NO SD, NO AI models, NO APIs
Fast, reliable, CPU-optimized for Intel hardware
"""

import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
from typing import Tuple

class OpenCVRestoration:
    """Pure OpenCV restoration - Simple, fast, reliable"""

    def __init__(self):
        print("✓ OpenCV Restoration Module initialized (CPU optimized)")

    def restore(self, image: Image.Image, restoration_type: str, strength: float = 0.5) -> Tuple[Image.Image, str]:
        """
        Restore image using OpenCV techniques

        Args:
            image: Input PIL Image
            restoration_type: Type of restoration
            strength: Restoration strength (0.0-1.0)

        Returns:
            (restored_image, status_message)
        """

        if restoration_type == "Old Photo Restoration":
            return self._old_photo_restore(image, strength)
        elif restoration_type == "Blurry Image":
            return self._deblur(image, strength)
        elif restoration_type == "Low Quality":
            return self._enhance_quality(image, strength)
        elif restoration_type == "Noisy Image":
            return self._denoise(image, strength)
        elif restoration_type == "Faded Colors":
            return self._color_correction(image, strength)
        elif restoration_type == "None":
            return image, "No restoration applied"
        else:
            return image, f"Unknown type: {restoration_type}"

    def _old_photo_restore(self, image: Image.Image, strength: float) -> Tuple[Image.Image, str]:
        """Restore old photo: denoise + contrast + color"""

        img_array = np.array(image)

        # 1. Denoise (remove noise/scratches)
        denoised = cv2.fastNlMeansDenoisingColored(img_array, None,
                                                    h=int(10 * strength),
                                                    hColor=int(10 * strength),
                                                    templateWindowSize=7,
                                                    searchWindowSize=21)

        # 2. Enhance contrast
        lab = cv2.cvtColor(denoised, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)

        # CLAHE for contrast
        clahe = cv2.createCLAHE(clipLimit=2.0 * strength, tileGridSize=(8, 8))
        l = clahe.apply(l)

        enhanced = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2RGB)

        # 3. Slight sharpening
        kernel = np.array([[-1, -1, -1],
                          [-1,  9, -1],
                          [-1, -1, -1]]) * (strength * 0.5)
        sharpened = cv2.filter2D(enhanced, -1, kernel)

        result = Image.fromarray(sharpened)
        return result, "Old Photo Restored (OpenCV)"

    def _deblur(self, image: Image.Image, strength: float) -> Tuple[Image.Image, str]:
        """Deblur using sharpening"""

        # Convert to numpy
        img_array = np.array(image)

        # Unsharp mask
        gaussian = cv2.GaussianBlur(img_array, (0, 0), 3)
        sharpened = cv2.addWeighted(img_array, 1 + strength, gaussian, -strength, 0)

        result = Image.fromarray(sharpened)
        return result, "Deblurred (OpenCV)"

    def _enhance_quality(self, image: Image.Image, strength: float) -> Tuple[Image.Image, str]:
        """Enhance overall quality"""

        # Sharpness
        enhancer = ImageEnhance.Sharpness(image)
        enhanced = enhancer.enhance(1 + strength * 0.5)

        # Contrast
        enhancer = ImageEnhance.Contrast(enhanced)
        enhanced = enhancer.enhance(1 + strength * 0.3)

        # Color
        enhancer = ImageEnhance.Color(enhanced)
        enhanced = enhancer.enhance(1 + strength * 0.2)

        return enhanced, "Quality Enhanced (PIL)"

    def _denoise(self, image: Image.Image, strength: float) -> Tuple[Image.Image, str]:
        """Remove noise"""

        img_array = np.array(image)

        # Non-local means denoising
        denoised = cv2.fastNlMeansDenoisingColored(
            img_array, None,
            h=int(10 * strength),
            hColor=int(10 * strength),
            templateWindowSize=7,
            searchWindowSize=21
        )

        result = Image.fromarray(denoised)
        return result, "Denoised (OpenCV)"

    def _color_correction(self, image: Image.Image, strength: float) -> Tuple[Image.Image, str]:
        """Correct faded colors"""

        # Convert to LAB
        img_array = np.array(image)
        lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)

        # Apply CLAHE to L channel
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)

        # Merge and convert back
        enhanced_lab = cv2.merge([l, a, b])
        enhanced_rgb = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2RGB)

        # Boost saturation
        hsv = cv2.cvtColor(enhanced_rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
        hsv[:, :, 1] = hsv[:, :, 1] * (1 + strength * 0.5)  # Saturation boost
        hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)
        enhanced_rgb = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)

        result = Image.fromarray(enhanced_rgb)
        return result, "Colors Corrected (OpenCV)"


# Test
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python opencv_restoration.py <image_path>")
        print("Example: python opencv_restoration.py input/old_photo.jpg")
        exit(1)

    # Load image
    img = Image.open(sys.argv[1]).convert('RGB')
    print(f"Loaded: {img.size}")

    # Create restoration module
    restoration = OpenCVRestoration()

    # Test all types
    types = [
        "Old Photo Restoration",
        "Blurry Image",
        "Low Quality",
        "Noisy Image",
        "Faded Colors"
    ]

    for resto_type in types:
        print(f"\nTesting: {resto_type}")
        result, status = restoration.restore(img, resto_type, strength=0.5)
        output_path = f"test_{resto_type.lower().replace(' ', '_')}.png"
        result.save(output_path)
        print(f"  ✓ {status}")
        print(f"  Saved: {output_path}")

    print("\n✅ All tests completed!")
