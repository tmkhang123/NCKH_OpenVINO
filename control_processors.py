"""
Control Image Processors for ControlNet
Generate control images (canny, pose, depth) from input images
"""

import cv2
import numpy as np
from PIL import Image
from typing import Tuple, Optional
import torch

class ControlImageProcessor:
    """Base class for control image processors"""

    def __init__(self):
        self.name = "base"

    def process(self, image: Image.Image) -> Image.Image:
        """Process image and return control image"""
        raise NotImplementedError

    def preprocess_image(self, image: Image.Image, target_size: int = 512) -> np.ndarray:
        """Convert PIL Image to numpy array and resize"""
        # Resize maintaining aspect ratio
        w, h = image.size
        if w > h:
            new_w = target_size
            new_h = int(h * target_size / w)
        else:
            new_h = target_size
            new_w = int(w * target_size / h)

        image = image.resize((new_w, new_h), Image.Resampling.LANCZOS)

        # Convert to numpy
        img_array = np.array(image)

        # Pad to square
        canvas = np.ones((target_size, target_size, 3), dtype=np.uint8) * 255
        paste_y = (target_size - new_h) // 2
        paste_x = (target_size - new_w) // 2
        canvas[paste_y:paste_y+new_h, paste_x:paste_x+new_w] = img_array

        return canvas

    def postprocess_image(self, img_array: np.ndarray) -> Image.Image:
        """Convert numpy array back to PIL Image"""
        return Image.fromarray(img_array.astype(np.uint8))


class CannyProcessor(ControlImageProcessor):
    """Canny edge detection processor"""

    def __init__(self, low_threshold: int = 100, high_threshold: int = 200):
        super().__init__()
        self.name = "canny"
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold

    def process(self, image: Image.Image, low_threshold: Optional[int] = None,
                high_threshold: Optional[int] = None) -> Image.Image:
        """
        Apply Canny edge detection

        Args:
            image: Input PIL Image
            low_threshold: Low threshold for Canny (default: 100)
            high_threshold: High threshold for Canny (default: 200)

        Returns:
            PIL Image with detected edges
        """
        # Use provided thresholds or defaults
        low = low_threshold if low_threshold is not None else self.low_threshold
        high = high_threshold if high_threshold is not None else self.high_threshold

        # Preprocess
        img_array = self.preprocess_image(image)

        # Convert to grayscale
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)

        # Apply Canny edge detection
        edges = cv2.Canny(gray, low, high)

        # Convert to 3-channel for consistency
        edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)

        return self.postprocess_image(edges_rgb)


class PoseProcessor(ControlImageProcessor):
    """OpenPose-style pose detection processor"""

    def __init__(self):
        super().__init__()
        self.name = "openpose"
        self.detector = None
        self._init_detector()

    def _init_detector(self):
        """Initialize pose detector (MediaPipe)"""
        try:
            import mediapipe as mp
            self.mp_pose = mp.solutions.pose
            self.mp_drawing = mp.solutions.drawing_utils
            self.mp_drawing_styles = mp.solutions.drawing_styles
            self.detector = self.mp_pose.Pose(
                static_image_mode=True,
                model_complexity=2,
                enable_segmentation=False,
                min_detection_confidence=0.5
            )
            print("✓ MediaPipe pose detector initialized")
        except ImportError:
            print("⚠ MediaPipe not installed. Install with: pip install mediapipe")
            self.detector = None

    def process(self, image: Image.Image) -> Image.Image:
        """
        Extract pose skeleton from image

        Args:
            image: Input PIL Image with person

        Returns:
            PIL Image with pose skeleton drawn
        """
        if self.detector is None:
            print("❌ Pose detector not available")
            # Return blank image as fallback
            img_array = self.preprocess_image(image)
            blank = np.zeros_like(img_array)
            return self.postprocess_image(blank)

        # Preprocess
        img_array = self.preprocess_image(image)

        # Convert RGB to BGR for MediaPipe
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

        # Detect pose
        results = self.detector.process(img_bgr)

        # Create blank canvas for pose
        pose_canvas = np.zeros_like(img_array)

        # Draw pose landmarks
        if results.pose_landmarks:
            # Convert back to RGB for drawing
            pose_canvas_bgr = cv2.cvtColor(pose_canvas, cv2.COLOR_RGB2BGR)

            self.mp_drawing.draw_landmarks(
                pose_canvas_bgr,
                results.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style()
            )

            pose_canvas = cv2.cvtColor(pose_canvas_bgr, cv2.COLOR_BGR2RGB)

        return self.postprocess_image(pose_canvas)


class DepthProcessor(ControlImageProcessor):
    """Depth estimation processor"""

    def __init__(self):
        super().__init__()
        self.name = "depth"
        self.model = None
        self._init_model()

    def _init_model(self):
        """Initialize depth estimation model (MiDaS small)"""
        try:
            import torch
            # Use MiDaS small for faster inference
            self.model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
            self.model.eval()

            self.transform = torch.hub.load("intel-isl/MiDaS", "transforms").small_transform

            print("✓ MiDaS depth model initialized")
        except Exception as e:
            print(f"⚠ Failed to load MiDaS: {e}")
            print("  Depth estimation will use simple fallback")
            self.model = None

    def process(self, image: Image.Image) -> Image.Image:
        """
        Estimate depth map from image

        Args:
            image: Input PIL Image

        Returns:
            PIL Image with depth map (grayscale)
        """
        if self.model is None:
            # Fallback: Use simple gradient-based pseudo-depth
            return self._fallback_depth(image)

        # Preprocess
        img_array = self.preprocess_image(image)

        # Convert to torch tensor
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        input_batch = self.transform(img_bgr)

        # Predict depth
        with torch.no_grad():
            prediction = self.model(input_batch)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=img_array.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()

        depth_map = prediction.cpu().numpy()

        # Normalize to 0-255
        depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min()) * 255
        depth_map = depth_map.astype(np.uint8)

        # Convert to RGB for consistency
        depth_rgb = cv2.cvtColor(depth_map, cv2.COLOR_GRAY2RGB)

        return self.postprocess_image(depth_rgb)

    def _fallback_depth(self, image: Image.Image) -> Image.Image:
        """Simple fallback depth estimation using gradients"""
        img_array = self.preprocess_image(image)

        # Convert to grayscale
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)

        # Compute gradients
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

        # Combine gradients
        depth = np.sqrt(grad_x**2 + grad_y**2)

        # Normalize
        depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255
        depth = depth.astype(np.uint8)

        # Invert (closer = brighter)
        depth = 255 - depth

        # Convert to RGB
        depth_rgb = cv2.cvtColor(depth, cv2.COLOR_GRAY2RGB)

        return self.postprocess_image(depth_rgb)


class SegmentationProcessor(ControlImageProcessor):
    """Segmentation processor"""

    def __init__(self):
        super().__init__()
        self.name = "segmentation"

    def process(self, image: Image.Image) -> Image.Image:
        """
        Simple segmentation using color quantization

        Args:
            image: Input PIL Image

        Returns:
            PIL Image with segmented regions
        """
        # Preprocess
        img_array = self.preprocess_image(image)

        # Convert to BGR for OpenCV
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

        # Apply mean-shift segmentation
        shifted = cv2.pyrMeanShiftFiltering(img_bgr, 21, 51)

        # Convert back to RGB
        segmented = cv2.cvtColor(shifted, cv2.COLOR_BGR2RGB)

        return self.postprocess_image(segmented)


class ControlProcessorFactory:
    """Factory to create control processors"""

    @staticmethod
    def create(control_type: str) -> ControlImageProcessor:
        """
        Create control processor by type

        Args:
            control_type: Type of control (canny, pose, depth, seg)

        Returns:
            ControlImageProcessor instance
        """
        processors = {
            "canny": CannyProcessor,
            "pose": PoseProcessor,
            "openpose": PoseProcessor,
            "depth": DepthProcessor,
            "segmentation": SegmentationProcessor,
            "seg": SegmentationProcessor
        }

        processor_class = processors.get(control_type.lower())
        if processor_class is None:
            raise ValueError(f"Unknown control type: {control_type}")

        return processor_class()

    @staticmethod
    def available_types() -> list:
        """Get list of available control types"""
        return ["canny", "pose", "depth", "segmentation"]


# Example usage
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Usage: python control_processors.py <input_image> <control_type>")
        print("Control types: canny, pose, depth, segmentation")
        sys.exit(1)

    input_path = sys.argv[1]
    control_type = sys.argv[2]

    # Load image
    image = Image.open(input_path).convert('RGB')

    # Create processor
    processor = ControlProcessorFactory.create(control_type)

    # Process
    print(f"Processing with {processor.name}...")
    control_image = processor.process(image)

    # Save
    output_path = f"control_{control_type}.png"
    control_image.save(output_path)
    print(f"✓ Saved to {output_path}")
