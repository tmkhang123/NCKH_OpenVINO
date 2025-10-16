"""
Edge-Guided Generation for OpenVINO CPU
100% CPU optimized approach using OpenCV + OpenVINO
Simplified edge-based guidance for Intel hardware
"""

import numpy as np
from PIL import Image
import openvino as ov
import openvino_genai as ov_genai
from control_processors import CannyProcessor
import time
import random

class EdgeGuidedGenerator:
    """
    Edge-guided image generation using OpenCV and OpenVINO
    Optimized for Intel CPU hardware
    """

    def __init__(self, sd_model_path="models/sd15_int8_ov", device="CPU"):
        """
        Initialize edge-guided generator

        Args:
            sd_model_path: Path to OpenVINO SD model
            device: Device (CPU recommended)
        """
        self.sd_model_path = sd_model_path
        self.device = device
        self.pipeline = None
        self.canny_processor = CannyProcessor()

        print(f"✓ EdgeGuidedGenerator initialized")
        print(f"  Device: {device}")
        print(f"  Model: {sd_model_path}")

    def load(self):
        """Load SD pipeline"""
        if self.pipeline is not None:
            return True

        try:
            print("\nLoading Stable Diffusion pipeline...")
            self.pipeline = ov_genai.Image2ImagePipeline(
                self.sd_model_path,
                self.device
            )
            print("✓ Pipeline loaded successfully")
            return True

        except Exception as e:
            print(f"❌ Failed to load pipeline: {e}")
            return False

    def generate(self,
                 input_image,
                 prompt,
                 negative_prompt="",
                 edge_strength=0.7,
                 low_threshold=100,
                 high_threshold=200,
                 num_steps=20,
                 guidance_scale=7.5,
                 seed=None):
        """
        Generate image with edge guidance

        Args:
            input_image: PIL Image input
            prompt: Text prompt
            negative_prompt: Negative prompt
            edge_strength: Control strength (0.3-0.9 recommended)
            low_threshold: Canny low threshold
            high_threshold: Canny high threshold
            num_steps: Inference steps
            guidance_scale: CFG scale
            seed: Random seed (None for random)

        Returns:
            (generated_image, edge_image, status_message)
        """

        # Load if needed
        if self.pipeline is None:
            success = self.load()
            if not success:
                return None, None, "❌ Failed to load pipeline"

        try:
            print(f"\n{'='*60}")
            print("Edge-Guided Generation")
            print(f"{'='*60}")

            # Step 1: Detect edges
            print(f"\n[1/4] Detecting edges...")
            print(f"  Thresholds: low={low_threshold}, high={high_threshold}")

            edge_image = self.canny_processor.process(
                input_image,
                low_threshold=int(low_threshold),
                high_threshold=int(high_threshold)
            )
            print("  ✓ Edges detected")

            # Step 2: Prepare guidance image
            print(f"\n[2/4] Creating guidance image...")
            print(f"  Edge strength: {edge_strength}")

            # Resize both to 512x512
            input_resized = input_image.resize((512, 512), Image.Resampling.LANCZOS)
            edge_resized = edge_image.resize((512, 512), Image.Resampling.LANCZOS)

            # Convert to numpy
            input_array = np.array(input_resized, dtype=np.float32)
            edge_array = np.array(edge_resized, dtype=np.float32)

            # Blend: more strength = more edge influence
            guide_array = (
                input_array * (1 - edge_strength) +
                edge_array * edge_strength
            ).astype(np.uint8)

            guide_image = Image.fromarray(guide_array)
            print("  ✓ Guidance image ready")

            # Step 3: Prepare tensor
            print(f"\n[3/4] Converting to OpenVINO tensor...")

            guide_tensor_array = np.array(guide_image, dtype=np.uint8)
            guide_tensor_array = guide_tensor_array.reshape(
                1,
                guide_tensor_array.shape[0],
                guide_tensor_array.shape[1],
                3
            )
            guide_tensor = ov.Tensor(guide_tensor_array)
            print("  ✓ Tensor prepared")

            # Step 4: Generate with SD
            print(f"\n[4/4] Generating with Stable Diffusion...")
            print(f"  Prompt: {prompt}")
            print(f"  Steps: {num_steps}")
            print(f"  Guidance scale: {guidance_scale}")
            print(f"  Seed: {seed if seed else 'random'}")

            # Use prompt as-is (already enhanced in app if needed)
            start_time = time.time()

            output = self.pipeline.generate(
                prompt,
                guide_tensor,
                negative_prompt=negative_prompt or "blurry, low quality, distorted, messy, chaotic, ugly",
                num_inference_steps=int(num_steps),
                strength=float(edge_strength),
                guidance_scale=float(guidance_scale),
                seed=int(seed) if seed else random.randint(0, 2**31-1)
            )

            gen_time = time.time() - start_time

            # Convert output
            result_image = Image.fromarray(output.data[0])

            status = f"✅ Generated in {gen_time:.2f}s | Edge-Guided (OpenVINO CPU)"

            print(f"\n  {status}")
            print(f"{'='*60}\n")

            return result_image, edge_resized, status

        except Exception as e:
            error_msg = f"❌ Generation failed: {str(e)}"
            print(f"\n{error_msg}")
            print(f"{'='*60}\n")
            return None, None, error_msg


# Global instance (singleton pattern)
_generator_instance = None
_current_model_path = None

def get_edge_guided_generator(model_path: str = "models/sd15_int8_ov"):
    """Get or create generator instance"""
    global _generator_instance, _current_model_path

    # Create new instance if model path changed
    if _generator_instance is None or _current_model_path != model_path:
        _generator_instance = EdgeGuidedGenerator(sd_model_path=model_path)
        _current_model_path = model_path

    return _generator_instance


# Test function
if __name__ == "__main__":
    print("Testing Edge-Guided Generation...")

    # Create generator
    generator = EdgeGuidedGenerator()

    # Load test image
    try:
        test_image = Image.open("input/download.png").convert('RGB')
        print(f"✓ Loaded test image: {test_image.size}")

        # Generate
        result, edges, status = generator.generate(
            input_image=test_image,
            prompt="beautiful watercolor painting",
            edge_strength=0.7,
            num_steps=20
        )

        if result:
            result.save("test_edge_guided.png")
            edges.save("test_edges.png")
            print("✓ Saved results!")

    except Exception as e:
        print(f"❌ Test failed: {e}")
        print("Make sure you have an image at input/download.png")
