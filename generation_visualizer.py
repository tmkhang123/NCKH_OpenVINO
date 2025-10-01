import numpy as np
from PIL import Image, ImageDraw, ImageFont
import openvino as ov
import openvino_genai as ov_genai
from typing import List, Tuple, Optional
import matplotlib.pyplot as plt
import cv2
from pathlib import Path
import json
import time

class GenerationVisualizer:
    """Visualize the generation process step by step"""
    
    def __init__(self, device: str = "CPU"):
        self.device = device
        self.intermediate_images = []
        self.latent_steps = []
        self.timesteps = []
        self.noise_levels = []
        
    def create_callback(self, save_intermediates: bool = True, save_frequency: int = 5):
        """Create callback function for generation process"""
        
        def progress_callback(step: int, num_steps: int, latent: np.ndarray):
            """Callback function called during generation"""
            
            if save_intermediates and step % save_frequency == 0:
                # Decode latent to image
                try:
                    # This is a simplified version - in practice you'd need proper VAE decoder
                    intermediate_img = self._decode_latent(latent)
                    self.intermediate_images.append({
                        'step': step,
                        'image': intermediate_img,
                        'latent': latent.copy()
                    })
                    
                    # Calculate noise level
                    noise_level = np.std(latent)
                    self.noise_levels.append(noise_level)
                    self.timesteps.append(step)
                    
                except Exception as e:
                    print(f"Error decoding latent at step {step}: {e}")
            
            return False  # Don't stop generation
        
        return progress_callback
    
    def _decode_latent(self, latent: np.ndarray) -> Image.Image:
        """Decode latent to image (simplified version)"""
        # This is a placeholder - you'd need the actual VAE decoder
        # For now, we'll create a visualization of the latent
        
        # Normalize latent to 0-255 range
        latent_norm = (latent - latent.min()) / (latent.max() - latent.min())
        latent_norm = (latent_norm * 255).astype(np.uint8)
        
        # Reshape to image format
        if len(latent_norm.shape) == 4:  # [1, C, H, W]
            latent_norm = latent_norm[0]  # Remove batch dimension
        
        if len(latent_norm.shape) == 3:  # [C, H, W]
            # Convert to [H, W, C] for PIL
            latent_norm = np.transpose(latent_norm, (1, 2, 0))
        
        # Ensure 3 channels
        if latent_norm.shape[2] == 4:  # RGBA
            latent_norm = latent_norm[:, :, :3]
        elif latent_norm.shape[2] == 1:  # Grayscale
            latent_norm = np.repeat(latent_norm, 3, axis=2)
        
        return Image.fromarray(latent_norm)
    
    def create_generation_gif(self, output_path: str, duration: int = 500):
        """Create GIF showing generation process"""
        
        if not self.intermediate_images:
            print("No intermediate images to create GIF")
            return
        
        # Sort by step
        images = sorted(self.intermediate_images, key=lambda x: x['step'])
        
        # Create GIF
        images_list = [img['image'] for img in images]
        images_list[0].save(
            output_path,
            save_all=True,
            append_images=images_list[1:],
            duration=duration,
            loop=0
        )
        
        print(f"Generation GIF saved to {output_path}")
    
    def create_comparison_grid(self, output_path: str, grid_size: Tuple[int, int] = (4, 3)):
        """Create grid comparison of generation steps"""
        
        if not self.intermediate_images:
            print("No intermediate images to create grid")
            return
        
        # Select images for grid
        num_images = min(len(self.intermediate_images), grid_size[0] * grid_size[1])
        selected_images = self.intermediate_images[:num_images]
        
        # Create grid
        img_width, img_height = selected_images[0]['image'].size
        grid_width = img_width * grid_size[0]
        grid_height = img_height * grid_size[1]
        
        grid_image = Image.new('RGB', (grid_width, grid_height), (255, 255, 255))
        
        for i, img_data in enumerate(selected_images):
            row = i // grid_size[0]
            col = i % grid_size[0]
            
            x = col * img_width
            y = row * img_height
            
            # Add step number to image
            img_with_text = self._add_step_text(img_data['image'], img_data['step'])
            grid_image.paste(img_with_text, (x, y))
        
        grid_image.save(output_path)
        print(f"Comparison grid saved to {output_path}")
    
    def _add_step_text(self, image: Image.Image, step: int) -> Image.Image:
        """Add step number text to image"""
        img_copy = image.copy()
        draw = ImageDraw.Draw(img_copy)
        
        # Try to use a font, fallback to default if not available
        try:
            font = ImageFont.truetype("arial.ttf", 20)
        except:
            font = ImageFont.load_default()
        
        # Add text
        text = f"Step {step}"
        draw.text((10, 10), text, fill=(255, 255, 255), font=font)
        
        return img_copy
    
    def plot_noise_reduction(self, output_path: str):
        """Plot noise reduction over time"""
        
        if not self.noise_levels:
            print("No noise level data to plot")
            return
        
        plt.figure(figsize=(10, 6))
        plt.plot(self.timesteps, self.noise_levels, 'b-', linewidth=2)
        plt.xlabel('Generation Step')
        plt.ylabel('Noise Level (std)')
        plt.title('Noise Reduction During Generation')
        plt.grid(True, alpha=0.3)
        
        # Add annotations
        plt.annotate('Start', xy=(self.timesteps[0], self.noise_levels[0]), 
                    xytext=(10, 10), textcoords='offset points',
                    arrowprops=dict(arrowstyle='->', color='red'))
        plt.annotate('End', xy=(self.timesteps[-1], self.noise_levels[-1]), 
                    xytext=(10, -20), textcoords='offset points',
                    arrowprops=dict(arrowstyle='->', color='green'))
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Noise reduction plot saved to {output_path}")
    
    def create_side_by_side_comparison(self, original_image: Image.Image, 
                                     generated_image: Image.Image, 
                                     output_path: str):
        """Create side-by-side comparison"""
        
        # Resize images to same size
        width = max(original_image.width, generated_image.width)
        height = max(original_image.height, generated_image.height)
        
        original_resized = original_image.resize((width, height), Image.Resampling.LANCZOS)
        generated_resized = generated_image.resize((width, height), Image.Resampling.LANCZOS)
        
        # Create side-by-side image
        comparison = Image.new('RGB', (width * 2, height), (255, 255, 255))
        comparison.paste(original_resized, (0, 0))
        comparison.paste(generated_resized, (width, 0))
        
        # Add labels
        draw = ImageDraw.Draw(comparison)
        try:
            font = ImageFont.truetype("arial.ttf", 24)
        except:
            font = ImageFont.load_default()
        
        draw.text((10, 10), "Original", fill=(0, 0, 0), font=font)
        draw.text((width + 10, 10), "Generated", fill=(0, 0, 0), font=font)
        
        comparison.save(output_path)
        print(f"Side-by-side comparison saved to {output_path}")
    
    def save_generation_report(self, output_path: str, 
                             original_image: Image.Image,
                             generated_image: Image.Image,
                             generation_time: float,
                             parameters: dict):
        """Save comprehensive generation report"""
        
        report = {
            "generation_info": {
                "total_steps": len(self.intermediate_images),
                "generation_time": generation_time,
                "parameters": parameters
            },
            "noise_reduction": {
                "initial_noise": self.noise_levels[0] if self.noise_levels else 0,
                "final_noise": self.noise_levels[-1] if self.noise_levels else 0,
                "noise_reduction": self.noise_levels[0] - self.noise_levels[-1] if self.noise_levels else 0
            },
            "intermediate_steps": [
                {
                    "step": img['step'],
                    "noise_level": self.noise_levels[i] if i < len(self.noise_levels) else 0
                }
                for i, img in enumerate(self.intermediate_images)
            ]
        }
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"Generation report saved to {output_path}")
    
    def clear_data(self):
        """Clear stored data"""
        self.intermediate_images = []
        self.latent_steps = []
        self.timesteps = []
        self.noise_levels = []

class AdvancedGenerationPipeline:
    """Enhanced generation pipeline with visualization"""
    
    def __init__(self, model_path: str, device: str = "CPU"):
        self.model_path = model_path
        self.device = device
        self.pipe = None
        self.visualizer = GenerationVisualizer(device)
        
    def load_model(self):
        """Load OpenVINO model"""
        print("Loading model...")
        self.pipe = ov_genai.Image2ImagePipeline(self.model_path, self.device)
        print("Model loaded successfully")
    
    def generate_with_visualization(self, 
                                  prompt: str,
                                  input_image: Image.Image,
                                  num_steps: int = 30,
                                  strength: float = 0.8,
                                  guidance_scale: float = 7.5,
                                  seed: int = 42,
                                  save_intermediates: bool = True,
                                  output_dir: str = "./generation_output"):
        """Generate image with full visualization"""
        
        if self.pipe is None:
            self.load_model()
        
        # Clear previous data
        self.visualizer.clear_data()
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Preprocess input image
        input_tensor = self._image_to_tensor(input_image)
        
        # Setup callback
        callback = self.visualizer.create_callback(
            save_intermediates=save_intermediates,
            save_frequency=max(1, num_steps // 10)  # Save 10 intermediate steps
        )
        
        # Generate
        start_time = time.time()
        
        result = self.pipe.generate(
            prompt,
            input_tensor,
            num_inference_steps=num_steps,
            strength=strength,
            guidance_scale=guidance_scale,
            seed=seed,
            callback=callback
        )
        
        generation_time = time.time() - start_time
        
        # Convert result to PIL Image
        generated_image = Image.fromarray(result.data[0])
        
        # Save results
        self._save_generation_results(
            output_path,
            input_image,
            generated_image,
            generation_time,
            {
                "prompt": prompt,
                "num_steps": num_steps,
                "strength": strength,
                "guidance_scale": guidance_scale,
                "seed": seed
            }
        )
        
        return generated_image, generation_time
    
    def _image_to_tensor(self, image: Image.Image) -> ov.Tensor:
        """Convert PIL Image to OpenVINO Tensor"""
        image = image.convert("RGB")
        image_array = np.array(image, dtype=np.uint8)
        image_array = image_array.reshape(1, image_array.shape[0], image_array.shape[1], 3)
        return ov.Tensor(image_array)
    
    def _save_generation_results(self, 
                               output_path: Path,
                               original_image: Image.Image,
                               generated_image: Image.Image,
                               generation_time: float,
                               parameters: dict):
        """Save all generation results"""
        
        # Save images
        original_image.save(output_path / "original.png")
        generated_image.save(output_path / "generated.png")
        
        # Create side-by-side comparison
        self.visualizer.create_side_by_side_comparison(
            original_image, generated_image, 
            str(output_path / "comparison.png")
        )
        
        # Create GIF if we have intermediate steps
        if self.visualizer.intermediate_images:
            self.visualizer.create_generation_gif(
                str(output_path / "generation_process.gif")
            )
            
            # Create comparison grid
            self.visualizer.create_comparison_grid(
                str(output_path / "steps_grid.png")
            )
        
        # Create noise reduction plot
        if self.visualizer.noise_levels:
            self.visualizer.plot_noise_reduction(
                str(output_path / "noise_reduction.png")
            )
        
        # Save generation report
        self.visualizer.save_generation_report(
            str(output_path / "generation_report.json"),
            original_image,
            generated_image,
            generation_time,
            parameters
        )
        
        print(f"All generation results saved to {output_path}")

# Example usage
if __name__ == "__main__":
    # Initialize pipeline
    pipeline = AdvancedGenerationPipeline("models/sd15_int8_ov")
    
    # Load test image
    input_image = Image.open("input/download.png")
    
    # Generate with visualization
    generated_image, gen_time = pipeline.generate_with_visualization(
        prompt="a beautiful landscape, high quality, detailed",
        input_image=input_image,
        num_steps=30,
        strength=0.8,
        save_intermediates=True,
        output_dir="./generation_output"
    )
    
    print(f"Generation completed in {gen_time:.2f} seconds")

