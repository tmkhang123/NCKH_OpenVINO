import random
import numpy as np
from PIL import Image
import gradio as gr
import openvino as ov
import openvino_genai as ov_genai
from pathlib import Path
import time

# Import custom modules
from style_transfer_module import AdvancedImageProcessor
from edge_guided_generation import get_edge_guided_generator
from opencv_restoration import OpenCVRestoration

# Model configuration
MODEL_PATH = "models/sd15_int8_ov"
MODEL_NAME = "SD 1.5 INT8"
DEVICE = "CPU"

# Global variables
current_model = None
image_processor = AdvancedImageProcessor(DEVICE)
opencv_restoration = OpenCVRestoration()

# OpenCV Restoration Types (Pure CV - No AI/API!)
RESTORATION_TYPES = {
    "None": "Không áp dụng",
    "Old Photo Restoration": "Ảnh cũ (denoise + contrast + color)",
    "Blurry Image": "Làm rõ ảnh mờ (sharpening)",
    "Low Quality": "Nâng cao chất lượng tổng thể",
    "Noisy Image": "Giảm nhiễu hạt",
    "Faded Colors": "Làm tươi màu đã có (NOT colorization!)"
}

def load_model() -> ov_genai.Image2ImagePipeline:
    """Load SD 1.5 model"""
    global current_model
    
    if current_model is not None:
        return current_model
    
    model_path = Path(MODEL_PATH)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    print(f"Loading model: {MODEL_NAME}")
    current_model = ov_genai.Image2ImagePipeline(str(model_path), DEVICE)
    return current_model

def enhance_prompt(prompt: str, style: str = "None") -> str:
    """Enhance prompt with style"""
    style_additions = {
        "Photorealistic": ", photorealistic, high quality, detailed, 8k, professional photography",
        "Artistic": ", artistic, painterly, beautiful composition, vibrant colors, masterpiece",
        "Cinematic": ", cinematic lighting, dramatic atmosphere, film photography",
        "Anime": ", anime style, manga style, beautiful anime art, detailed illustration",
        "Oil Painting": ", oil painting style, classical art, rich colors, brushstrokes",
        "Watercolor": ", watercolor painting, soft colors, artistic style"
    }
    
    if style != "None" and style in style_additions:
        prompt += style_additions[style]
    
    return prompt

def preprocess_image(image: Image.Image, target_size: tuple) -> Image.Image:
    """Preprocess input image"""
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    w, h = image.size
    target_w, target_h = target_size
    
    ratio = min(target_w/w, target_h/h)
    new_w, new_h = int(w*ratio), int(h*ratio)
    
    image = image.resize((new_w, new_h), Image.Resampling.LANCZOS)
    
    canvas = Image.new('RGB', target_size, (255, 255, 255))
    paste_x = (target_w - new_w) // 2
    paste_y = (target_h - new_h) // 2
    canvas.paste(image, (paste_x, paste_y))
    
    return canvas

def image_to_tensor(image: Image.Image) -> ov.Tensor:
    """Convert PIL Image to OpenVINO Tensor"""
    image = image.convert("RGB")
    image_array = np.array(image, dtype=np.uint8)
    image_array = image_array.reshape(1, image_array.shape[0], image_array.shape[1], 3)
    return ov.Tensor(image_array)

def create_comparison(original: Image.Image, generated: Image.Image) -> Image.Image:
    """Create side-by-side comparison"""
    width = max(original.width, generated.width)
    height = max(original.height, generated.height)
    
    orig = original.resize((width, height), Image.Resampling.LANCZOS)
    gen = generated.resize((width, height), Image.Resampling.LANCZOS)
    
    comparison = Image.new('RGB', (width * 2, height))
    comparison.paste(orig, (0, 0))
    comparison.paste(gen, (width, 0))
    
    return comparison

def generate_image(mode, inp_img, prompt, style_enh, restoration_type,
                  neg_prompt, steps, strength, guidance, width, height, seed, randomize):
    """Main generation function"""
    
    if inp_img is None:
        return None, None, "Please upload an image", None
    
    try:
        if randomize:
            seed = random.randint(0, 2**31 - 1)

        # Build prompt
        if mode == "Image Restoration":
            resto_prompt = RESTORATION_PROMPTS.get(restoration_type, "")
            enhanced_prompt = f"{prompt}, {resto_prompt}" if prompt and resto_prompt else (resto_prompt or prompt or "restored, enhanced")
        else:
            enhanced_prompt = enhance_prompt(prompt, style_enh)

        start_time = time.time()

        # DIFFERENT APPROACH FOR EACH MODE
        if mode == "Image Restoration":
            # Pure OpenCV Restoration - KHÔNG DÙNG SD!
            print(f"🔧 OpenCV Restoration: {restoration_type}")

            result_img, resto_status = opencv_restoration.restore(
                image=inp_img,
                restoration_type=restoration_type,
                strength=float(strength)
            )

            status = f"✅ {time.time()-start_time:.2f}s | {resto_status}"

        else:
            # Image Generation: Use edge guidance (for better structure)
            generator = get_edge_guided_generator()

            result_img, _edges, _status = generator.generate(
                input_image=inp_img,
                prompt=enhanced_prompt,
                negative_prompt=neg_prompt or "blurry, low quality, distorted, ugly",
                edge_strength=float(strength),
                num_steps=int(steps),
                guidance_scale=float(guidance),
                seed=int(seed) if seed else None
            )
            status = f"✅ {time.time()-start_time:.2f}s | Generation | Edge-Guided"

        comparison = create_comparison(inp_img, result_img)
        return result_img, int(seed) if seed else 42, status, comparison

    except Exception as e:
        return None, seed if seed else 42, f"❌ Error: {str(e)}", None

def restyle_image(inp_img, style_type, prompt):
    """Restyle function - OpenCV style transfer (FAST!)"""
    if inp_img is None:
        return None, "Please upload an image"

    try:
        if style_type == "None":
            return inp_img, "No style selected"

        # Use OpenCV style transfer (INSTANT - like before!)
        result_img = image_processor.process_image(
            inp_img,
            style=style_type,
            restoration=None,
            enhancement=None
        )

        return result_img, f"✅ Style applied: {style_type}"

    except Exception as e:
        return None, f"❌ Error: {str(e)}"

def update_ui(mode):
    """Update UI based on mode - Keep parameters unchanged!"""
    if mode == "Image Generation":
        return (
            gr.update(visible=True),   # gen_group
            gr.update(visible=False),  # resto_group
        )
    else:  # Image Restoration
        return (
            gr.update(visible=False),
            gr.update(visible=True),
        )

# Custom CSS
custom_css = """
.disabled { opacity: 0.4; pointer-events: none; }
.style-btn { 
    margin: 5px; 
    padding: 10px; 
    border: 2px solid #ddd; 
    border-radius: 8px; 
    cursor: pointer;
    transition: all 0.3s;
}
.style-btn:hover { border-color: #667eea; transform: scale(1.05); }
"""

# Gradio Interface
with gr.Blocks(title="OpenVINO Image-to-Image", theme=gr.themes.Soft(), css=custom_css) as demo:
    
    gr.Markdown("# OpenVINO Image-to-Image Generator")
    
    with gr.Tabs():
        # TAB 1: Image Generation & Restoration
        with gr.TabItem("Image Generation"):
            # Mode selector
            mode = gr.Radio(
                choices=["Image Generation", "Image Restoration"],
                value="Image Generation",
                label="Mode"
            )
            
            with gr.Row():
                # Left Column
                with gr.Column(scale=1):
                    inp = gr.Image(type="pil", label="Input Image")
                    
                    gr.Markdown("### Prompt Settings")
                    
                    prompt = gr.Textbox(
                        label="Prompt",
                        value="a beautiful landscape, high quality, detailed",
                        lines=3
                    )
                    
                    # Generation-specific controls
                    with gr.Group(visible=True) as gen_group:
                        style_enh = gr.Dropdown(
                            choices=["None", "Photorealistic", "Artistic", "Cinematic", "Anime", "Oil Painting", "Watercolor"],
                            value="None",
                            label="Style Enhancement"
                        )
                    
                    # Restoration-specific controls
                    with gr.Group(visible=False) as resto_group:
                        gr.Markdown("**🔧 Restoration Type**")
                        restoration_type = gr.Dropdown(
                            choices=list(RESTORATION_TYPES.keys()),
                            value="None",
                            label="Select Restoration Type",
                            info="OpenCV image processing - Fast & reliable"
                        )
                        gr.Markdown("*💡 OpenCV processing: Fast, reliable, CPU-optimized*")
                    
                    neg_prompt = gr.Textbox(
                        label="Negative Prompt",
                        value="blurry, low quality, distorted, ugly",
                        lines=2
                    )
                    
                    gr.Markdown("### Parameters")
                    
                    with gr.Row():
                        steps = gr.Slider(5, 50, value=20, step=1, label="Steps")
                        strength = gr.Slider(0.1, 1.0, value=0.5, step=0.05, label="Strength")

                    guidance = gr.Slider(1.0, 20.0, value=12.0, step=0.5, label="Guidance")

                    with gr.Row():
                        width = gr.Slider(256, 1024, value=512, step=64, label="Width")
                        height = gr.Slider(256, 1024, value=512, step=64, label="Height")

                    with gr.Row():
                        seed = gr.Number(value=42, label="Seed", precision=0)
                        randomize = gr.Checkbox(value=True, label="Random Seed")
                    
                    btn = gr.Button("Generate", variant="primary", size="lg")
                
                # Right Column
                with gr.Column(scale=1):
                    gr.Markdown("### Generated Result")
                    out_img = gr.Image(type="pil", label="Output")
                    
                    used_seed = gr.Number(label="Used Seed", interactive=False)
                    status = gr.Markdown("Ready")
                    
                    gr.Markdown("### Before/After Comparison")
                    comparison = gr.Image(type="pil", label="Comparison")
        
        # TAB 2: Restyle
        with gr.TabItem("Restyle"):
            gr.Markdown("### Transform your image with artistic styles")
            
            with gr.Row():
                with gr.Column(scale=1):
                    restyle_inp = gr.Image(type="pil", label="Input Image")
                    
                    restyle_prompt = gr.Textbox(
                        label="Optional Prompt",
                        value="",
                        lines=2,
                        placeholder="Add optional description..."
                    )
                    
                    gr.Markdown("### Try these styles:")
                    
                    style_type = gr.Radio(
                        choices=["None"] + image_processor.get_available_styles(),
                        value="None",
                        label="Select Style"
                    )
                    
                    restyle_btn = gr.Button("Apply Style", variant="primary", size="lg")
                
                with gr.Column(scale=1):
                    gr.Markdown("### Styled Result")
                    restyle_out = gr.Image(type="pil", label="Output")
                    restyle_status = gr.Markdown("Upload an image and select a style")

    # Event Handlers
    mode.change(
        fn=update_ui,
        inputs=[mode],
        outputs=[gen_group, resto_group]
    )

    btn.click(
        fn=generate_image,
        inputs=[mode, inp, prompt, style_enh, restoration_type,
                neg_prompt, steps, strength, guidance, width, height, seed, randomize],
        outputs=[out_img, used_seed, status, comparison]
    )
    
    restyle_btn.click(
        fn=restyle_image,
        inputs=[restyle_inp, style_type, restyle_prompt],
        outputs=[restyle_out, restyle_status]
    )

    gr.Markdown("""
    ---
    **OpenVINO Image-to-Image** | Optimized for CPU Inference | Powered by Stable Diffusion 1.5
    """)

if __name__ == "__main__":
    demo.queue().launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,
        inbrowser=True
    )
