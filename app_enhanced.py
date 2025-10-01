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

# Model configuration
MODEL_PATH = "models/sd15_int8_ov"
MODEL_NAME = "SD 1.5 INT8"
DEVICE = "CPU"

# Global variables
current_model = None
image_processor = AdvancedImageProcessor(DEVICE)

# Vietnamese presets
VIETNAMESE_PRESETS = {
    "None": "",
    "Vietnamese Landscape": "beautiful Vietnamese landscape, majestic mountains, high quality, detailed, photorealistic",
    "Ha Long Bay": "Ha Long Bay, limestone karsts, sailing boats, clear blue water, dramatic lighting, cinematic",
    "Hoi An Ancient Town": "Hoi An ancient town, colorful lanterns, traditional architecture, warm lighting, atmospheric",
    "Traditional Ao Dai": "Vietnamese woman wearing traditional ao dai, elegant, portrait photography, soft lighting",
    "Vietnamese Pagoda": "Vietnamese ancient pagoda, traditional architecture, curved tile roof, peaceful, spiritual atmosphere",
    "Sapa Rice Terraces": "Sapa terraced rice fields, green color, morning mist, golden hour, landscape photography"
}

# Restoration presets
RESTORATION_PROMPTS = {
    "None": "",
    "Old Photo": "old photograph, vintage photo, historical image, restored, enhanced quality",
    "Blurry Image": "blurry image, out of focus photo, sharpened, clear details, enhanced",
    "Low Quality": "low quality image, pixelated photo, upscaled, high resolution, enhanced",
    "Noisy Image": "noisy image, grainy photo, denoised, clean, smooth",
    "Faded Colors": "faded colors, washed out photo, color corrected, vibrant, enhanced"
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

def generate_image(mode, inp_img, prompt, preset, style_enh, restoration_type, 
                  neg_prompt, steps, strength, guidance, width, height, seed, randomize):
    """Main generation function"""
    
    if inp_img is None:
        return None, None, "Please upload an image", None
    
    try:
        pipe = load_model()
        
        if randomize:
            seed = random.randint(0, 2**31 - 1)
        
        # Mode-specific processing
        if mode == "Image Restoration":
            # Use restoration prompt
            resto_prompt = RESTORATION_PROMPTS.get(restoration_type, "")
            enhanced_prompt = f"{prompt}, {resto_prompt}" if prompt and resto_prompt else (resto_prompt or prompt or "restored, enhanced")
        else:  # Image Generation
            if preset != "None":
                prompt = VIETNAMESE_PRESETS.get(preset, prompt)
            enhanced_prompt = enhance_prompt(prompt, style_enh)
        
        img = preprocess_image(inp_img, (int(width), int(height)))
        tensor = image_to_tensor(img)
        
        start_time = time.time()
        out = pipe.generate(
            enhanced_prompt,
            tensor,
            negative_prompt=neg_prompt or "blurry, low quality, distorted, ugly",
            num_inference_steps=int(steps),
            strength=float(strength),
            guidance_scale=float(guidance),
            seed=int(seed)
        )
        
        gen_time = time.time() - start_time
        result_img = Image.fromarray(out.data[0])
        status = f"Completed in {gen_time:.2f}s | Mode: {mode}"
        comparison = create_comparison(inp_img, result_img)
        
        return result_img, int(seed), status, comparison
        
    except Exception as e:
        return None, seed, f"Error: {str(e)}", None

def restyle_image(inp_img, style_type, prompt):
    """Restyle function for Tab 2"""
    if inp_img is None:
        return None, "Please upload an image"
    
    try:
        if style_type == "None":
            return inp_img, "No style selected"
        
        result_img = image_processor.process_image(
            inp_img,
            style=style_type,
            restoration=None,
            enhancement=None
        )
        return result_img, f"Style applied: {style_type}"
    except Exception as e:
        return None, f"Error: {str(e)}"

def update_ui(mode):
    """Update UI based on mode"""
    if mode == "Image Generation":
        return (
            gr.update(visible=True),   # gen_group
            gr.update(visible=False),  # resto_group
            gr.update(value=30),       # steps
            gr.update(value=0.8),      # strength
            gr.update(value=7.5),      # guidance
        )
    else:  # Image Restoration
        return (
            gr.update(visible=False),
            gr.update(visible=True),
            gr.update(value=20),
            gr.update(value=0.4),
            gr.update(value=5.0),
        )

def apply_preset(preset):
    return VIETNAMESE_PRESETS.get(preset, "")

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
                        preset = gr.Dropdown(
                            choices=list(VIETNAMESE_PRESETS.keys()),
                            value="None",
                            label="Vietnamese Presets"
                        )
                        style_enh = gr.Dropdown(
                            choices=["None", "Photorealistic", "Artistic", "Cinematic", "Anime", "Oil Painting", "Watercolor"],
                            value="None",
                            label="Style Enhancement"
                        )
                    
                    # Restoration-specific controls
                    with gr.Group(visible=False) as resto_group:
                        gr.Markdown("**Restoration Type**")
                        restoration_type = gr.Dropdown(
                            choices=list(RESTORATION_PROMPTS.keys()),
                            value="None",
                            label="Select Restoration Type"
                        )
                        gr.Markdown("*Note: Prompt above will be combined with restoration settings*")
                    
                    neg_prompt = gr.Textbox(
                        label="Negative Prompt",
                        value="blurry, low quality, distorted, ugly",
                        lines=2
                    )
                    
                    gr.Markdown("### Parameters")
                    
                    with gr.Row():
                        steps = gr.Slider(5, 50, value=30, step=1, label="Steps")
                        strength = gr.Slider(0.1, 1.0, value=0.8, step=0.05, label="Strength")
                    
                    guidance = gr.Slider(1.0, 20.0, value=7.5, step=0.5, label="Guidance")
                    
                    with gr.Row():
                        width = gr.Slider(256, 1024, value=512, step=64, label="Width")
                        height = gr.Slider(256, 1024, value=512, step=64, label="Height")
                    
                    with gr.Row():
                        seed = gr.Number(value=42, label="Seed", precision=0)
                        randomize = gr.Checkbox(value=False, label="Random Seed")
                    
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
        outputs=[gen_group, resto_group, steps, strength, guidance]
    )
    
    preset.change(fn=apply_preset, inputs=[preset], outputs=[prompt])
    
    btn.click(
        fn=generate_image,
        inputs=[mode, inp, prompt, preset, style_enh, restoration_type,
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
