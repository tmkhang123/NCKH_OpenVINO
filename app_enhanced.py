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
AVAILABLE_MODELS = {
    "SD 1.5 (Base)": "models/sd15_int8_ov",
    "SD 1.5 + Vietnamese LoRA": "models/vietnamese_ov",
}

DEFAULT_MODEL = "SD 1.5 (Base)"
DEVICE = "CPU"

# Global variables
current_model = None
current_model_name = DEFAULT_MODEL
image_processor = AdvancedImageProcessor(DEVICE)
opencv_restoration = OpenCVRestoration()

# OpenCV Restoration Types (Pure CV - No AI/API!)
RESTORATION_TYPES = {
    "None": "No restoration applied",
    "Old Photo Restoration": "Restore old photos (denoise + contrast + color)",
    "Blurry Image": "Sharpen blurry images",
    "Low Quality": "Overall quality enhancement",
    "Noisy Image": "Remove image noise",
    "Faded Colors": "Restore color vibrancy"
}

# Restoration prompts (not used for OpenCV restoration, only for UI)
RESTORATION_PROMPTS = {
    "Old Photo Restoration": "restored, enhanced, high quality, clear",
    "Blurry Image": "sharp, clear, detailed, high quality",
    "Low Quality": "high quality, detailed, enhanced, professional",
    "Noisy Image": "clean, smooth, high quality, detailed",
    "Faded Colors": "vibrant colors, enhanced, colorful, vivid"
}

def load_model(model_name: str = None) -> ov_genai.Image2ImagePipeline:
    """Load SD model (base or LoRA)"""
    global current_model, current_model_name

    # Use default if not specified
    if model_name is None:
        model_name = DEFAULT_MODEL

    # Return cached model if same
    if current_model is not None and current_model_name == model_name:
        return current_model

    # Get model path
    if model_name not in AVAILABLE_MODELS:
        raise ValueError(f"Model not found: {model_name}")

    model_path = Path(AVAILABLE_MODELS[model_name])

    # Check if model exists
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found: {model_path}\n"
            f"If this is a LoRA model, please convert it first using:\n"
            f"  python convert_lora_to_ov.py"
        )

    print(f"Loading model: {model_name}")
    print(f"  Path: {model_path}")

    # Load new model
    current_model = ov_genai.Image2ImagePipeline(str(model_path), DEVICE)
    current_model_name = model_name

    print(f"  ✓ Model loaded successfully")
    return current_model

def get_model_status() -> str:
    """Get current model status"""
    if current_model is None:
        return "Model will load on first generation"

    # Show LoRA info if using LoRA
    if "LoRA" in current_model_name:
        return f"Active: {current_model_name}\n(Fine-tuned on Vietnamese dataset)"
    return f"Active: {current_model_name}"

def switch_model(model_name: str) -> str:
    """Switch to a different model"""
    global current_model, current_model_name

    try:
        # Check if model exists
        model_path = Path(AVAILABLE_MODELS[model_name])

        if not model_path.exists():
            return f"Model not found: {model_path}\n\nConvert LoRA first: python convert_lora_to_ov.py"

        # Clear current model
        current_model = None

        # Load new model (will be loaded on next generate)
        current_model_name = model_name

        # Show info
        info = f"Ready: {model_name}\n"
        if "LoRA" in model_name:
            info += "(Fine-tuned on Vietnamese dataset)"
        else:
            info += "(Base Stable Diffusion 1.5)"

        return info

    except Exception as e:
        return f"Error: {str(e)}"

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

def generate_image(inp_img, prompt, style_enh, neg_prompt, steps, strength, guidance, width, height, seed, randomize):
    """Image generation function"""

    if inp_img is None:
        return None, None, "Please upload an image", None

    try:
        if randomize:
            seed = random.randint(0, 2**31 - 1)

        enhanced_prompt = enhance_prompt(prompt, style_enh)
        start_time = time.time()

        # Get current model path
        model_path = AVAILABLE_MODELS[current_model_name]

        # Image Generation: Use edge guidance with selected model
        generator = get_edge_guided_generator(model_path=model_path)

        result_img, _edges, _status = generator.generate(
            input_image=inp_img,
            prompt=enhanced_prompt or "high quality, detailed",
            negative_prompt=neg_prompt or "blurry, low quality, distorted, ugly",
            edge_strength=float(strength),
            num_steps=int(steps),
            guidance_scale=float(guidance),
            seed=int(seed) if seed else None
        )
        status = f"Success: {time.time()-start_time:.2f}s | {current_model_name} | Generation Complete"

        comparison = create_comparison(inp_img, result_img)
        return result_img, int(seed) if seed else 42, status, comparison

    except Exception as e:
        return None, seed if seed else 42, f"Error: {str(e)}", None

def restore_image(inp_img, restoration_type, strength):
    """Image restoration function (OpenCV-based)"""

    if inp_img is None:
        return None, "Please upload an image", None

    try:
        start_time = time.time()

        result_img, resto_status = opencv_restoration.restore(
            image=inp_img,
            restoration_type=restoration_type,
            strength=float(strength)
        )

        status = f"Success: {time.time()-start_time:.2f}s | {resto_status}"
        comparison = create_comparison(inp_img, result_img)

        return result_img, status, comparison

    except Exception as e:
        return None, f"Error: {str(e)}", None

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

        return result_img, f"Success: Style applied - {style_type}"

    except Exception as e:
        return None, f"Error: {str(e)}"

def apply_preset(preset_name):
    """Apply quick preset configurations"""
    presets = {
        "Fast": {
            "steps": 10,
            "strength": 0.4,
            "guidance": 10.0,
            "width": 512,
            "height": 512
        },
        "Balanced": {
            "steps": 20,
            "strength": 0.5,
            "guidance": 12.0,
            "width": 512,
            "height": 512
        },
        "Quality": {
            "steps": 40,
            "strength": 0.7,
            "guidance": 15.0,
            "width": 512,
            "height": 512
        }
    }

    if preset_name in presets:
        p = presets[preset_name]
        return p["steps"], p["strength"], p["guidance"], p["width"], p["height"]
    return 20, 0.5, 12.0, 512, 512  # default fallback

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

/* Header styling */
.header-box {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 30px;
    border-radius: 15px;
    margin-bottom: 20px;
    text-align: center;
    color: white;
}

.header-title {
    font-size: 2.5em;
    font-weight: bold;
    margin-bottom: 10px;
}

.header-subtitle {
    font-size: 1.2em;
    opacity: 0.95;
}

/* Info boxes */
.info-box {
    background: #f8f9fa;
    border-left: 4px solid #667eea;
    padding: 15px;
    border-radius: 8px;
    margin: 10px 0;
}

/* Feature cards */
.feature-card {
    background: white;
    border: 2px solid #e0e0e0;
    border-radius: 12px;
    padding: 20px;
    margin: 10px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    transition: transform 0.3s;
}

.feature-card:hover {
    transform: translateY(-5px);
    border-color: #667eea;
    box-shadow: 0 4px 12px rgba(102,126,234,0.2);
}
"""

# Gradio Interface
with gr.Blocks(title="OpenVINO Image-to-Image", theme=gr.themes.Soft(), css=custom_css) as demo:

    # Header
    gr.HTML("""
    <div class="header-box">
        <div class="header-title">OpenVINO GenAI Image to Image</div>
    </div>
    """)

    # Model Selector (Global - affects all tabs)
    with gr.Row():
        with gr.Column(scale=3):
            model_selector = gr.Dropdown(
                choices=list(AVAILABLE_MODELS.keys()),
                value=DEFAULT_MODEL,
                label="Select Model",
                interactive=True,
                info="Base model or Vietnamese fine-tuned LoRA"
            )
        with gr.Column(scale=1):
            model_status = gr.Textbox(
                value=get_model_status(),
                label="Model Status",
                interactive=False
            )

    gr.Markdown("---")

    with gr.Tabs():
        # TAB 1: Image Generation
        with gr.TabItem("Image Generation"):
            with gr.Row():
                # Left Column
                with gr.Column(scale=1):
                    inp = gr.Image(type="pil", label="Input Image")

                    gr.Markdown("### Prompt Settings")

                    prompt = gr.Textbox(
                        label="Prompt",
                        value="",
                        lines=3,
                        placeholder="Describe what you want to generate..."
                    )

                    style_enh = gr.Dropdown(
                        choices=["None", "Photorealistic", "Artistic", "Cinematic", "Anime", "Oil Painting", "Watercolor"],
                        value="None",
                        label="Prompt Keywords",
                        info="Add style keywords to your prompt automatically"
                    )

                    neg_prompt = gr.Textbox(
                        label="Negative Prompt",
                        value="blurry, low quality, distorted, ugly",
                        lines=2
                    )

                    gr.Markdown("### Parameters")

                    # Quick Presets
                    gr.Markdown("**Quick Presets:**")
                    with gr.Row():
                        preset_fast = gr.Button("Fast", size="sm")
                        preset_balanced = gr.Button("Balanced", size="sm")
                        preset_quality = gr.Button("Quality", size="sm")

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

        # TAB 2: Image Restoration
        with gr.TabItem("Image Restoration"):
            with gr.Row():
                # Left Column
                with gr.Column(scale=1):
                    resto_inp = gr.Image(type="pil", label="Input Image")

                    gr.Markdown("### Restoration Settings")

                    restoration_type = gr.Dropdown(
                        choices=list(RESTORATION_TYPES.keys()),
                        value="Old Photo Restoration",
                        label="Restoration Type"
                    )

                    resto_strength = gr.Slider(0.1, 1.0, value=0.5, step=0.05, label="Strength")

                    resto_btn = gr.Button("Restore Image", variant="primary", size="lg")

                    # Restoration Guide
                    gr.Markdown("""
                    ---
                    **Hướng dẫn Restoration:**

                    **Restoration Types:**
                    - **Old Photo Restoration**: Khôi phục ảnh cũ (giảm nhiễu + tăng contrast + cải thiện màu)
                    - **Blurry Image**: Làm rõ ảnh bị mờ (sharpening)
                    - **Low Quality**: Nâng cao chất lượng tổng thể
                    - **Noisy Image**: Giảm nhiễu hạt trên ảnh
                    - **Faded Colors**: Làm tươi lại màu sắc đã phai

                    **Strength (0.1-1.0)**: Mức độ áp dụng restoration
                    - 0.3-0.5: Nhẹ, giữ nguyên ảnh gốc
                    - 0.6-0.7: Vừa phải (khuyên dùng)
                    - 0.8-1.0: Mạnh, thay đổi nhiều

                    *Xử lý bằng OpenCV - Nhanh, không cần AI*
                    """)

                # Right Column
                with gr.Column(scale=1):
                    gr.Markdown("### Restored Result")
                    resto_out = gr.Image(type="pil", label="Output")

                    resto_status = gr.Markdown("Ready")

                    gr.Markdown("### Before/After Comparison")
                    resto_comparison = gr.Image(type="pil", label="Comparison")
        
        # TAB 3: Restyle
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

                    # Restyle Guide
                    gr.Markdown("""
                    ---
                    **Hướng dẫn Restyle:**

                    **Style Types:**
                    - **Oil Painting**: Phong cách sơn dầu cổ điển
                    - **Watercolor**: Phong cách màu nước nhẹ nhàng
                    - **Cartoon**: Phong cách hoạt hình
                    - **Pencil Sketch**: Phác họa bút chì
                    - **HDR**: Tăng cường độ tương phản và màu sắc
                    - **Vintage**: Phong cách cổ điển, retro

                    *Áp dụng ngay lập tức bằng OpenCV*
                    """)

                with gr.Column(scale=1):
                    gr.Markdown("### Styled Result")
                    restyle_out = gr.Image(type="pil", label="Output")
                    restyle_status = gr.Markdown("Upload an image and select a style")

    # Event Handlers

    # Image Generation Tab
    btn.click(
        fn=generate_image,
        inputs=[inp, prompt, style_enh, neg_prompt, steps, strength, guidance, width, height, seed, randomize],
        outputs=[out_img, used_seed, status, comparison]
    )

    # Image Restoration Tab
    resto_btn.click(
        fn=restore_image,
        inputs=[resto_inp, restoration_type, resto_strength],
        outputs=[resto_out, resto_status, resto_comparison]
    )

    # Restyle Tab
    restyle_btn.click(
        fn=restyle_image,
        inputs=[restyle_inp, style_type, restyle_prompt],
        outputs=[restyle_out, restyle_status]
    )

    # Quick Preset Event Handlers
    preset_fast.click(
        fn=lambda: apply_preset("Fast"),
        inputs=[],
        outputs=[steps, strength, guidance, width, height]
    )
    preset_balanced.click(
        fn=lambda: apply_preset("Balanced"),
        inputs=[],
        outputs=[steps, strength, guidance, width, height]
    )
    preset_quality.click(
        fn=lambda: apply_preset("Quality"),
        inputs=[],
        outputs=[steps, strength, guidance, width, height]
    )

    # Model Selector Event Handler
    model_selector.change(
        fn=switch_model,
        inputs=[model_selector],
        outputs=[model_status]
    )

    # Vietnamese Parameter Guide
    gr.Markdown("""
    ---
    ### Hướng dẫn Parameters

    **Quick Presets:**
    - **Fast**: Tạo nhanh cho xem trước (10 steps, ~10-15 giây)
    - **Balanced**: Cài đặt cân bằng (20 steps, strength 0.5, ~20-30 giây) - Khuyên dùng
    - **Quality**: Chất lượng cao nhất (40 steps, ~45-60 giây)

    **Giải thích Parameters:**

    - **Steps (5-50)**: Số bước xử lý. Càng cao = chất lượng tốt hơn nhưng chậm hơn
      - 10-15: Nhanh, xem trước
      - 20-25: Cân bằng (khuyên dùng)
      - 40-50: Chất lượng cao

    - **Strength (0.1-1.0)**: Mức độ thay đổi ảnh gốc
      - 0.3-0.5: Giữ nguyên nhiều, thay đổi nhẹ
      - 0.6-0.7: Thay đổi vừa phải (khuyên dùng)
      - 0.8-1.0: Thay đổi mạnh, sáng tạo cao

    - **Guidance (1.0-20.0)**: Mức độ tuân theo prompt
      - 5-7: Tự do sáng tạo
      - 10-12: Cân bằng (khuyên dùng)
      - 15-20: Tuân thủ prompt nghiêm ngặt

    - **Width/Height**: Độ phân giải output
      - 512x512: Nhanh, chất lượng tốt
      - 768x768 trở lên: Chậm hơn, chi tiết hơn

    **Tips:**
    - Dùng preset Balanced làm điểm khởi đầu
    - Negative prompt giúp tránh các yếu tố không mong muốn
    - **Prompt Keywords**: Chọn template tự động thêm từ khóa vào prompt
      - Photorealistic: Thêm "photorealistic, 8k, detailed..."
      - Anime: Thêm "anime style, manga style..."
      - None: Không thêm gì, dùng prompt gốc

    ---
    **OpenVINO Image-to-Image** | CPU Optimized | Stable Diffusion 1.5
    """)

if __name__ == "__main__":
    demo.queue(max_size=10).launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,
        inbrowser=True,
        max_threads=10
    )
