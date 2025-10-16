# 🎨 NCKH OpenVINO - Image-to-Image Generation

Advanced image-to-image generation using OpenVINO and Stable Diffusion, optimized for CPU inference.

## 🌟 Features

- ✅ **Image-to-Image Generation**: Transform images with AI-powered Stable Diffusion v1.5
- ✅ **Image Restoration**: Restore old, blurry, or noisy images using OpenCV
- ✅ **Style Transfer**: Apply artistic styles instantly (Oil Painting, Watercolor, Cartoon, etc.)
- ✅ **OpenVINO Optimization**: INT8 quantized model for fast CPU inference
- ✅ **Edge-Guided Generation**: Preserve image structure with edge detection
- ✅ **User-Friendly UI**: Clean Gradio interface with real-time preview
- ✅ **Flexible Parameters**: Control quality, strength, guidance, resolution

## 📁 Project Structure

```
NCKH_OpenVINO/
├── app_enhanced.py                 # Main Gradio web application
├── img2img.py                      # Standalone processing script
├── prepare_model.py                # Download models from HuggingFace
├── requirements.txt                # Python dependencies
├──
├── style_transfer_module.py        # OpenCV style transfer
├── edge_guided_generation.py       # Edge-guided generation
├── opencv_restoration.py           # Image restoration module
├── control_processors.py           # Control processors
├──
├── models/                         # Models directory
│   └── sd15_int8_ov/              # Stable Diffusion 1.5 INT8
├──
└── input/                          # Input images folder
```

## 🚀 Quick Start

### 1. **Setup Environment**

```bash
# Clone repository
git clone <your-repo>
cd NCKH_OpenVINO

# Install dependencies
pip install -r requirements.txt

# Download base model
python prepare_model.py
```

### 2. **Run Application**

```bash
# Launch web interface
python app_enhanced.py

# Or run standalone script
python img2img.py
```

### 3. **Usage**

1. **Open browser** at `http://localhost:7860`
2. **Upload an image** in the Input panel
3. **Choose mode**:
   - **Image Generation**: Transform image with text prompt
   - **Image Restoration**: Fix old/blurry/noisy images
4. **Adjust parameters** (steps, strength, guidance)
5. **Click Generate** and wait for results

## ⚙️ Parameters Guide

### **Generation Parameters:**

- **Steps** (5-50): Number of denoising steps. Higher = better quality but slower
  - Quick: 10-15 steps
  - Balanced: 20-30 steps
  - Quality: 40-50 steps

- **Strength** (0.1-1.0): How much to transform the input image
  - 0.3-0.5: Subtle changes, preserve original
  - 0.6-0.7: Moderate transformation
  - 0.8-1.0: Strong changes, creative freedom

- **Guidance Scale** (1.0-20.0): How closely to follow the prompt
  - 5-7: Creative, less strict
  - 10-12: Balanced (recommended)
  - 15-20: Very strict adherence

- **Resolution**: Higher resolution = better detail but slower
  - Fast: 512x512
  - Quality: 768x768 or 1024x1024

### **Style Enhancement:**
- **Photorealistic**: High quality, detailed, professional photography
- **Artistic**: Painterly, vibrant colors, masterpiece
- **Cinematic**: Dramatic lighting, film photography
- **Anime**: Anime/manga style illustration
- **Oil Painting**: Classical art with brushstrokes
- **Watercolor**: Soft artistic watercolor style

## 🔧 Image Restoration Types

The application includes OpenCV-based restoration (no AI model required, fast CPU processing):

- **Old Photo Restoration**: Denoise, enhance contrast, restore colors
- **Blurry Image**: Sharpen and enhance clarity
- **Low Quality**: Overall quality enhancement
- **Noisy Image**: Remove grain and noise
- **Faded Colors**: Restore color vibrancy (for colored images)

## 🛠 Troubleshooting

**Model not found:**
```bash
ls models/
python prepare_model.py  # Re-download if needed
```

**Out of memory:**
- Reduce image resolution (512x512 instead of 1024x1024)
- Decrease number of steps
- Close other applications

**Slow generation:**
- This is normal for CPU inference (INT8 optimized)
- Expected time: 10-30 seconds per image depending on resolution
- Intel CPUs with AVX512 will be faster

## 📈 Performance Tips

### **Speed Optimization:**
- Use lower resolution (512x512) for faster results
- Reduce steps to 15-20 for quick previews
- INT8 quantization already applied for CPU efficiency

### **Quality Optimization:**
- Increase steps to 30-50 for better quality
- Use detailed prompts with specific descriptions
- Add negative prompts to avoid unwanted elements
- Experiment with strength values (0.5-0.7 recommended)

## 🙏 Acknowledgments

- [OpenVINO](https://github.com/openvinotoolkit/openvino) - AI inference optimization toolkit
- [OpenVINO GenAI](https://github.com/openvinotoolkit/openvino.genai) - Generative AI pipelines
- [Stable Diffusion](https://huggingface.co/runwayml/stable-diffusion-v1-5) - Image generation model
- [Gradio](https://github.com/gradio-app/gradio) - Web interface framework

## 📝 License

MIT License - See LICENSE file for details

---

**Research Project**: OpenVINO Image-to-Image Generation
**Technology**: OpenVINO + Stable Diffusion v1.5 + Python

🚀 **Optimized for CPU inference with INT8 quantization**