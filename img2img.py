import numpy as np
from pathlib import Path
from PIL import Image, ImageEnhance, ImageFilter
import openvino as ov
import openvino_genai as ov_genai
import cv2
from typing import Tuple, Optional
import time

MODEL_DIR = Path("models/sd15_int8_ov")   
DEVICE    = "CPU"

class ImagePreprocessor:
    """Tiền xử lý ảnh để tăng chất lượng đầu vào"""
    
    @staticmethod
    def enhance_image(image: Image.Image) -> Image.Image:
        """Tăng cường chất lượng ảnh"""
        # Tăng độ sắc nét
        enhancer = ImageEnhance.Sharpness(image)
        image = enhancer.enhance(1.2)
        
        # Tăng độ tương phản nhẹ
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(1.1)
        
        return image
    
    @staticmethod
    def resize_smart(image: Image.Image, target_size: Tuple[int, int]) -> Image.Image:
        """Resize thông minh giữ tỷ lệ"""
        w, h = image.size
        target_w, target_h = target_size
        
        # Tính tỷ lệ để giữ nguyên aspect ratio
        ratio = min(target_w/w, target_h/h)
        new_w, new_h = int(w*ratio), int(h*ratio)
        
        # Resize và pad
        image = image.resize((new_w, new_h), Image.Resampling.LANCZOS)
        
        # Tạo canvas mới và paste ảnh vào giữa
        canvas = Image.new('RGB', target_size, (255, 255, 255))
        paste_x = (target_w - new_w) // 2
        paste_y = (target_h - new_h) // 2
        canvas.paste(image, (paste_x, paste_y))
        
        return canvas

def load_image_as_tensor(path: Path, size=(512, 512), enhance=True) -> ov.Tensor:
    """Load và tiền xử lý ảnh thành tensor"""
    img = Image.open(path).convert("RGB")
    
    if enhance:
        img = ImagePreprocessor.enhance_image(img)
    
    img = ImagePreprocessor.resize_smart(img, size)
    arr = np.array(img, dtype=np.uint8)[None, ...]  # [1, H, W, 3]
    return ov.Tensor(arr)

def calculate_image_metrics(original: np.ndarray, generated: np.ndarray) -> dict:
    """Tính toán metrics đánh giá chất lượng ảnh"""
    # MSE (Mean Squared Error)
    mse = np.mean((original - generated) ** 2)
    
    # PSNR (Peak Signal-to-Noise Ratio)
    if mse == 0:
        psnr = float('inf')
    else:
        psnr = 20 * np.log10(255.0 / np.sqrt(mse))
    
    # SSIM sử dụng OpenCV
    def ssim(img1, img2):
        # Chuyển sang grayscale
        gray1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
        
        # Tính mean và variance
        mu1 = cv2.GaussianBlur(gray1.astype(np.float64), (11, 11), 1.5)
        mu2 = cv2.GaussianBlur(gray2.astype(np.float64), (11, 11), 1.5)
        
        mu1_sq = mu1 * mu1
        mu2_sq = mu2 * mu2
        mu1_mu2 = mu1 * mu2
        
        sigma1_sq = cv2.GaussianBlur(gray1.astype(np.float64) * gray1.astype(np.float64), (11, 11), 1.5) - mu1_sq
        sigma2_sq = cv2.GaussianBlur(gray2.astype(np.float64) * gray2.astype(np.float64), (11, 11), 1.5) - mu2_sq
        sigma12 = cv2.GaussianBlur(gray1.astype(np.float64) * gray2.astype(np.float64), (11, 11), 1.5) - mu1_mu2
        
        c1 = (0.01 * 255) ** 2
        c2 = (0.03 * 255) ** 2
        
        ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))
        return ssim_map.mean()
    
    ssim_score = ssim(original, generated)
    
    return {
        "MSE": mse,
        "PSNR": psnr,
        "SSIM": ssim_score
    }

def main():
    print("🚀 Bắt đầu Image-to-Image Generation với OpenVINO")
    
    # Load pipeline
    start_time = time.time()
    pipe = ov_genai.Image2ImagePipeline(str(MODEL_DIR), DEVICE)
    load_time = time.time() - start_time
    print(f"⏱️  Load model: {load_time:.2f}s")
    
    # Load và xử lý ảnh đầu vào
    input_path = Path("input/download.png")
    if not input_path.exists():
        print(f"❌ Không tìm thấy ảnh đầu vào: {input_path}")
        return
    
    init_tensor = load_image_as_tensor(input_path, size=(512, 512), enhance=True)
    original_img = np.array(Image.open(input_path).convert("RGB").resize((512, 512)))
    
    # Cấu hình generation
    prompts = [
        "astronaut in jungle, photorealistic, high quality, detailed",
        "beautiful landscape, sunset, cinematic lighting, 4k",
        "portrait of a person, professional photography, soft lighting"
    ]
    
    for i, prompt in enumerate(prompts):
        print(f"\n🎨 Prompt {i+1}: {prompt}")
        
        def progress_callback(step, num_steps, _latent):
            if step % 5 == 0 or step == num_steps - 1:
                progress = (step + 1) / num_steps * 100
                print(f"   Progress: {progress:.1f}% ({step+1}/{num_steps})")
        
        # Generate
        start_time = time.time()
        out = pipe.generate(
            prompt,
            init_tensor,
            negative_prompt="blurry, low quality, distorted, ugly, bad anatomy",
            num_inference_steps=30,
            strength=0.7,
            guidance_scale=7.5,
            seed=42 + i,
            callback=progress_callback
        )
        gen_time = time.time() - start_time
        
        # Lưu kết quả
        result_img = Image.fromarray(out.data[0])
        output_path = f"result_{i+1}.png"
        result_img.save(output_path)
        
        # Tính metrics
        generated_array = out.data[0]
        metrics = calculate_image_metrics(original_img, generated_array)
        
        print(f"   ⏱️  Generation time: {gen_time:.2f}s")
        print(f"   📊 Metrics - PSNR: {metrics['PSNR']:.2f}dB, SSIM: {metrics['SSIM']:.3f}")
        print(f"   💾 Saved: {output_path}")
    
    print(f"\n✅ Hoàn thành! Tạo ra {len(prompts)} ảnh")

if __name__ == "__main__":
    main()
