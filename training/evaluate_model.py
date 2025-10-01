import torch
import numpy as np
from PIL import Image
import argparse
from pathlib import Path
import json
import matplotlib.pyplot as plt
import seaborn as sns
from diffusers import StableDiffusionPipeline
import openvino_genai as ov_genai
from typing import List, Dict, Tuple
import time
from tqdm import tqdm
import cv2

class ModelEvaluator:
    """Class để đánh giá và so sánh models"""
    
    def __init__(self, base_model_path: str, trained_model_path: str = None, openvino_model_path: str = None):
        self.base_model_path = base_model_path
        self.trained_model_path = trained_model_path
        self.openvino_model_path = openvino_model_path
        
        # Load models
        self.base_model = None
        self.trained_model = None
        self.openvino_model = None
        
        self.load_models()
        
        # Test prompts
        self.test_prompts = [
            "phong cảnh Việt Nam đẹp, núi non hùng vĩ",
            "vịnh Hạ Long, đá vôi, thuyền buồm",
            "phố cổ Hội An, đèn lồng đầy màu sắc",
            "người phụ nữ Việt mặc áo dài truyền thống",
            "beautiful Vietnamese landscape, high quality, detailed",
            "traditional Vietnamese architecture, ancient pagoda",
            "Vietnamese woman in ao dai, portrait photography"
        ]
        
        # Evaluation results
        self.results = {
            "base_model": [],
            "trained_model": [],
            "openvino_model": []
        }
    
    def load_models(self):
        """Load các models để so sánh"""
        
        print("🔄 Loading models...")
        
        # Base model
        if self.base_model_path:
            print(f"  📦 Loading base model: {self.base_model_path}")
            self.base_model = StableDiffusionPipeline.from_pretrained(
                self.base_model_path,
                torch_dtype=torch.float16,
                safety_checker=None,
                requires_safety_checker=False
            ).to("cuda" if torch.cuda.is_available() else "cpu")
        
        # Trained model (với LoRA)
        if self.trained_model_path:
            print(f"  🎯 Loading trained model: {self.trained_model_path}")
            self.trained_model = StableDiffusionPipeline.from_pretrained(
                self.base_model_path,
                torch_dtype=torch.float16,
                safety_checker=None,
                requires_safety_checker=False
            ).to("cuda" if torch.cuda.is_available() else "cpu")
            
            # Load LoRA weights
            self.trained_model.unet.load_attn_procs(self.trained_model_path)
        
        # OpenVINO model
        if self.openvino_model_path:
            print(f"  ⚡ Loading OpenVINO model: {self.openvino_model_path}")
            self.openvino_model = ov_genai.Image2ImagePipeline(self.openvino_model_path, "CPU")
    
    def calculate_metrics(self, img1: np.ndarray, img2: np.ndarray) -> Dict[str, float]:
        """Tính toán metrics so sánh giữa 2 ảnh"""
        
        # Ensure same size
        if img1.shape != img2.shape:
            img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
        
        # MSE
        mse = np.mean((img1.astype(np.float32) - img2.astype(np.float32)) ** 2)
        
        # PSNR
        if mse == 0:
            psnr = float('inf')
        else:
            psnr = 20 * np.log10(255.0 / np.sqrt(mse))
        
        # SSIM (simplified version)
        def ssim(img1, img2):
            gray1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
            gray2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
            
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
        
        ssim_score = ssim(img1, img2)
        
        return {
            "MSE": mse,
            "PSNR": psnr,
            "SSIM": ssim_score
        }
    
    def generate_image(self, model, prompt: str, seed: int = 42, model_type: str = "diffusers") -> Tuple[np.ndarray, float]:
        """Generate ảnh từ prompt và tính thời gian"""
        
        start_time = time.time()
        
        if model_type == "diffusers":
            # Diffusers pipeline
            with torch.no_grad():
                result = model(
                    prompt,
                    num_inference_steps=30,
                    guidance_scale=7.5,
                    height=512,
                    width=512,
                    generator=torch.Generator().manual_seed(seed)
                ).images[0]
            
            img_array = np.array(result)
            
        elif model_type == "openvino":
            # OpenVINO pipeline cần input image cho image2image
            # Tạo dummy white image
            dummy_img = Image.new('RGB', (512, 512), (255, 255, 255))
            tensor = ov_genai.Tensor(np.array(dummy_img, dtype=np.uint8)[None, ...])
            
            result = model.generate(
                prompt,
                tensor,
                negative_prompt="blurry, low quality",
                num_inference_steps=30,
                strength=0.9,  # High strength để gần như text-to-image
                guidance_scale=7.5,
                seed=seed
            )
            
            img_array = result.data[0]
        
        generation_time = time.time() - start_time
        
        return img_array, generation_time
    
    def evaluate_all_models(self, output_dir: str = "./evaluation_results"):
        """Đánh giá tất cả models"""
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        print("🧪 Bắt đầu evaluation...")
        
        for i, prompt in enumerate(tqdm(self.test_prompts, desc="Testing prompts")):
            print(f"\n🎨 Prompt {i+1}: {prompt}")
            
            prompt_results = {
                "prompt": prompt,
                "base_model": None,
                "trained_model": None,
                "openvino_model": None
            }
            
            # Test base model
            if self.base_model:
                try:
                    img, gen_time = self.generate_image(self.base_model, prompt, seed=42, model_type="diffusers")
                    img_path = output_path / f"base_model_prompt_{i+1}.png"
                    Image.fromarray(img).save(img_path)
                    
                    prompt_results["base_model"] = {
                        "generation_time": gen_time,
                        "image_path": str(img_path),
                        "image_array": img
                    }
                    print(f"  ✅ Base model: {gen_time:.2f}s")
                    
                except Exception as e:
                    print(f"  ❌ Base model error: {e}")
            
            # Test trained model
            if self.trained_model:
                try:
                    img, gen_time = self.generate_image(self.trained_model, prompt, seed=42, model_type="diffusers")
                    img_path = output_path / f"trained_model_prompt_{i+1}.png"
                    Image.fromarray(img).save(img_path)
                    
                    prompt_results["trained_model"] = {
                        "generation_time": gen_time,
                        "image_path": str(img_path),
                        "image_array": img
                    }
                    print(f"  ✅ Trained model: {gen_time:.2f}s")
                    
                except Exception as e:
                    print(f"  ❌ Trained model error: {e}")
            
            # Test OpenVINO model
            if self.openvino_model:
                try:
                    img, gen_time = self.generate_image(self.openvino_model, prompt, seed=42, model_type="openvino")
                    img_path = output_path / f"openvino_model_prompt_{i+1}.png"
                    Image.fromarray(img).save(img_path)
                    
                    prompt_results["openvino_model"] = {
                        "generation_time": gen_time,
                        "image_path": str(img_path),
                        "image_array": img
                    }
                    print(f"  ✅ OpenVINO model: {gen_time:.2f}s")
                    
                except Exception as e:
                    print(f"  ❌ OpenVINO model error: {e}")
            
            self.results["prompt_results"] = self.results.get("prompt_results", [])
            self.results["prompt_results"].append(prompt_results)
    
    def calculate_comparison_metrics(self):
        """Tính toán metrics so sánh giữa các models"""
        
        print("\n📊 Calculating comparison metrics...")
        
        comparisons = []
        
        for i, prompt_result in enumerate(self.results.get("prompt_results", [])):
            base_img = prompt_result.get("base_model", {}).get("image_array")
            trained_img = prompt_result.get("trained_model", {}).get("image_array")
            openvino_img = prompt_result.get("openvino_model", {}).get("image_array")
            
            comparison = {
                "prompt_index": i,
                "prompt": prompt_result["prompt"]
            }
            
            # So sánh trained vs base
            if base_img is not None and trained_img is not None:
                metrics = self.calculate_metrics(base_img, trained_img)
                comparison["trained_vs_base"] = metrics
                print(f"  Prompt {i+1} - Trained vs Base: PSNR={metrics['PSNR']:.2f}, SSIM={metrics['SSIM']:.3f}")
            
            # So sánh OpenVINO vs trained
            if trained_img is not None and openvino_img is not None:
                metrics = self.calculate_metrics(trained_img, openvino_img)
                comparison["openvino_vs_trained"] = metrics
                print(f"  Prompt {i+1} - OpenVINO vs Trained: PSNR={metrics['PSNR']:.2f}, SSIM={metrics['SSIM']:.3f}")
            
            comparisons.append(comparison)
        
        self.results["comparisons"] = comparisons
    
    def generate_report(self, output_dir: str = "./evaluation_results"):
        """Tạo báo cáo đánh giá"""
        
        output_path = Path(output_dir)
        
        # Tính toán statistics
        self.calculate_comparison_metrics()
        
        # Tạo visualization
        self.create_visualizations(output_path)
        
        # Tạo JSON report
        report = {
            "evaluation_date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "models_evaluated": {
                "base_model": self.base_model_path,
                "trained_model": self.trained_model_path,
                "openvino_model": self.openvino_model_path
            },
            "test_prompts": self.test_prompts,
            "results": self.results
        }
        
        with open(output_path / "evaluation_report.json", 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False, default=str)
        
        # Tạo markdown report
        self.create_markdown_report(output_path, report)
        
        print(f"\n✅ Evaluation completed! Results saved to {output_path}")
    
    def create_visualizations(self, output_path: Path):
        """Tạo các biểu đồ visualization"""
        
        print("📈 Creating visualizations...")
        
        # Generation time comparison
        plt.figure(figsize=(12, 6))
        
        models = []
        times = []
        
        for prompt_result in self.results.get("prompt_results", []):
            for model_name in ["base_model", "trained_model", "openvino_model"]:
                if prompt_result.get(model_name):
                    models.append(model_name)
                    times.append(prompt_result[model_name]["generation_time"])
        
        if models and times:
            plt.subplot(1, 2, 1)
            plt.boxplot([times[i::3] for i in range(3)], labels=["Base", "Trained", "OpenVINO"])
            plt.title("Generation Time Comparison")
            plt.ylabel("Time (seconds)")
        
        # PSNR comparison
        plt.subplot(1, 2, 2)
        psnr_trained_vs_base = []
        psnr_openvino_vs_trained = []
        
        for comp in self.results.get("comparisons", []):
            if "trained_vs_base" in comp:
                psnr_trained_vs_base.append(comp["trained_vs_base"]["PSNR"])
            if "openvino_vs_trained" in comp:
                psnr_openvino_vs_trained.append(comp["openvino_vs_trained"]["PSNR"])
        
        x = np.arange(len(psnr_trained_vs_base))
        width = 0.35
        
        if psnr_trained_vs_base:
            plt.bar(x - width/2, psnr_trained_vs_base, width, label='Trained vs Base', alpha=0.8)
        if psnr_openvino_vs_trained:
            plt.bar(x + width/2, psnr_openvino_vs_trained, width, label='OpenVINO vs Trained', alpha=0.8)
        
        plt.xlabel('Prompt Index')
        plt.ylabel('PSNR (dB)')
        plt.title('Image Quality Comparison (PSNR)')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(output_path / "evaluation_charts.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_markdown_report(self, output_path: Path, report: dict):
        """Tạo markdown report"""
        
        with open(output_path / "README.md", 'w', encoding='utf-8') as f:
            f.write("# Model Evaluation Report\n\n")
            f.write(f"**Date:** {report['evaluation_date']}\n\n")
            
            f.write("## Models Evaluated\n\n")
            for model_type, model_path in report['models_evaluated'].items():
                if model_path:
                    f.write(f"- **{model_type}:** `{model_path}`\n")
            
            f.write("\n## Test Prompts\n\n")
            for i, prompt in enumerate(report['test_prompts'], 1):
                f.write(f"{i}. {prompt}\n")
            
            f.write("\n## Results Summary\n\n")
            
            # Generation time summary
            f.write("### Generation Time Comparison\n\n")
            avg_times = {}
            for prompt_result in report['results'].get("prompt_results", []):
                for model_name in ["base_model", "trained_model", "openvino_model"]:
                    if prompt_result.get(model_name):
                        if model_name not in avg_times:
                            avg_times[model_name] = []
                        avg_times[model_name].append(prompt_result[model_name]["generation_time"])
            
            f.write("| Model | Average Time (s) | Min Time (s) | Max Time (s) |\n")
            f.write("|-------|------------------|--------------|---------------|\n")
            
            for model_name, times in avg_times.items():
                if times:
                    f.write(f"| {model_name} | {np.mean(times):.2f} | {np.min(times):.2f} | {np.max(times):.2f} |\n")
            
            # Quality metrics
            f.write("\n### Image Quality Metrics\n\n")
            
            if report['results'].get("comparisons"):
                f.write("#### Trained Model vs Base Model\n\n")
                f.write("| Prompt | PSNR (dB) | SSIM | MSE |\n")
                f.write("|--------|-----------|------|---------|\n")
                
                for comp in report['results']["comparisons"]:
                    if "trained_vs_base" in comp:
                        metrics = comp["trained_vs_base"]
                        prompt_idx = comp["prompt_index"] + 1
                        f.write(f"| {prompt_idx} | {metrics['PSNR']:.2f} | {metrics['SSIM']:.3f} | {metrics['MSE']:.2f} |\n")
                
                f.write("\n#### OpenVINO Model vs Trained Model\n\n")
                f.write("| Prompt | PSNR (dB) | SSIM | MSE |\n")
                f.write("|--------|-----------|------|---------|\n")
                
                for comp in report['results']["comparisons"]:
                    if "openvino_vs_trained" in comp:
                        metrics = comp["openvino_vs_trained"]
                        prompt_idx = comp["prompt_index"] + 1
                        f.write(f"| {prompt_idx} | {metrics['PSNR']:.2f} | {metrics['SSIM']:.3f} | {metrics['MSE']:.2f} |\n")
            
            f.write("\n## Generated Images\n\n")
            for i, prompt_result in enumerate(report['results'].get("prompt_results", []), 1):
                f.write(f"### Prompt {i}: {prompt_result['prompt']}\n\n")
                
                for model_name in ["base_model", "trained_model", "openvino_model"]:
                    if prompt_result.get(model_name):
                        img_path = Path(prompt_result[model_name]["image_path"]).name
                        f.write(f"**{model_name}:**\n")
                        f.write(f"![{model_name}_prompt_{i}]({img_path})\n\n")

def main():
    parser = argparse.ArgumentParser(description="Evaluate Stable Diffusion models")
    
    parser.add_argument("--base_model", type=str, required=True, help="Base model path/name")
    parser.add_argument("--trained_model", type=str, help="Trained LoRA model path")
    parser.add_argument("--openvino_model", type=str, help="OpenVINO model path")
    parser.add_argument("--output_dir", type=str, default="./evaluation_results", help="Output directory")
    
    args = parser.parse_args()
    
    # Initialize evaluator
    evaluator = ModelEvaluator(
        base_model_path=args.base_model,
        trained_model_path=args.trained_model,
        openvino_model_path=args.openvino_model
    )
    
    # Run evaluation
    evaluator.evaluate_all_models(args.output_dir)
    
    # Generate report
    evaluator.generate_report(args.output_dir)

if __name__ == "__main__":
    main() 