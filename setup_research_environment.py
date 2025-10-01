#!/usr/bin/env python3
"""
Setup script for NCKH OpenVINO GenAI Image2Image Research Environment
"""

import os
import sys
import subprocess
import json
from pathlib import Path
import argparse
import shutil

def run_command(command, description=""):
    """Run command and handle errors"""
    print(f"🔄 {description}")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        print(f"Error output: {e.stderr}")
        return False

def create_directory_structure():
    """Create necessary directories for research"""
    print("📁 Creating directory structure...")
    
    directories = [
        "models",
        "models/sd15_int8_ov",
        "models/custom_trained_ov", 
        "models/sdxl_int8_ov",
        "input",
        "output",
        "output/generation_results",
        "output/training_results",
        "output/evaluation_results",
        "dataset",
        "dataset/processed",
        "dataset/augmented",
        "dataset/vietnamese_content",
        "logs",
        "checkpoints",
        "reports"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"  ✅ Created: {directory}")
    
    print("📁 Directory structure created successfully")

def install_dependencies():
    """Install required dependencies"""
    print("📦 Installing dependencies...")
    
    # Core dependencies
    core_deps = [
        "openvino>=2024.0.0",
        "openvino-genai>=2024.0.0",
        "Pillow>=10.0.0",
        "numpy>=1.24.0",
        "opencv-python>=4.8.0",
        "gradio>=4.0.0",
        "huggingface-hub>=0.17.0",
        "transformers>=4.35.0",
        "diffusers>=0.24.0",
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "accelerate>=0.24.0",
        "xformers>=0.0.22",
        "datasets>=2.14.0",
        "pandas>=2.0.0",
        "lpips>=0.1.4",
        "torchmetrics>=1.2.0",
        "tqdm>=4.65.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "peft>=0.6.0",
        "optimum-intel>=1.15.0"
    ]
    
    for dep in core_deps:
        if not run_command(f"pip install {dep}", f"Installing {dep}"):
            print(f"⚠️  Warning: Failed to install {dep}")
    
    print("📦 Dependencies installation completed")

def download_base_models():
    """Download base models"""
    print("🤖 Downloading base models...")
    
    # Download Stable Diffusion 1.5 INT8
    if not Path("models/sd15_int8_ov").exists():
        print("📥 Downloading Stable Diffusion 1.5 INT8...")
        run_command(
            "python prepare_model.py",
            "Downloading SD 1.5 INT8 model"
        )
    else:
        print("✅ SD 1.5 INT8 model already exists")
    
    print("🤖 Base models download completed")

def create_config_files():
    """Create configuration files"""
    print("⚙️ Creating configuration files...")
    
    # Research configuration
    research_config = {
        "project_name": "NCKH OpenVINO GenAI Image2Image Optimization",
        "version": "1.0.0",
        "device": "CPU",
        "models": {
            "base_model": "runwayml/stable-diffusion-v1-5",
            "openvino_model": "models/sd15_int8_ov",
            "custom_model": "models/custom_trained_ov"
        },
        "training": {
            "batch_size": 1,
            "learning_rate": 1e-4,
            "num_epochs": 100,
            "lora_rank": 16,
            "lora_alpha": 32
        },
        "evaluation": {
            "metrics": ["PSNR", "SSIM", "LPIPS", "FID"],
            "test_images": 100,
            "resolution": 512
        },
        "vietnamese_content": {
            "presets": [
                "Phong cảnh Việt Nam",
                "Vịnh Hạ Long", 
                "Phố cổ Hội An",
                "Áo dài truyền thống",
                "Chùa Việt Nam",
                "Ruộng lúa Sapa"
            ]
        }
    }
    
    with open("research_config.json", "w", encoding="utf-8") as f:
        json.dump(research_config, f, indent=2, ensure_ascii=False)
    
    # Training configuration
    training_config = {
        "data_dir": "dataset/vietnamese_content",
        "output_dir": "output/training_results",
        "num_epochs": 100,
        "batch_size": 1,
        "learning_rate": 1e-4,
        "lora_rank": 16,
        "lora_alpha": 32,
        "save_steps": 500,
        "mixed_precision": "fp16",
        "gradient_accumulation_steps": 1
    }
    
    with open("training_config.json", "w", encoding="utf-8") as f:
        json.dump(training_config, f, indent=2, ensure_ascii=False)
    
    # Evaluation configuration
    evaluation_config = {
        "test_images": 100,
        "metrics": ["PSNR", "SSIM", "LPIPS", "FID"],
        "resolution": 512,
        "batch_size": 1,
        "output_dir": "output/evaluation_results"
    }
    
    with open("evaluation_config.json", "w", encoding="utf-8") as f:
        json.dump(evaluation_config, f, indent=2, ensure_ascii=False)
    
    print("⚙️ Configuration files created successfully")

def create_sample_dataset():
    """Create sample dataset structure"""
    print("📊 Creating sample dataset structure...")
    
    # Create sample metadata
    sample_metadata = [
        {
            "image": "sample_1.jpg",
            "prompt": "phong cảnh Việt Nam đẹp, núi non hùng vĩ, thiên nhiên xanh tươi",
            "category": "landscape"
        },
        {
            "image": "sample_2.jpg", 
            "prompt": "vịnh Hạ Long, đá vôi, thuyền buồm, nước biển xanh trong",
            "category": "landscape"
        },
        {
            "image": "sample_3.jpg",
            "prompt": "phố cổ Hội An, đèn lồng đầy màu sắc, kiến trúc cổ",
            "category": "architecture"
        }
    ]
    
    with open("dataset/vietnamese_content/metadata.json", "w", encoding="utf-8") as f:
        json.dump(sample_metadata, f, indent=2, ensure_ascii=False)
    
    print("📊 Sample dataset structure created")
    print("💡 Add your Vietnamese images to dataset/vietnamese_content/ and update metadata.json")

def create_scripts():
    """Create utility scripts"""
    print("📝 Creating utility scripts...")
    
    # Training script
    training_script = """#!/usr/bin/env python3
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from training_pipeline import main
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train LoRA model")
    parser.add_argument("--data_dir", type=str, default="dataset/vietnamese_content")
    parser.add_argument("--output_dir", type=str, default="output/training_results")
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--alpha", type=int, default=32)
    
    args = parser.parse_args()
    main()
"""
    
    with open("train_model.py", "w") as f:
        f.write(training_script)
    
    # Evaluation script
    evaluation_script = """#!/usr/bin/env python3
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from evaluation_framework import main
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate models")
    parser.add_argument("--model_path", type=str, default="models/sd15_int8_ov")
    parser.add_argument("--test_dir", type=str, default="input")
    parser.add_argument("--output_dir", type=str, default="output/evaluation_results")
    
    args = parser.parse_args()
    main()
"""
    
    with open("evaluate_model.py", "w") as f:
        f.write(evaluation_script)
    
    # Make scripts executable
    os.chmod("train_model.py", 0o755)
    os.chmod("evaluate_model.py", 0o755)
    
    print("📝 Utility scripts created successfully")

def create_documentation():
    """Create documentation files"""
    print("📚 Creating documentation...")
    
    # README for research
    readme_content = """# 🎓 NCKH OpenVINO GenAI Image2Image Research

## 📋 Mục tiêu nghiên cứu
Tối ưu hóa OpenVINO GenAI Image2Image pipeline để cải thiện performance, quality và khả năng xử lý nội dung Việt Nam.

## 🚀 Quick Start

### 1. Setup Environment
```bash
python setup_research_environment.py
```

### 2. Run Enhanced App
```bash
python app_enhanced.py
```

### 3. Train Custom Model
```bash
python train_model.py --data_dir dataset/vietnamese_content --num_epochs 100
```

### 4. Evaluate Models
```bash
python evaluate_model.py --model_path models/sd15_int8_ov
```

## 📊 Research Features

### ✅ Implemented
- **Enhanced Image2Image Generation**: Fixed generation issues
- **Style Transfer**: Oil painting, watercolor, sketch, cartoon, vintage
- **Image Restoration**: Super resolution, denoising, deblurring, color correction
- **Generation Visualization**: See intermediate steps during generation
- **Vietnamese Content Optimization**: Preset prompts and cultural context
- **LoRA Training Pipeline**: Custom model training for Vietnamese content

### 🔄 In Progress
- **Performance Optimization**: Quantization strategies
- **Quality Metrics**: Comprehensive evaluation framework
- **Real-time Inference**: Streaming and mobile optimization

## 📁 Project Structure
```
NCKH_OpenVINO/
├── app_enhanced.py              # Enhanced main application
├── style_transfer_module.py     # Style transfer and restoration
├── generation_visualizer.py     # Generation process visualization
├── training_pipeline.py         # LoRA training pipeline
├── setup_research_environment.py # Environment setup
├── models/                      # Model storage
├── dataset/                     # Training datasets
├── output/                      # Results and outputs
└── logs/                        # Training and evaluation logs
```

## 🎯 Research Questions
1. **Performance**: Làm thế nào để tối ưu hóa tốc độ inference?
2. **Quality**: Có thể cải thiện chất lượng ảnh sinh ra bằng cách nào?
3. **Vietnamese Content**: Làm sao để model hiểu nội dung Việt Nam tốt hơn?

## 📈 Expected Contributions
- Optimized quantization strategy for Image2Image
- Vietnamese-specific LoRA model
- Performance benchmarking framework
- Cultural adaptation methods for AI image generation

## 🔬 Research Methodology
- Quantitative analysis with statistical significance testing
- Qualitative analysis with human evaluation studies
- Comparative studies with A/B testing

## 📚 Next Steps
1. **Week 1-2**: Setup evaluation framework, baseline measurements
2. **Week 3-4**: Quantization experiments, custom quantization
3. **Week 5-8**: LoRA training, Vietnamese content specialization
4. **Week 9-12**: Advanced optimization, real-time deployment

## 🙏 Acknowledgments
- OpenVINO team for optimization tools
- HuggingFace for model ecosystem
- Vietnamese AI community for cultural insights
"""
    
    with open("README_RESEARCH.md", "w", encoding="utf-8") as f:
        f.write(readme_content)
    
    print("📚 Documentation created successfully")

def main():
    """Main setup function"""
    parser = argparse.ArgumentParser(description="Setup NCKH research environment")
    parser.add_argument("--skip-deps", action="store_true", help="Skip dependency installation")
    parser.add_argument("--skip-models", action="store_true", help="Skip model download")
    
    args = parser.parse_args()
    
    print("🎓 Setting up NCKH OpenVINO GenAI Image2Image Research Environment")
    print("=" * 70)
    
    # Create directory structure
    create_directory_structure()
    
    # Install dependencies
    if not args.skip_deps:
        install_dependencies()
    else:
        print("⏭️  Skipping dependency installation")
    
    # Download base models
    if not args.skip_models:
        download_base_models()
    else:
        print("⏭️  Skipping model download")
    
    # Create configuration files
    create_config_files()
    
    # Create sample dataset
    create_sample_dataset()
    
    # Create utility scripts
    create_scripts()
    
    # Create documentation
    create_documentation()
    
    print("\n" + "=" * 70)
    print("🎉 Setup completed successfully!")
    print("\n📋 Next steps:")
    print("1. Add your Vietnamese images to dataset/vietnamese_content/")
    print("2. Update dataset/vietnamese_content/metadata.json")
    print("3. Run: python app_enhanced.py")
    print("4. Start your research!")
    print("\n📚 Documentation: README_RESEARCH.md")
    print("🔧 Configuration: research_config.json")

if __name__ == "__main__":
    main()

