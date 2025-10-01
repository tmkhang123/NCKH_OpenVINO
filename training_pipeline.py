import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
from tqdm import tqdm
import argparse
from accelerate import Accelerator
from diffusers import StableDiffusionPipeline, DDPMScheduler
from diffusers.optimization import get_scheduler
import openvino as ov
from peft import LoraConfig, get_peft_model, TaskType
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VietnameseImageDataset(Dataset):
    """Dataset cho Vietnamese content training"""
    
    def __init__(self, data_dir: str, tokenizer, size: int = 512):
        self.data_dir = Path(data_dir)
        self.tokenizer = tokenizer
        self.size = size
        
        # Load metadata
        metadata_path = self.data_dir / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path, 'r', encoding='utf-8') as f:
                self.metadata = json.load(f)
        else:
            self.metadata = self._create_metadata()
        
        # Image transforms
        self.image_transforms = transforms.Compose([
            transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])
    
    def _create_metadata(self) -> List[Dict]:
        """Tạo metadata từ cấu trúc thư mục"""
        metadata = []
        
        # Vietnamese prompt templates
        vietnamese_templates = {
            "landscape": [
                "phong cảnh Việt Nam đẹp, núi non hùng vĩ, thiên nhiên xanh tươi",
                "cảnh quan Việt Nam, đồng quê yên bình, làng quê truyền thống",
                "thiên nhiên Việt Nam, rừng nhiệt đới, sông nước mênh mông"
            ],
            "architecture": [
                "kiến trúc Việt Nam cổ, chùa chiền, đình làng",
                "phố cổ Hội An, đèn lồng đầy màu sắc, kiến trúc cổ",
                "nhà cổ Việt Nam, mái ngói cong, kiến trúc truyền thống"
            ],
            "culture": [
                "văn hóa Việt Nam, áo dài truyền thống, trang phục dân tộc",
                "lễ hội Việt Nam, múa lân, rồng phượng",
                "nghệ thuật Việt Nam, tranh dân gian, gốm sứ"
            ],
            "food": [
                "ẩm thực Việt Nam, phở bò, bánh mì, nem nướng",
                "món ăn Việt Nam, cơm tấm, bún bò Huế",
                "đặc sản Việt Nam, bánh xèo, chả cá Lã Vọng"
            ]
        }
        
        # Tìm tất cả ảnh trong thư mục
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        for img_path in self.data_dir.rglob("*"):
            if img_path.suffix.lower() in image_extensions:
                # Xác định category từ tên thư mục
                category = img_path.parent.name.lower()
                if category in vietnamese_templates:
                    prompt = np.random.choice(vietnamese_templates[category])
                else:
                    prompt = f"ảnh Việt Nam đẹp, {img_path.stem.replace('_', ' ')}"
                
                metadata.append({
                    "image": str(img_path.relative_to(self.data_dir)),
                    "prompt": prompt,
                    "category": category
                })
        
        # Lưu metadata
        with open(self.data_dir / "metadata.json", 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        return metadata
    
    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, idx):
        item = self.metadata[idx]
        
        # Load image
        image_path = self.data_dir / item["image"]
        image = Image.open(image_path).convert("RGB")
        image = self.image_transforms(image)
        
        # Tokenize prompt
        input_ids = self.tokenizer(
            item["prompt"],
            max_length=self.tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        ).input_ids[0]
        
        return {
            "pixel_values": image,
            "input_ids": input_ids,
            "prompt": item["prompt"],
            "category": item.get("category", "general")
        }

class LoRATrainer:
    """LoRA trainer cho Stable Diffusion"""
    
    def __init__(self, model_name: str = "runwayml/stable-diffusion-v1-5", device: str = "cuda"):
        self.model_name = model_name
        self.device = device
        self.pipe = None
        self.unet = None
        self.tokenizer = None
        self.text_encoder = None
        self.vae = None
        
    def setup_models(self):
        """Setup models for training"""
        logger.info("Setting up models...")
        
        # Load pipeline
        self.pipe = StableDiffusionPipeline.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
        )
        
        self.unet = self.pipe.unet
        self.tokenizer = self.pipe.tokenizer
        self.text_encoder = self.pipe.text_encoder
        self.vae = self.pipe.vae
        
        # Freeze models
        self.vae.requires_grad_(False)
        self.text_encoder.requires_grad_(False)
        self.unet.requires_grad_(False)
        
        logger.info("Models setup completed")
    
    def setup_lora(self, rank: int = 16, alpha: int = 32, dropout: float = 0.1):
        """Setup LoRA configuration"""
        logger.info(f"Setting up LoRA with rank={rank}, alpha={alpha}")
        
        lora_config = LoraConfig(
            r=rank,
            lora_alpha=alpha,
            target_modules=["to_k", "to_q", "to_v", "to_out.0"],
            lora_dropout=dropout,
            bias="none",
            task_type=TaskType.DIFFUSION,
        )
        
        self.unet = get_peft_model(self.unet, lora_config)
        self.unet.print_trainable_parameters()
        
        logger.info("LoRA setup completed")
    
    def train(self, 
              data_dir: str,
              output_dir: str,
              num_epochs: int = 100,
              batch_size: int = 1,
              learning_rate: float = 1e-4,
              save_steps: int = 500):
        """Train LoRA model"""
        
        # Setup accelerator
        accelerator = Accelerator(
            gradient_accumulation_steps=1,
            mixed_precision="fp16" if self.device == "cuda" else "no"
        )
        
        # Create dataset
        train_dataset = VietnameseImageDataset(data_dir, self.tokenizer)
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4
        )
        
        # Setup optimizer
        optimizer = optim.AdamW(
            self.unet.parameters(),
            lr=learning_rate,
            betas=(0.9, 0.999),
            weight_decay=1e-2
        )
        
        # Setup scheduler
        lr_scheduler = get_scheduler(
            "constant",
            optimizer=optimizer,
            num_warmup_steps=0,
            num_training_steps=len(train_dataloader) * num_epochs
        )
        
        # Prepare with accelerator
        self.unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            self.unet, optimizer, train_dataloader, lr_scheduler
        )
        
        # Move other models to device
        self.vae.to(accelerator.device)
        self.text_encoder.to(accelerator.device)
        
        # Training loop
        logger.info("Starting training...")
        global_step = 0
        
        for epoch in range(num_epochs):
            self.unet.train()
            train_loss = 0.0
            
            progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
            
            for step, batch in enumerate(progress_bar):
                with accelerator.accumulate(self.unet):
                    # Encode images to latent space
                    latents = self.vae.encode(batch["pixel_values"]).latent_dist.sample()
                    latents = latents * self.vae.config.scaling_factor
                    
                    # Sample noise
                    noise = torch.randn_like(latents)
                    timesteps = torch.randint(
                        0, self.pipe.scheduler.config.num_train_timesteps,
                        (latents.shape[0],), device=latents.device
                    ).long()
                    
                    # Add noise to latents
                    noisy_latents = self.pipe.scheduler.add_noise(latents, noise, timesteps)
                    
                    # Get text embeddings
                    encoder_hidden_states = self.text_encoder(batch["input_ids"])[0]
                    
                    # Predict noise
                    model_pred = self.unet(noisy_latents, timesteps, encoder_hidden_states).sample
                    
                    # Calculate loss
                    loss = nn.functional.mse_loss(model_pred.float(), noise.float(), reduction="mean")
                    
                    # Backward pass
                    accelerator.backward(loss)
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()
                    
                    # Update progress
                    train_loss += loss.item()
                    progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})
                    
                    global_step += 1
                    
                    # Save checkpoint
                    if global_step % save_steps == 0:
                        if accelerator.is_main_process:
                            self._save_checkpoint(output_dir, global_step)
            
            # Log epoch loss
            avg_loss = train_loss / len(train_dataloader)
            logger.info(f"Epoch {epoch+1} - Average Loss: {avg_loss:.4f}")
        
        # Save final model
        if accelerator.is_main_process:
            self._save_final_model(output_dir)
        
        logger.info("Training completed!")
    
    def _save_checkpoint(self, output_dir: str, step: int):
        """Save training checkpoint"""
        checkpoint_dir = Path(output_dir) / f"checkpoint-{step}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Save LoRA weights
        self.unet.save_pretrained(checkpoint_dir)
        
        # Save training config
        config = {
            "model_name": self.model_name,
            "step": step,
            "lora_config": self.unet.peft_config
        }
        
        with open(checkpoint_dir / "training_config.json", 'w') as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"Checkpoint saved at step {step}")
    
    def _save_final_model(self, output_dir: str):
        """Save final trained model"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save LoRA weights
        self.unet.save_pretrained(output_path)
        
        # Save training info
        info = {
            "model_name": self.model_name,
            "training_completed": True,
            "lora_config": self.unet.peft_config
        }
        
        with open(output_path / "model_info.json", 'w') as f:
            json.dump(info, f, indent=2)
        
        logger.info(f"Final model saved to {output_path}")

class OpenVINOConverter:
    """Convert trained LoRA model to OpenVINO format"""
    
    def __init__(self, device: str = "CPU"):
        self.device = device
    
    def convert_lora_to_openvino(self, 
                                base_model_path: str,
                                lora_path: str,
                                output_path: str):
        """Convert LoRA model to OpenVINO format"""
        
        logger.info("Converting LoRA model to OpenVINO...")
        
        try:
            # Load base model
            pipe = StableDiffusionPipeline.from_pretrained(base_model_path)
            
            # Load LoRA weights
            from peft import PeftModel
            pipe.unet = PeftModel.from_pretrained(pipe.unet, lora_path)
            
            # Convert to OpenVINO
            from optimum.intel import OVStableDiffusionPipeline
            
            ov_pipe = OVStableDiffusionPipeline.from_pretrained(
                base_model_path,
                export=True,
                compile=False
            )
            
            # Save OpenVINO model
            ov_pipe.save_pretrained(output_path)
            
            logger.info(f"OpenVINO model saved to {output_path}")
            
        except Exception as e:
            logger.error(f"Conversion failed: {e}")
            raise

def main():
    parser = argparse.ArgumentParser(description="Train LoRA model for Vietnamese content")
    parser.add_argument("--data_dir", type=str, required=True, help="Dataset directory")
    parser.add_argument("--output_dir", type=str, default="./lora_output", help="Output directory")
    parser.add_argument("--num_epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--rank", type=int, default=16, help="LoRA rank")
    parser.add_argument("--alpha", type=int, default=32, help="LoRA alpha")
    parser.add_argument("--convert_to_openvino", action="store_true", help="Convert to OpenVINO after training")
    
    args = parser.parse_args()
    
    # Create trainer
    trainer = LoRATrainer()
    
    # Setup models
    trainer.setup_models()
    
    # Setup LoRA
    trainer.setup_lora(rank=args.rank, alpha=args.alpha)
    
    # Train
    trainer.train(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate
    )
    
    # Convert to OpenVINO if requested
    if args.convert_to_openvino:
        converter = OpenVINOConverter()
        converter.convert_lora_to_openvino(
            base_model_path="runwayml/stable-diffusion-v1-5",
            lora_path=args.output_dir,
            output_path=f"{args.output_dir}_openvino"
        )

if __name__ == "__main__":
    main()

