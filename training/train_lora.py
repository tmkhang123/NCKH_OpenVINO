import os
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionPipeline, UNet2DConditionModel
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
from diffusers.utils import check_min_version, is_wandb_available
from diffusers.utils.import_utils import is_xformers_available
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from transformers import CLIPTextModel, CLIPTokenizer
from tqdm.auto import tqdm
import argparse
from pathlib import Path
import json
import numpy as np
from typing import Dict, Any, Optional

# Check diffusers version
check_min_version("0.24.0.dev0")

logger = get_logger(__name__, log_level="INFO")

class ImageTextDataset(Dataset):
    """Dataset cho training image-to-image với text prompts"""
    
    def __init__(
        self,
        data_root: str,
        tokenizer: CLIPTokenizer,
        size: int = 512,
        interpolation: str = "bilinear",
    ):
        self.data_root = Path(data_root)
        self.tokenizer = tokenizer
        self.size = size
        
        # Load metadata
        metadata_path = self.data_root / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path, 'r', encoding='utf-8') as f:
                self.metadata = json.load(f)
        else:
            # Tạo metadata từ cấu trúc thư mục
            self.metadata = self._create_metadata()
        
        # Setup transforms
        self.image_transforms = transforms.Compose([
            transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])
    
    def _create_metadata(self) -> list:
        """Tạo metadata từ cấu trúc thư mục"""
        metadata = []
        
        # Tìm tất cả ảnh trong thư mục
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        for img_path in self.data_root.rglob("*"):
            if img_path.suffix.lower() in image_extensions:
                # Tạo prompt mặc định từ tên file/thư mục
                prompt = img_path.stem.replace('_', ' ').replace('-', ' ')
                
                metadata.append({
                    "image": str(img_path.relative_to(self.data_root)),
                    "prompt": prompt,
                    "caption": prompt  # Có thể thêm caption chi tiết hơn
                })
        
        # Lưu metadata
        with open(self.data_root / "metadata.json", 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        return metadata
    
    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, idx):
        item = self.metadata[idx]
        
        # Load image
        image_path = self.data_root / item["image"]
        image = Image.open(image_path).convert("RGB")
        image = self.image_transforms(image)
        
        # Tokenize text
        prompt = item.get("prompt", item.get("caption", ""))
        input_ids = self.tokenizer(
            prompt,
            max_length=self.tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        ).input_ids[0]
        
        return {
            "pixel_values": image,
            "input_ids": input_ids,
            "prompt": prompt
        }

def collate_fn(examples):
    """Collate function cho DataLoader"""
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
    
    input_ids = torch.stack([example["input_ids"] for example in examples])
    
    batch = {
        "pixel_values": pixel_values,
        "input_ids": input_ids,
    }
    return batch

def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune Stable Diffusion với LoRA")
    
    # Dataset arguments
    parser.add_argument("--data_dir", type=str, required=True, help="Thư mục chứa dataset")
    parser.add_argument("--output_dir", type=str, default="./lora_output", help="Thư mục output")
    
    # Model arguments  
    parser.add_argument("--pretrained_model_name_or_path", type=str, default="runwayml/stable-diffusion-v1-5")
    parser.add_argument("--revision", type=str, default=None)
    
    # Training arguments
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--train_batch_size", type=int, default=1)
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument("--max_train_steps", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--lr_scheduler", type=str, default="constant")
    parser.add_argument("--lr_warmup_steps", type=int, default=0)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--max_grad_norm", default=1.0, type=float)
    parser.add_argument("--mixed_precision", type=str, default=None, choices=["no", "fp16", "bf16"])
    
    # LoRA arguments
    parser.add_argument("--rank", type=int, default=4, help="LoRA rank")
    parser.add_argument("--alpha", type=int, default=32, help="LoRA alpha")
    parser.add_argument("--dropout", type=float, default=0.0, help="LoRA dropout")
    
    # Validation arguments
    parser.add_argument("--validation_prompt", type=str, default="a beautiful landscape")
    parser.add_argument("--validation_steps", type=int, default=100)
    parser.add_argument("--num_validation_images", type=int, default=4)
    
    # Other arguments
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--logging_dir", type=str, default="logs")
    parser.add_argument("--report_to", type=str, default="tensorboard")
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--checkpointing_steps", type=int, default=500)
    
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    
    # Initialize accelerator
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_dir=args.logging_dir,
    )
    
    # Set seed
    if args.seed is not None:
        set_seed(args.seed)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load models
    tokenizer = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="tokenizer", revision=args.revision
    )
    text_encoder = CLIPTextModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision
    )
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision
    )
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="unet", revision=args.revision
    )
    
    # Freeze models
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.requires_grad_(False)
    
    # Enable LoRA
    from peft import LoraConfig, get_peft_model
    
    lora_config = LoraConfig(
        r=args.rank,
        lora_alpha=args.alpha,
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
        lora_dropout=args.dropout,
    )
    
    unet = get_peft_model(unet, lora_config)
    
    # Enable xformers if available
    if is_xformers_available():
        unet.enable_xformers_memory_efficient_attention()
        vae.enable_xformers_memory_efficient_attention()
    
    # Create dataset
    train_dataset = ImageTextDataset(
        data_root=args.data_dir,
        tokenizer=tokenizer,
        size=args.resolution,
    )
    
    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=args.train_batch_size,
        num_workers=4,
    )
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        unet.parameters(),
        lr=args.learning_rate,
        betas=(0.9, 0.999),
        weight_decay=1e-2,
        eps=1e-08,
    )
    
    # Scheduler
    noise_scheduler = DDPMScheduler.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="scheduler"
    )
    
    # Calculate training steps
    num_update_steps_per_epoch = len(train_dataloader) // args.gradient_accumulation_steps
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
    )
    
    # Prepare everything with accelerator
    unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        unet, optimizer, train_dataloader, lr_scheduler
    )
    
    # Move vae and text_encoder to device
    vae.to(accelerator.device)
    text_encoder.to(accelerator.device)
    
    # Training loop
    logger.info("***** Bắt đầu training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size = {args.train_batch_size * accelerator.num_processes}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    
    global_step = 0
    first_epoch = 0
    
    progress_bar = tqdm(range(global_step, args.max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")
    
    for epoch in range(first_epoch, args.num_train_epochs):
        unet.train()
        train_loss = 0.0
        
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(unet):
                # Convert images to latent space
                latents = vae.encode(batch["pixel_values"]).latent_dist.sample()
                latents = latents * vae.config.scaling_factor
                
                # Sample noise
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
                timesteps = timesteps.long()
                
                # Add noise to latents
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
                
                # Get text embeddings
                encoder_hidden_states = text_encoder(batch["input_ids"])[0]
                
                # Predict noise
                model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
                
                # Calculate loss
                loss = F.mse_loss(model_pred.float(), noise.float(), reduction="mean")
                
                # Gather losses across processes
                avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
                train_loss += avg_loss.item() / args.gradient_accumulation_steps
                
                # Backpropagate
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(unet.parameters(), args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
            
            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                
                logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
                progress_bar.set_postfix(**logs)
                
                if global_step % args.save_steps == 0:
                    if accelerator.is_main_process:
                        # Save LoRA weights
                        unet_lora = unet.module if hasattr(unet, 'module') else unet
                        unet_lora.save_pretrained(os.path.join(args.output_dir, f"checkpoint-{global_step}"))
                
            if global_step >= args.max_train_steps:
                break
    
    # Save final model
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unet_lora = unet.module if hasattr(unet, 'module') else unet
        unet_lora.save_pretrained(args.output_dir)
        
        # Save training config
        config = {
            "args": vars(args),
            "final_step": global_step,
            "final_loss": train_loss,
        }
        with open(os.path.join(args.output_dir, "training_config.json"), 'w') as f:
            json.dump(config, f, indent=2)
    
    accelerator.end_training()
    logger.info("Training completed!")

if __name__ == "__main__":
    main() 