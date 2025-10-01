import torch
import openvino as ov
from pathlib import Path
import argparse
from diffusers import StableDiffusionPipeline, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer
import numpy as np
from typing import Tuple, Dict, Any
import json
import tempfile
import shutil

def export_text_encoder(
    text_encoder: CLIPTextModel,
    tokenizer: CLIPTokenizer,
    output_path: Path,
    input_shape: Tuple[int, int] = (1, 77)
) -> None:
    """Export CLIP text encoder to OpenVINO format"""
    
    print("🔄 Converting Text Encoder...")
    
    text_encoder.eval()
    
    # Create dummy input
    dummy_input = torch.ones(input_shape, dtype=torch.long)
    
    # Export to ONNX first (intermediate step)
    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as tmp_file:
        torch.onnx.export(
            text_encoder,
            dummy_input,
            tmp_file.name,
            input_names=["input_ids"],
            output_names=["hidden_states"],
            dynamic_axes={
                "input_ids": {0: "batch_size"},
                "hidden_states": {0: "batch_size"}
            },
            opset_version=14,
            do_constant_folding=True
        )
        
        # Convert ONNX to OpenVINO
        ov_model = ov.convert_model(tmp_file.name)
        
        # Save OpenVINO model
        output_path.mkdir(parents=True, exist_ok=True)
        ov.save_model(ov_model, output_path / "openvino_model.xml")
        
        # Save config
        config = {
            "model_type": "CLIPTextModel",
            "max_position_embeddings": text_encoder.config.max_position_embeddings,
            "vocab_size": text_encoder.config.vocab_size,
            "hidden_size": text_encoder.config.hidden_size,
        }
        
        with open(output_path / "config.json", 'w') as f:
            json.dump(config, f, indent=2)
    
    print(f"✅ Text Encoder saved to {output_path}")

def export_unet(
    unet: UNet2DConditionModel,
    output_path: Path,
    input_shape: Tuple[int, int, int, int] = (1, 4, 64, 64),
    encoder_hidden_states_shape: Tuple[int, int, int] = (1, 77, 768)
) -> None:
    """Export UNet to OpenVINO format"""
    
    print("🔄 Converting UNet...")
    
    unet.eval()
    
    # Create dummy inputs
    sample = torch.randn(input_shape)
    timestep = torch.tensor([1], dtype=torch.long)
    encoder_hidden_states = torch.randn(encoder_hidden_states_shape)
    
    # Export to ONNX first
    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as tmp_file:
        torch.onnx.export(
            unet,
            (sample, timestep, encoder_hidden_states),
            tmp_file.name,
            input_names=["sample", "timestep", "encoder_hidden_states"],
            output_names=["noise_pred"],
            dynamic_axes={
                "sample": {0: "batch_size", 2: "height", 3: "width"},
                "encoder_hidden_states": {0: "batch_size"},
                "noise_pred": {0: "batch_size", 2: "height", 3: "width"}
            },
            opset_version=14,
            do_constant_folding=True
        )
        
        # Convert ONNX to OpenVINO
        ov_model = ov.convert_model(tmp_file.name)
        
        # Optimize model
        core = ov.Core()
        
        # Apply optimizations
        pass_config = {"FORCE_FP32_OUTPUT_NAMES": "noise_pred"}
        ov_model = core.compile_model(ov_model, device_name="CPU", config=pass_config)
        
        # Save OpenVINO model
        output_path.mkdir(parents=True, exist_ok=True)
        ov.save_model(ov_model.model, output_path / "openvino_model.xml")
        
        # Save config
        config = unet.config
        if hasattr(config, 'to_dict'):
            config_dict = config.to_dict()
        else:
            config_dict = dict(config)
            
        with open(output_path / "config.json", 'w') as f:
            json.dump(config_dict, f, indent=2)
    
    print(f"✅ UNet saved to {output_path}")

def export_vae_encoder(vae, output_path: Path) -> None:
    """Export VAE encoder to OpenVINO format"""
    
    print("🔄 Converting VAE Encoder...")
    
    vae.encoder.eval()
    
    # Create dummy input (RGB image)
    dummy_input = torch.randn(1, 3, 512, 512)
    
    # Create a wrapper for VAE encoder
    class VAEEncoderWrapper(torch.nn.Module):
        def __init__(self, encoder):
            super().__init__()
            self.encoder = encoder
            
        def forward(self, sample):
            return self.encoder(sample)
    
    wrapper = VAEEncoderWrapper(vae.encoder)
    
    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as tmp_file:
        torch.onnx.export(
            wrapper,
            dummy_input,
            tmp_file.name,
            input_names=["sample"],
            output_names=["latent"],
            dynamic_axes={
                "sample": {0: "batch_size", 2: "height", 3: "width"},
                "latent": {0: "batch_size", 2: "height", 3: "width"}
            },
            opset_version=14,
            do_constant_folding=True
        )
        
        # Convert to OpenVINO
        ov_model = ov.convert_model(tmp_file.name)
        
        # Save
        output_path.mkdir(parents=True, exist_ok=True)
        ov.save_model(ov_model, output_path / "openvino_model.xml")
        
        # Save config
        config = {
            "model_type": "AutoencoderKL",
            "in_channels": vae.config.in_channels,
            "out_channels": vae.config.out_channels,
            "latent_channels": vae.config.latent_channels,
            "scaling_factor": vae.config.scaling_factor,
        }
        
        with open(output_path / "config.json", 'w') as f:
            json.dump(config, f, indent=2)
    
    print(f"✅ VAE Encoder saved to {output_path}")

def export_vae_decoder(vae, output_path: Path) -> None:
    """Export VAE decoder to OpenVINO format"""
    
    print("🔄 Converting VAE Decoder...")
    
    vae.decoder.eval()
    
    # Create dummy input (latent)
    dummy_input = torch.randn(1, 4, 64, 64)
    
    class VAEDecoderWrapper(torch.nn.Module):
        def __init__(self, decoder, post_quant_conv):
            super().__init__()
            self.decoder = decoder
            self.post_quant_conv = post_quant_conv
            
        def forward(self, latent):
            latent = self.post_quant_conv(latent)
            return self.decoder(latent)
    
    wrapper = VAEDecoderWrapper(vae.decoder, vae.post_quant_conv)
    
    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as tmp_file:
        torch.onnx.export(
            wrapper,
            dummy_input,
            tmp_file.name,
            input_names=["latent"],
            output_names=["sample"],
            dynamic_axes={
                "latent": {0: "batch_size", 2: "height", 3: "width"},
                "sample": {0: "batch_size", 2: "height", 3: "width"}
            },
            opset_version=14,
            do_constant_folding=True
        )
        
        # Convert to OpenVINO
        ov_model = ov.convert_model(tmp_file.name)
        
        # Save
        output_path.mkdir(parents=True, exist_ok=True)
        ov.save_model(ov_model, output_path / "openvino_model.xml")
        
        # Save config (same as encoder)
        config = {
            "model_type": "AutoencoderKL",
            "in_channels": vae.config.in_channels,
            "out_channels": vae.config.out_channels,
            "latent_channels": vae.config.latent_channels,
            "scaling_factor": vae.config.scaling_factor,
        }
        
        with open(output_path / "config.json", 'w') as f:
            json.dump(config, f, indent=2)
    
    print(f"✅ VAE Decoder saved to {output_path}")

def copy_additional_files(source_path: Path, output_path: Path) -> None:
    """Copy tokenizer, scheduler và các files cần thiết khác"""
    
    print("📁 Copying additional files...")
    
    # Files to copy
    files_to_copy = [
        "tokenizer/merges.txt",
        "tokenizer/vocab.json", 
        "tokenizer/special_tokens_map.json",
        "tokenizer/tokenizer_config.json",
        "scheduler/scheduler_config.json",
        "model_index.json",
        "feature_extractor/preprocessor_config.json"
    ]
    
    for file_path in files_to_copy:
        src = source_path / file_path
        dst = output_path / file_path
        
        if src.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            print(f"  ✅ Copied {file_path}")
        else:
            print(f"  ⚠️  {file_path} not found, skipping...")

def main():
    parser = argparse.ArgumentParser(description="Convert Stable Diffusion model to OpenVINO")
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained model hoặc HuggingFace model name")
    parser.add_argument("--lora_path", type=str, default=None, help="Path to LoRA weights (optional)")
    parser.add_argument("--output_path", type=str, required=True, help="Output directory for OpenVINO model")
    parser.add_argument("--fp16", action="store_true", help="Convert to FP16 (smaller model)")
    parser.add_argument("--int8", action="store_true", help="Apply INT8 quantization")
    
    args = parser.parse_args()
    
    print("🚀 Starting conversion to OpenVINO...")
    print(f"📁 Model: {args.model_path}")
    print(f"📁 Output: {args.output_path}")
    
    # Load model
    if Path(args.model_path).exists():
        # Local model
        pipe = StableDiffusionPipeline.from_pretrained(
            args.model_path,
            torch_dtype=torch.float16 if args.fp16 else torch.float32,
            safety_checker=None,
            requires_safety_checker=False
        )
    else:
        # HuggingFace model
        pipe = StableDiffusionPipeline.from_pretrained(
            args.model_path,
            torch_dtype=torch.float16 if args.fp16 else torch.float32,
            safety_checker=None,
            requires_safety_checker=False
        )
    
    # Load LoRA if provided
    if args.lora_path:
        print(f"🔧 Loading LoRA weights from {args.lora_path}")
        pipe.unet.load_attn_procs(args.lora_path)
    
    output_path = Path(args.output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Export each component
    export_text_encoder(pipe.text_encoder, pipe.tokenizer, output_path / "text_encoder")
    export_unet(pipe.unet, output_path / "unet")
    export_vae_encoder(pipe.vae, output_path / "vae_encoder")
    export_vae_decoder(pipe.vae, output_path / "vae_decoder")
    
    # Copy additional files
    if hasattr(pipe, 'save_pretrained'):
        # Save original model temporarily to copy files
        with tempfile.TemporaryDirectory() as tmp_dir:
            pipe.save_pretrained(tmp_dir)
            copy_additional_files(Path(tmp_dir), output_path)
    
    # Create model_index.json
    model_index = {
        "text_encoder": ["transformers", "CLIPTextModel"],
        "tokenizer": ["transformers", "CLIPTokenizer"],
        "unet": ["diffusers", "UNet2DConditionModel"],
        "vae": ["diffusers", "AutoencoderKL"],
        "scheduler": ["diffusers", "PNDMScheduler"],
        "safety_checker": None,
        "feature_extractor": None,
        "_class_name": "StableDiffusionPipeline",
        "_diffusers_version": "0.24.0"
    }
    
    with open(output_path / "model_index.json", 'w') as f:
        json.dump(model_index, f, indent=2)
    
    print(f"✅ Conversion completed! Model saved to {output_path}")
    
    # List output files
    print("\n📋 Generated files:")
    for file_path in sorted(output_path.rglob("*")):
        if file_path.is_file():
            print(f"  {file_path.relative_to(output_path)}")

if __name__ == "__main__":
    main() 