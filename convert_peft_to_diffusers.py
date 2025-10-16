"""
Convert PEFT LoRA to Diffusers format
"""

import os
os.environ['HF_HOME'] = 'D:/huggingface_cache'

import torch
from safetensors.torch import load_file, save_file
from pathlib import Path

def convert_peft_to_diffusers(peft_path, output_path):
    """Convert PEFT LoRA to Diffusers format"""

    print(f"Converting PEFT LoRA to Diffusers format...")
    print(f"Input: {peft_path}")
    print(f"Output: {output_path}")

    # Load PEFT weights
    peft_weights = load_file(peft_path)

    print(f"Loaded {len(peft_weights)} tensors")

    # Convert keys: Remove "base_model.model." prefix
    diffusers_weights = {}

    for key, value in peft_weights.items():
        # Remove PEFT prefix
        new_key = key.replace("base_model.model.", "")
        diffusers_weights[new_key] = value

        if len(diffusers_weights) <= 3:
            print(f"  {key} -> {new_key}")

    print(f"Converted {len(diffusers_weights)} tensors")

    # Save in Diffusers format
    save_file(diffusers_weights, output_path)

    print(f"Saved to: {output_path}")
    print("Done!")

if __name__ == "__main__":
    convert_peft_to_diffusers(
        peft_path="lora_vietnamese/adapter_model.safetensors",
        output_path="lora_vietnamese/pytorch_lora_weights.safetensors"
    )
