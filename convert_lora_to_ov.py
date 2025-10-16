"""
Convert LoRA to OpenVINO - With cache on D drive
"""

import os
from pathlib import Path

# Set cache to D drive BEFORE importing transformers/diffusers
os.environ['HF_HOME'] = 'D:/huggingface_cache'
os.environ['TRANSFORMERS_CACHE'] = 'D:/huggingface_cache'
os.environ['HF_DATASETS_CACHE'] = 'D:/huggingface_cache'

print(f"Cache location: {os.environ['HF_HOME']}")

# Now import
import subprocess
import sys

def convert_lora(lora_folder, output_name, lora_file="pytorch_lora_weights.safetensors"):
    """
    Convert LoRA to OpenVINO

    Args:
        lora_folder: Path to LoRA folder (e.g., "lora_vietnamese")
        output_name: Output model name (e.g., "vietnamese_ov")
        lora_file: LoRA weights filename (default: pytorch_lora_weights.safetensors)
    """

    lora_folder_path = Path(lora_folder)
    lora_path = lora_folder_path / lora_file
    output_path = Path("models") / output_name

    if not lora_path.exists():
        print(f"ERROR: LoRA file not found: {lora_path}")
        print(f"Checking folder: {lora_folder_path}")
        if lora_folder_path.exists():
            print(f"Available files:")
            for f in lora_folder_path.iterdir():
                print(f"  - {f.name}")
        return False

    print("="*60)
    print("Converting LoRA to OpenVINO")
    print("="*60)
    print(f"LoRA file: {lora_path}")
    print(f"Output: {output_path}")
    print(f"Cache: D:/huggingface_cache (not C:)")
    print("="*60)
    print()
    print("Step 1: Download SD 1.5 (if needed) - will use cache")
    print("Step 2: Load LoRA from file and merge")
    print("Step 3: Convert to OpenVINO format")
    print("Step 4: Save model")
    print()
    print("Time: 30-45 minutes (depends on CPU)")
    print("Press Ctrl+C to cancel...")
    print()

    # Run conversion script with LoRA FILE (not folder)
    cmd = [
        sys.executable,
        "training/convert_to_openvino.py",
        "--model_path", "runwayml/stable-diffusion-v1-5",
        "--lora_path", str(lora_path),  # This is now a FILE path
        "--output_path", str(output_path),
        "--fp16"
    ]

    try:
        subprocess.run(cmd, check=True)
        print()
        print("="*60)
        print("SUCCESS!")
        print(f"Model saved: {output_path}")
        print("="*60)
        return True

    except subprocess.CalledProcessError as e:
        print(f"ERROR: Conversion failed: {e}")
        return False
    except KeyboardInterrupt:
        print("\nCancelled by user")
        return False


if __name__ == "__main__":
    print("LoRA to OpenVINO Converter")
    print()

    # Convert Vietnamese LoRA
    success = convert_lora(
        lora_folder="lora_vietnamese",
        output_name="vietnamese_ov"
    )

    if success:
        print("\nReady to use in app!")
        print("Next: Update app_enhanced.py to use this model")
    else:
        print("\nConversion failed. Check errors above.")
