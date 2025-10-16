"""
LoRA Integration Module for App
Giúp load và merge LoRA weights vào pipeline
"""

import openvino_genai as ov_genai
from pathlib import Path
from typing import Optional, List
import json

class LoRAManager:
    """Manager cho LoRA models trong app"""

    def __init__(self, lora_dir: str = "models/lora"):
        self.lora_dir = Path(lora_dir)
        self.lora_dir.mkdir(parents=True, exist_ok=True)

        self.available_loras = self._scan_loras()
        print(f"✓ Found {len(self.available_loras)} LoRA models")

    def _scan_loras(self) -> dict:
        """Scan và list tất cả LoRA có trong thư mục"""
        loras = {}

        # Scan for .safetensors files
        for lora_file in self.lora_dir.glob("*.safetensors"):
            lora_name = lora_file.stem
            loras[lora_name] = {
                "path": str(lora_file),
                "name": lora_name,
                "type": "safetensors"
            }

        # Scan for diffusers format (folders)
        for lora_folder in self.lora_dir.iterdir():
            if lora_folder.is_dir():
                # Check if it's a valid LoRA folder
                if (lora_folder / "pytorch_lora_weights.safetensors").exists():
                    lora_name = lora_folder.name
                    loras[lora_name] = {
                        "path": str(lora_folder),
                        "name": lora_name,
                        "type": "diffusers"
                    }

        # Load metadata if exists
        metadata_path = self.lora_dir / "lora_metadata.json"
        if metadata_path.exists():
            with open(metadata_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)

                for name, info in metadata.items():
                    if name in loras:
                        loras[name].update(info)

        return loras

    def get_lora_list(self) -> List[str]:
        """Get list of available LoRA names"""
        return ["None"] + list(self.available_loras.keys())

    def get_lora_info(self, lora_name: str) -> Optional[dict]:
        """Get thông tin của 1 LoRA"""
        return self.available_loras.get(lora_name)

    def load_lora_for_openvino(self, lora_name: str, base_model_path: str) -> str:
        """
        Load LoRA và merge vào base model (cho OpenVINO)

        Returns:
            Path to merged model
        """
        if lora_name not in self.available_loras:
            raise ValueError(f"LoRA '{lora_name}' not found")

        lora_info = self.available_loras[lora_name]
        lora_path = lora_info["path"]

        # Check if merged model already exists
        merged_path = Path("models/merged") / f"sd15_with_{lora_name}_ov"

        if merged_path.exists():
            print(f"✓ Using cached merged model: {merged_path}")
            return str(merged_path)

        # Need to merge and convert
        print(f"🔄 Merging LoRA '{lora_name}' with base model...")
        print(f"   This may take a few minutes...")

        import subprocess

        # Run conversion script
        cmd = [
            "python", "training/convert_to_openvino.py",
            "--model_path", base_model_path,
            "--lora_path", lora_path,
            "--output_path", str(merged_path),
            "--fp16"
        ]

        try:
            subprocess.run(cmd, check=True)
            print(f"✓ Merged model saved to: {merged_path}")
            return str(merged_path)

        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to merge LoRA: {e}")
            raise

    def create_metadata_entry(self, lora_name: str, description: str,
                            style: str, trained_on: str, examples: List[str]) -> None:
        """Tạo metadata entry cho 1 LoRA"""

        metadata_path = self.lora_dir / "lora_metadata.json"

        # Load existing metadata
        if metadata_path.exists():
            with open(metadata_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
        else:
            metadata = {}

        # Add new entry
        metadata[lora_name] = {
            "description": description,
            "style": style,
            "trained_on": trained_on,
            "examples": examples,
            "recommended_prompts": self._generate_recommended_prompts(style)
        }

        # Save
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        print(f"✓ Metadata saved for '{lora_name}'")

    def _generate_recommended_prompts(self, style: str) -> List[str]:
        """Generate recommended prompts based on style"""

        prompt_templates = {
            "vietnamese": [
                "Vietnamese landscape, beautiful scenery",
                "traditional Vietnamese architecture",
                "Vietnamese countryside, peaceful",
                "Mekong Delta, rice fields"
            ],
            "portrait": [
                "professional portrait, high quality",
                "detailed face, realistic skin",
                "studio lighting, bokeh background"
            ],
            "anime": [
                "anime style, detailed character",
                "manga illustration, vibrant colors",
                "anime art style, high quality"
            ],
            "default": [
                "high quality, detailed",
                "professional photography",
                "masterpiece, best quality"
            ]
        }

        return prompt_templates.get(style.lower(), prompt_templates["default"])


# Example usage
if __name__ == "__main__":
    # Initialize manager
    manager = LoRAManager()

    # List available LoRAs
    print("\n📋 Available LoRAs:")
    for lora_name in manager.get_lora_list():
        print(f"  - {lora_name}")

    # Get info
    if len(manager.available_loras) > 0:
        first_lora = list(manager.available_loras.keys())[0]
        info = manager.get_lora_info(first_lora)
        print(f"\n📄 Info for '{first_lora}':")
        print(f"   Path: {info['path']}")
        print(f"   Type: {info['type']}")
