import os
import json
import shutil
from pathlib import Path
from PIL import Image, ImageEnhance, ImageFilter
import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
import argparse
from tqdm import tqdm
import hashlib
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

class DatasetPreparator:
    """Class để chuẩn bị và xử lý dataset cho training"""
    
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Thống kê
        self.stats = {
            "total_images": 0,
            "processed_images": 0,
            "filtered_images": 0,
            "augmented_images": 0
        }
    
    def download_vietnamese_dataset(self) -> None:
        """Download dataset về Việt Nam từ các nguồn mở"""
        
        print("🇻🇳 Downloading Vietnamese dataset...")
        
        # URLs của các dataset mở về Việt Nam
        datasets = [
            {
                "name": "vietnam_landscapes",
                "urls": [
                    "https://example.com/vietnam_landscape_1.jpg",  # Thay bằng URLs thật
                    # Thêm nhiều URLs khác...
                ],
                "prompts": [
                    "beautiful Vietnamese landscape, mountains and valleys",
                    "traditional Vietnamese architecture",
                    "Vietnamese countryside scene"
                ]
            }
        ]
        
        # Tạo thư mục cho từng category
        for dataset in datasets:
            category_dir = self.output_dir / dataset["name"]
            category_dir.mkdir(exist_ok=True)
            
            # Download images
            for i, url in enumerate(tqdm(dataset["urls"], desc=f"Downloading {dataset['name']}")):
                try:
                    img_path = category_dir / f"img_{i:04d}.jpg"
                    urllib.request.urlretrieve(url, img_path)
                    
                    # Verify image
                    try:
                        img = Image.open(img_path)
                        img.verify()
                        self.stats["total_images"] += 1
                    except:
                        img_path.unlink()  # Delete corrupted image
                        
                except Exception as e:
                    print(f"Error downloading {url}: {e}")
    
    def create_vietnamese_prompts(self) -> List[str]:
        """Tạo danh sách prompts tiếng Việt và tiếng Anh"""
        
        vietnamese_prompts = [
            # Phong cảnh
            "phong cảnh Việt Nam đẹp, núi non hùng vĩ",
            "vịnh Hạ Long, đá vôi, thuyền buồm",
            "ruộng lúa bậc thang Sapa, màu xanh",
            "phố cổ Hội An, đèn lồng đầy màu sắc",
            "sông Mekong, đồng bằng xanh tươi",
            
            # Kiến trúc
            "chùa Việt Nam cổ, kiến trúc truyền thống",
            "nhà rường xứ Huế, mái ngói cong",
            "dinh thự Pháp cổ, Đà Lạt",
            "lăng tẩm hoàng gia, Huế",
            
            # Con người và văn hóa
            "người phụ nữ Việt mặc áo dài truyền thống",
            "nghệ nhân làm gốm Bát Tràng",
            "múa rối nước truyền thống",
            "lễ hội đầu xuân, màu sắc rực rỡ",
            
            # Ẩm thực
            "phở Việt Nam, tô phở nóng hổi",
            "bánh mì Việt Nam, giòn tan",
            "chợ Việt Nam, hoa quả tươi ngon",
            
            # Thiên nhiên
            "rừng nguyên sinh Việt Nam",
            "bãi biển Việt Nam, cát trắng nước trong",
            "hang động Việt Nam, thạch nhũ kỳ thú"
        ]
        
        # Translate to English prompts  
        english_prompts = [
            "beautiful Vietnamese landscape, majestic mountains",
            "Ha Long Bay, limestone karsts, sailing boats",
            "Sapa terraced rice fields, green color",
            "Hoi An ancient town, colorful lanterns",
            "Mekong River, fresh green delta",
            
            "Vietnamese ancient pagoda, traditional architecture", 
            "Hue traditional house, curved tile roof",
            "French colonial mansion, Da Lat",
            "imperial mausoleum, Hue",
            
            "Vietnamese woman wearing traditional ao dai",
            "Bat Trang pottery artisan",
            "traditional water puppet show",
            "New Year festival, vibrant colors",
            
            "Vietnamese pho, hot steaming bowl",
            "Vietnamese banh mi, crispy bread",
            "Vietnamese market, fresh fruits",
            
            "Vietnamese pristine forest",
            "Vietnamese beach, white sand clear water",
            "Vietnamese cave, stalactites and stalagmites"
        ]
        
        return vietnamese_prompts + english_prompts
    
    def filter_and_resize_images(self, source_dir: Path, min_resolution: int = 256, max_resolution: int = 1024) -> None:
        """Lọc và resize ảnh theo tiêu chuẩn"""
        
        print(f"🔍 Filtering and resizing images from {source_dir}")
        
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
        processed_dir = self.output_dir / "processed"
        processed_dir.mkdir(exist_ok=True)
        
        # Find all images
        image_paths = []
        for ext in image_extensions:
            image_paths.extend(source_dir.rglob(f"*{ext}"))
            image_paths.extend(source_dir.rglob(f"*{ext.upper()}"))
        
        for img_path in tqdm(image_paths, desc="Processing images"):
            try:
                # Load image
                img = Image.open(img_path).convert('RGB')
                w, h = img.size
                
                # Filter by resolution
                if min(w, h) < min_resolution or max(w, h) > max_resolution * 4:
                    self.stats["filtered_images"] += 1
                    continue
                
                # Filter by aspect ratio
                aspect_ratio = w / h
                if aspect_ratio > 3 or aspect_ratio < 1/3:
                    self.stats["filtered_images"] += 1
                    continue
                
                # Enhance image quality
                img = self.enhance_image_quality(img)
                
                # Smart resize
                img = self.smart_resize(img, target_size=512)
                
                # Generate unique filename
                img_hash = hashlib.md5(img.tobytes()).hexdigest()[:8]
                output_path = processed_dir / f"img_{img_hash}.jpg"
                
                # Save
                img.save(output_path, quality=95, optimize=True)
                self.stats["processed_images"] += 1
                
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
                continue
    
    def enhance_image_quality(self, img: Image.Image) -> Image.Image:
        """Tăng cường chất lượng ảnh"""
        
        # Remove noise
        img_array = np.array(img)
        img_array = cv2.bilateralFilter(img_array, 9, 75, 75)
        img = Image.fromarray(img_array)
        
        # Enhance sharpness
        enhancer = ImageEnhance.Sharpness(img)
        img = enhancer.enhance(1.1)
        
        # Enhance contrast
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(1.05)
        
        # Enhance color
        enhancer = ImageEnhance.Color(img)
        img = enhancer.enhance(1.1)
        
        return img
    
    def smart_resize(self, img: Image.Image, target_size: int = 512) -> Image.Image:
        """Resize thông minh giữ tỷ lệ"""
        
        w, h = img.size
        
        # Calculate resize ratio
        if w > h:
            new_w = target_size
            new_h = int(h * target_size / w)
        else:
            new_h = target_size  
            new_w = int(w * target_size / h)
        
        # Resize
        img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        
        # Pad to square
        canvas = Image.new('RGB', (target_size, target_size), (255, 255, 255))
        paste_x = (target_size - new_w) // 2
        paste_y = (target_size - new_h) // 2
        canvas.paste(img, (paste_x, paste_y))
        
        return canvas
    
    def augment_dataset(self, multiplier: int = 3) -> None:
        """Tăng cường dataset bằng data augmentation"""
        
        print(f"🔄 Augmenting dataset with {multiplier}x multiplier")
        
        processed_dir = self.output_dir / "processed"
        augmented_dir = self.output_dir / "augmented" 
        augmented_dir.mkdir(exist_ok=True)
        
        # Copy original images
        for img_path in processed_dir.glob("*.jpg"):
            shutil.copy2(img_path, augmented_dir / img_path.name)
        
        # Generate augmented versions
        original_images = list(processed_dir.glob("*.jpg"))
        
        for img_path in tqdm(original_images, desc="Augmenting"):
            img = Image.open(img_path)
            base_name = img_path.stem
            
            for i in range(multiplier - 1):  # -1 vì đã copy original
                # Random augmentation
                aug_img = self.apply_random_augmentation(img)
                
                # Save augmented image
                aug_path = augmented_dir / f"{base_name}_aug_{i+1}.jpg"
                aug_img.save(aug_path, quality=95)
                self.stats["augmented_images"] += 1
    
    def apply_random_augmentation(self, img: Image.Image) -> Image.Image:
        """Áp dụng augmentation ngẫu nhiên"""
        
        import random
        
        # Random rotation (-15 to 15 degrees)
        if random.random() > 0.5:
            angle = random.uniform(-15, 15)
            img = img.rotate(angle, expand=False, fillcolor=(255, 255, 255))
        
        # Random brightness
        if random.random() > 0.5:
            enhancer = ImageEnhance.Brightness(img)
            factor = random.uniform(0.8, 1.2)
            img = enhancer.enhance(factor)
        
        # Random contrast
        if random.random() > 0.5:
            enhancer = ImageEnhance.Contrast(img)
            factor = random.uniform(0.8, 1.2)
            img = enhancer.enhance(factor)
        
        # Random saturation
        if random.random() > 0.5:
            enhancer = ImageEnhance.Color(img)
            factor = random.uniform(0.8, 1.2)
            img = enhancer.enhance(factor)
        
        # Random horizontal flip
        if random.random() > 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        
        return img
    
    def create_metadata(self, use_vietnamese_prompts: bool = True) -> None:
        """Tạo metadata file với prompts"""
        
        print("📝 Creating metadata...")
        
        # Get all processed images
        final_dir = self.output_dir / "augmented"
        if not final_dir.exists():
            final_dir = self.output_dir / "processed"
        
        if not final_dir.exists():
            print("❌ No processed images found!")
            return
        
        image_paths = list(final_dir.glob("*.jpg"))
        prompts = self.create_vietnamese_prompts() if use_vietnamese_prompts else []
        
        metadata = []
        
        for img_path in tqdm(image_paths, desc="Creating metadata"):
            # Random prompt selection
            import random
            if prompts:
                prompt = random.choice(prompts)
            else:
                # Generate generic prompt
                prompt = "high quality, detailed, photorealistic"
            
            metadata.append({
                "image": str(img_path.relative_to(final_dir)),
                "prompt": prompt,
                "caption": prompt,
                "width": 512,
                "height": 512
            })
        
        # Save metadata
        metadata_path = final_dir / "metadata.json"
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Metadata saved to {metadata_path}")
        print(f"📊 Created metadata for {len(metadata)} images")
    
    def print_statistics(self) -> None:
        """In thống kê dataset"""
        
        print("\n📊 Dataset Statistics:")
        print(f"  Total images found: {self.stats['total_images']}")
        print(f"  Successfully processed: {self.stats['processed_images']}")
        print(f"  Filtered out: {self.stats['filtered_images']}")
        print(f"  Augmented images: {self.stats['augmented_images']}")
        
        # Count final images
        final_dir = self.output_dir / "augmented"
        if not final_dir.exists():
            final_dir = self.output_dir / "processed"
        
        if final_dir.exists():
            final_count = len(list(final_dir.glob("*.jpg")))
            print(f"  Final dataset size: {final_count} images")

def main():
    parser = argparse.ArgumentParser(description="Prepare dataset for Stable Diffusion training")
    
    parser.add_argument("--source_dir", type=str, help="Source directory containing raw images")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for processed dataset")
    parser.add_argument("--download_vietnamese", action="store_true", help="Download Vietnamese dataset")
    parser.add_argument("--augment_multiplier", type=int, default=3, help="Data augmentation multiplier")
    parser.add_argument("--min_resolution", type=int, default=256, help="Minimum image resolution")
    parser.add_argument("--max_resolution", type=int, default=1024, help="Maximum image resolution")
    parser.add_argument("--use_vietnamese_prompts", action="store_true", help="Use Vietnamese prompts")
    
    args = parser.parse_args()
    
    # Initialize preparator
    preparator = DatasetPreparator(args.output_dir)
    
    # Download Vietnamese dataset if requested
    if args.download_vietnamese:
        preparator.download_vietnamese_dataset()
    
    # Process existing images
    if args.source_dir:
        source_path = Path(args.source_dir)
        if source_path.exists():
            preparator.filter_and_resize_images(
                source_path, 
                args.min_resolution, 
                args.max_resolution
            )
        else:
            print(f"❌ Source directory {args.source_dir} does not exist!")
    
    # Augment dataset
    if args.augment_multiplier > 1:
        preparator.augment_dataset(args.augment_multiplier)
    
    # Create metadata
    preparator.create_metadata(args.use_vietnamese_prompts)
    
    # Print statistics
    preparator.print_statistics()
    
    print("✅ Dataset preparation completed!")

if __name__ == "__main__":
    main() 