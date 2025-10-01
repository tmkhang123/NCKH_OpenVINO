# 🎨 NCKH OpenVINO - Advanced Image-to-Image Generation

Dự án nghiên cứu khoa học về ứng dụng OpenVINO để tạo ảnh từ ảnh (Image-to-Image) với Stable Diffusion, tối ưu hóa cho nội dung Việt Nam.

## 🌟 Tính năng

### ✨ **Hiện tại**
- ✅ Image-to-Image generation với Stable Diffusion v1.5 INT8
- ✅ Giao diện web thân thiện với Gradio
- ✅ Tối ưu hóa OpenVINO cho CPU Intel
- ✅ Preprocessing ảnh thông minh
- ✅ Metrics đánh giá chất lượng (PSNR, SSIM)

### 🚀 **Mới được thêm**
- ✅ **Multi-model support**: Chuyển đổi giữa nhiều models
- ✅ **LoRA training pipeline**: Tự train model với dữ liệu riêng
- ✅ **Vietnamese presets**: Prompts tối ưu cho nội dung Việt Nam
- ✅ **Style enhancement**: Tự động tăng cường prompt
- ✅ **Dataset preparation tools**: Chuẩn bị và augment dataset
- ✅ **Model evaluation**: So sánh chất lượng giữa các models
- ✅ **OpenVINO conversion**: Convert PyTorch sang OpenVINO

## 📁 Cấu trúc dự án

```
NCKH_OpenVINO/
├── app.py                          # Web app chính (đã nâng cấp)
├── img2img.py                      # Script xử lý độc lập (đã cải tiến)
├── prepare_model.py                # Tải model từ HuggingFace
├── requirements.txt                # Dependencies đầy đủ
├── run_training.sh                 # Script chạy training tự động
├── 
├── training/                       # 🆕 Training infrastructure
│   ├── train_lora.py              # LoRA fine-tuning
│   ├── convert_to_openvino.py     # Convert PyTorch → OpenVINO
│   ├── dataset_preparation.py     # Chuẩn bị dataset
│   └── evaluate_model.py          # Đánh giá model
├── 
├── models/                         # Models directory
│   ├── sd15_int8_ov/              # Base Stable Diffusion 1.5
│   ├── custom_trained_ov/         # 🆕 Custom trained model
│   └── sdxl_int8_ov/              # 🆕 SDXL (coming soon)
├──
├── input/                          # Ảnh đầu vào
├── dataset/                        # 🆕 Training dataset
│   ├── processed/                 # Ảnh đã xử lý
│   ├── augmented/                 # Ảnh đã augment
│   └── metadata.json             # Metadata với prompts
└── 
└── evaluation_results/             # 🆕 Kết quả đánh giá models
```

## 🚀 Hướng dẫn sử dụng

### 1. **Setup môi trường**

```bash
# Clone repository
git clone <your-repo>
cd NCKH_OpenVINO

# Cài đặt dependencies
pip install -r requirements.txt

# Tải base model
python prepare_model.py
```

### 2. **Chạy ứng dụng cơ bản**

```bash
# Web app với giao diện nâng cao
python app.py

# Hoặc chạy script độc lập
python img2img.py
```

### 3. **🎯 Training model riêng (ADVANCED)**

#### **Bước 1: Chuẩn bị dataset**

```bash
# Chuẩn bị dataset từ thư mục ảnh
python training/dataset_preparation.py \
    --source_dir "path/to/your/images" \
    --output_dir "./dataset" \
    --use_vietnamese_prompts \
    --augment_multiplier 3
```

#### **Bước 2: Training LoRA**

```bash
# Sử dụng script tự động (RECOMMENDED)
./run_training.sh --epochs 100 --rank 16

# Hoặc chạy manual
python training/train_lora.py \
    --data_dir "./dataset/augmented" \
    --output_dir "./lora_output" \
    --num_train_epochs 100 \
    --rank 16 \
    --alpha 32
```

#### **Bước 3: Convert sang OpenVINO**

```bash
python training/convert_to_openvino.py \
    --model_path "runwayml/stable-diffusion-v1-5" \
    --lora_path "./lora_output" \
    --output_path "./models/custom_trained_ov" \
    --fp16
```

#### **Bước 4: Đánh giá model**

```bash
python training/evaluate_model.py \
    --base_model "runwayml/stable-diffusion-v1-5" \
    --trained_model "./lora_output" \
    --openvino_model "./models/custom_trained_ov" \
    --output_dir "./evaluation_results"
```

## 🇻🇳 Tối ưu hóa cho nội dung Việt Nam

### **Presets có sẵn:**
- 🏔️ **Phong cảnh Việt Nam**: Núi non, thiên nhiên
- 🚢 **Vịnh Hạ Long**: Đá vôi, thuyền buồm
- 🏮 **Phố cổ Hội An**: Đèn lồng, kiến trúc cổ
- 👘 **Áo dài truyền thống**: Trang phục dân tộc
- 🏯 **Chùa Việt Nam**: Kiến trúc tâm linh
- 🌾 **Ruộng lúa Sapa**: Bậc thang, sương mù

### **Vietnamese Prompts Examples:**
```
"phong cảnh Việt Nam đẹp, núi non hùng vĩ, thiên nhiên xanh tươi"
"vịnh Hạ Long, đá vôi, thuyền buồm, nước biển xanh trong"
"phố cổ Hội An, đèn lồng đầy màu sắc, kiến trúc cổ"
"người phụ nữ Việt mặc áo dài truyền thống, elegant"
"chùa Việt Nam cổ, kiến trúc truyền thống, mái ngói cong"
```

## ⚙️ Cấu hình nâng cao

### **Training Parameters:**
```bash
# LoRA settings
--rank 16              # LoRA rank (4-64)
--alpha 32             # LoRA alpha (thường = rank * 2)
--dropout 0.1          # Dropout rate

# Training settings  
--learning_rate 1e-4   # Learning rate
--batch_size 1         # Batch size (depends on VRAM)
--epochs 100           # Number of epochs
--mixed_precision fp16 # Memory optimization
```

### **Model Selection:**
```python
# Trong app.py, có thể thêm models:
AVAILABLE_MODELS = {
    "SD 1.5 INT8": "models/sd15_int8_ov",
    "Custom Trained": "models/custom_trained_ov", 
    "SDXL": "models/sdxl_int8_ov",
    "Your Model": "path/to/your/model"
}
```

## 📊 Evaluation & Metrics

### **Metrics được sử dụng:**
- **PSNR**: Peak Signal-to-Noise Ratio (chất lượng ảnh)
- **SSIM**: Structural Similarity Index (độ tương đồng cấu trúc)
- **MSE**: Mean Squared Error (lỗi trung bình)
- **Generation Time**: Thời gian tạo ảnh

### **So sánh models:**
```bash
# Tạo báo cáo so sánh đầy đủ
python training/evaluate_model.py \
    --base_model "runwayml/stable-diffusion-v1-5" \
    --trained_model "./lora_output" \
    --openvino_model "./models/custom_trained_ov"
```

## 🛠 Troubleshooting

### **Lỗi thường gặp:**

**1. Model không tìm thấy:**
```bash
# Kiểm tra path model
ls models/
# Tải lại model nếu cần
python prepare_model.py
```

**2. CUDA out of memory:**
```bash
# Giảm batch size
--batch_size 1
# Sử dụng gradient accumulation
--gradient_accumulation_steps 4
```

**3. OpenVINO conversion error:**
```bash
# Kiểm tra dependencies
pip install openvino>=2024.0.0
# Sử dụng CPU thay vì GPU
DEVICE = "CPU"
```

## 📈 Performance Tips

### **Tối ưu tốc độ:**
1. **Sử dụng INT8 quantization** cho models
2. **Batch processing** cho nhiều ảnh
3. **OpenVINO optimization** cho CPU Intel
4. **Cached models** để tránh reload

### **Tối ưu chất lượng:**
1. **Tăng số steps** (20-50 steps)
2. **Fine-tune với dataset riêng**
3. **Sử dụng negative prompts** hiệu quả
4. **Experiment với strength values**

## 🎯 Roadmap

### **Đã hoàn thành:**
- ✅ Basic image-to-image generation
- ✅ LoRA training pipeline  
- ✅ Vietnamese content optimization
- ✅ Model evaluation system
- ✅ OpenVINO conversion

### **Đang phát triển:**
- 🔄 ControlNet integration
- 🔄 SDXL support
- 🔄 Real-time inference
- 🔄 Mobile deployment

### **Kế hoạch tương lai:**
- 📋 Inpainting/Outpainting
- 📋 Video processing
- 📋 3D model integration
- 📋 API service

## 👥 Contributing

Đóng góp cho dự án:

1. Fork repository
2. Tạo feature branch: `git checkout -b feature/AmazingFeature`
3. Commit changes: `git commit -m 'Add AmazingFeature'`
4. Push to branch: `git push origin feature/AmazingFeature`
5. Tạo Pull Request

## 📝 License

Dự án được phát hành dưới MIT License. Xem `LICENSE` file để biết thêm chi tiết.

## 🙏 Acknowledgments

- [OpenVINO](https://github.com/openvinotoolkit/openvino) - AI inference optimization
- [Diffusers](https://github.com/huggingface/diffusers) - Stable Diffusion implementation
- [Gradio](https://github.com/gradio-app/gradio) - Web interface framework
- [HuggingFace](https://huggingface.co/) - Model hosting and ecosystem

---

**Tác giả**: [Your Name]  
**Trường**: [Your University]  
**Khóa**: [Your Academic Year]  
**Email**: [your.email@university.edu]

🚀 **Happy researching!** 🇻🇳 