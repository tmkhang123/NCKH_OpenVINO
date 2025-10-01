#!/bin/bash

# ====================================
# Script Training LoRA cho Stable Diffusion
# ====================================

set -e  # Exit on any error

echo "🚀 Bắt đầu quy trình training Stable Diffusion với LoRA"

# Cấu hình mặc định
DATA_DIR="./dataset"
OUTPUT_DIR="./lora_output"
MODEL_NAME="runwayml/stable-diffusion-v1-5"
EPOCHS=50
BATCH_SIZE=1
LEARNING_RATE=1e-4
RANK=16
ALPHA=32

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --data_dir)
            DATA_DIR="$2"
            shift 2
            ;;
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --epochs)
            EPOCHS="$2"
            shift 2
            ;;
        --batch_size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --learning_rate)
            LEARNING_RATE="$2"
            shift 2
            ;;
        --rank)
            RANK="$2"
            shift 2
            ;;
        --alpha)
            ALPHA="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  --data_dir DIR        Dataset directory (default: ./dataset)"
            echo "  --output_dir DIR      Output directory (default: ./lora_output)"
            echo "  --epochs N            Number of epochs (default: 50)"
            echo "  --batch_size N        Batch size (default: 1)"
            echo "  --learning_rate LR    Learning rate (default: 1e-4)"
            echo "  --rank N              LoRA rank (default: 16)"
            echo "  --alpha N             LoRA alpha (default: 32)"
            echo "  -h, --help            Show this help"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Kiểm tra GPU
echo "🔍 Checking GPU availability..."
if python -c "import torch; print('✅ CUDA available:', torch.cuda.is_available())"; then
    DEVICE="cuda"
else
    echo "⚠️  CUDA not available, sử dụng CPU (sẽ chậm hơn)"
    DEVICE="cpu"
fi

# Tạo thư mục cần thiết
mkdir -p "$OUTPUT_DIR"
mkdir -p "logs"

# Step 1: Chuẩn bị dataset (nếu chưa có)
if [ ! -d "$DATA_DIR/processed" ] && [ ! -d "$DATA_DIR/augmented" ]; then
    echo "📁 Dataset chưa được chuẩn bị. Vui lòng chạy:"
    echo "python training/dataset_preparation.py --source_dir YOUR_SOURCE_DIR --output_dir $DATA_DIR --use_vietnamese_prompts"
    exit 1
fi

# Step 2: Training LoRA
echo "🎯 Bắt đầu training LoRA..."
echo "  📊 Epochs: $EPOCHS"
echo "  📦 Batch size: $BATCH_SIZE"
echo "  📈 Learning rate: $LEARNING_RATE"
echo "  🔧 LoRA rank: $RANK, alpha: $ALPHA"

# Xác định thư mục dataset
if [ -d "$DATA_DIR/augmented" ]; then
    TRAIN_DATA_DIR="$DATA_DIR/augmented"
else
    TRAIN_DATA_DIR="$DATA_DIR/processed"
fi

python training/train_lora.py \
    --data_dir "$TRAIN_DATA_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --pretrained_model_name_or_path "$MODEL_NAME" \
    --resolution 512 \
    --train_batch_size "$BATCH_SIZE" \
    --num_train_epochs "$EPOCHS" \
    --learning_rate "$LEARNING_RATE" \
    --lr_scheduler "cosine" \
    --lr_warmup_steps 100 \
    --rank "$RANK" \
    --alpha "$ALPHA" \
    --dropout 0.1 \
    --mixed_precision "fp16" \
    --gradient_accumulation_steps 4 \
    --max_grad_norm 1.0 \
    --save_steps 250 \
    --logging_dir "logs" \
    --validation_prompt "phong cảnh Việt Nam đẹp, núi non hùng vĩ, high quality, detailed" \
    --validation_steps 250 \
    --num_validation_images 4 \
    --seed 42

echo "✅ Training hoàn thành!"

# Step 3: Convert sang OpenVINO
echo "🔄 Converting model sang OpenVINO format..."

OPENVINO_OUTPUT="./models/custom_trained_ov"
python training/convert_to_openvino.py \
    --model_path "$MODEL_NAME" \
    --lora_path "$OUTPUT_DIR" \
    --output_path "$OPENVINO_OUTPUT" \
    --fp16

echo "✅ Conversion hoàn thành!"

# Step 4: Test model
echo "🧪 Testing trained model..."
python img2img.py

# Hiển thị kết quả
echo ""
echo "🎉 Quy trình hoàn thành!"
echo "📁 LoRA weights: $OUTPUT_DIR"
echo "📁 OpenVINO model: $OPENVINO_OUTPUT"
echo "📁 Logs: logs/"
echo ""
echo "💡 Để sử dụng model mới, cập nhật MODEL_DIR trong app.py và img2img.py:"
echo "   MODEL_DIR = \"$OPENVINO_OUTPUT\""
echo ""
echo "🚀 Chạy ứng dụng:"
echo "   python app.py" 