# 🐶🐱 Pet Classification

Phân loại giống thú cưng (chó và mèo) sử dụng Deep Learning với PyTorch.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://pet-clssification.streamlit.app/)
[![WandB](https://img.shields.io/badge/Weights_&_Biases-FFBE00?style=flat&logo=weightsandbiases&logoColor=black)](https://wandb.ai/duckyuh/pet-classification)

## 📋 Giới thiệu

Dự án này xây dựng mô hình phân loại **37 giống thú cưng** (12 giống mèo và 25 giống chó) sử dụng **The Oxford-IIIT Pet Dataset**. Hỗ trợ 2 kiến trúc mô hình:

- **SimpleCNN**: Mô hình CNN baseline 3 lớp convolution
- **ResNet18**: Transfer learning từ pretrained ImageNet weights

## 🗂️ Cấu trúc dự án

```
Pet-Classification/
├── app.py                 # Streamlit web app
├── train.py               # Script huấn luyện
├── evaluate.py            # Script đánh giá
├── dataset.py             # Dataset và data transforms
├── requirements.txt       # Dependencies
├── models/
│   ├── cnn_baseline.py    # SimpleCNN architecture
│   └── resnet_transfer.py # ResNet18 transfer learning
├── data/
│   ├── raw/               # Ảnh gốc
│   └── processed/         # Train/val/test splits
└── wandb/                 # Experiment logs
```

## 🚀 Cài đặt

### 1. Clone repository

```bash
git clone https://github.com/your-username/Pet-Classification.git
cd Pet-Classification
```

### 2. Tạo môi trường ảo (khuyến nghị)

```bash
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac
```

### 3. Cài đặt dependencies

```bash
pip install -r requirements.txt
```

## 📊 Dataset

**The Oxford-IIIT Pet Dataset** bao gồm:
- **37 classes**: 12 giống mèo + 25 giống chó
- Mỗi class có khoảng ~200 ảnh

### Danh sách các giống

| Mèo | Chó |
|-----|-----|
| Abyssinian | American Bulldog |
| Bengal | American Pit Bull Terrier |
| Birman | Basset Hound |
| Bombay | Beagle |
| British Shorthair | Boxer |
| Egyptian Mau | Chihuahua |
| Maine Coon | English Cocker Spaniel |
| Persian | English Setter |
| Ragdoll | German Shorthaired |
| Russian Blue | Great Pyrenees |
| Siamese | Havanese |
| Sphynx | Japanese Chin |
| | Keeshond |
| | Leonberger |
| | Miniature Pinscher |
| | Newfoundland |
| | Pomeranian |
| | Pug |
| | Saint Bernard |
| | Samoyed |
| | Scottish Terrier |
| | Shiba Inu |
| | Staffordshire Bull Terrier |
| | Wheaten Terrier |
| | Yorkshire Terrier |

## 🏋️ Huấn luyện

### Train CNN Baseline

```bash
python train.py --model cnn --epochs 10
```

### Train ResNet18 (Freeze backbone)

```bash
python train.py --model resnet18 --freeze-backbone --epochs 10
```

### Fine-tune ResNet18

```bash
python train.py --model resnet18 --epochs 10 --lr 1e-4 --resume resnet18_best.pth --save-path resnet18_finetune.pth
```

### Các tùy chọn khác

```bash
python train.py --model resnet18 \
    --batch-size 32 \
    --lr 0.0001 \
    --image-size 224 \
    --save-path my_model.pth
```

### Train với Weights & Biases logging

```bash
python train.py --model cnn --epochs 10 --wandb
```

### Tất cả arguments

| Argument | Default | Mô tả |
|----------|---------|-------|
| `--model` | `cnn` | Kiến trúc model (`cnn` hoặc `resnet18`) |
| `--batch-size` | `16` | Batch size |
| `--epochs` | `10` | Số epochs |
| `--lr` | `1e-3` | Learning rate |
| `--num-workers` | `4` | DataLoader workers |
| `--image-size` | `128` | Kích thước ảnh đầu vào |
| `--freeze-backbone` | `False` | Freeze backbone (chỉ ResNet) |
| `--resume` | `None` | Path checkpoint để continue training |
| `--save-path` | `None` | Path lưu model (default: `{model}_best.pth`) |
| `--wandb` | `False` | Enable wandb logging |

## 📈 Đánh giá

### Evaluate model

```bash
# Evaluate CNN
python evaluate.py --model cnn --checkpoint cnn_best.pth

# Evaluate ResNet18
python evaluate.py --model resnet18 --checkpoint resnet18_best.pth
```

### Với visualization plots

```bash
python evaluate.py --model resnet18 --checkpoint resnet18_best.pth --show-plots
```

### Các arguments

| Argument | Default | Mô tả |
|----------|---------|-------|
| `--model` | `cnn` | Kiến trúc model |
| `--checkpoint` | Required | Path đến file checkpoint |
| `--batch-size` | `16` | Batch size |
| `--image-size` | `128` | Kích thước ảnh |
| `--show-plots` | `False` | Hiển thị confusion matrix và predictions |

## 🌐 Web Application

Chạy ứng dụng Streamlit để demo:

```bash
python -m streamlit run app.py  
```

Hoặc truy cập bản deploy online: **[pet-clssification.streamlit.app](https://pet-clssification.streamlit.app/)**

## 🔧 Kiến trúc Model

### SimpleCNN

```
Conv2d(3, 32) → ReLU → MaxPool2d
Conv2d(32, 64) → ReLU → MaxPool2d
Conv2d(64, 128) → ReLU → MaxPool2d
AdaptiveAvgPool2d(1, 1)
Linear(128, 37)
```

### ResNet18 Transfer Learning

- Pretrained trên ImageNet
- Thay thế FC layer cuối: `Linear(512, 37)`
- Hỗ trợ freeze backbone hoặc fine-tune toàn bộ

## 📝 Data Augmentation

**Training transforms:**
- Resize to (128, 128)
- Random Horizontal Flip
- Color Jitter (brightness=0.2, contrast=0.2)
- Normalize (ImageNet mean/std)

**Evaluation transforms:**
- Resize to (128, 128)
- Normalize (ImageNet mean/std)

## 📊 Experiment Tracking

Xem các experiment trên Weights & Biases: **[wandb.ai/duckyuh/pet-classification](https://wandb.ai/duckyuh/pet-classification)**

## 📄 License

MIT License

## 👤 Tác giả

**DuckYuh** - 2026