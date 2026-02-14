# Train CNN baseline
python train.py --model cnn --epochs 10

# Train ResNet18 (freeze backbone)
python train.py --model resnet18 --freeze-backbone --epochs 10

# Train ResNet18 Fine-tune (Load Best Model để train tiếp tục)
python train.py --model resnet18 --epochs 10 --lr 1e-4 --resume resnet18_best.pth --save-path resnet18_fine-tune.pth

# Các options khác
python train.py --model resnet18 --batch-size 32 --lr 0.0001 --image-size 224 --save-path my_model.pth

# Train với wandb logging
python train.py --model cnn --epochs 10 --wandb

# Evaluate CNN model
python evaluate.py --model cnn --checkpoint cnn_best.pth

# Evaluate ResNet18 model
python evaluate.py --model resnet18 --checkpoint resnet18_best.pth

# Với visualization plots
python evaluate.py --model resnet18 --checkpoint resnet18_best.pth --show-plots

# Với image size khác
python evaluate.py --model resnet18 --checkpoint resnet18_best.pth --image-size 224

# Run app
python -m streamlit run app.py