import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import json

MODEL_PATH = "resnet18_finetune.pth"
IMAGE_SIZE = 128
device = torch.device("cpu")

st.set_page_config(
    page_title="Pet Classifier",
    page_icon="🐶",
    layout="wide"
)

with open("data/processed/class_to_idx.json", "r") as f:
    class_to_idx = json.load(f)

idx_to_class = {v: k for k, v in class_to_idx.items()}
num_classes = len(class_to_idx)

model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, num_classes)
checkpoint = torch.load(MODEL_PATH, map_location=device)
if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
    model.load_state_dict(checkpoint["model_state_dict"])
else:
    model.load_state_dict(checkpoint)
model.eval()

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

st.sidebar.title("Model & Dataset Info")
st.sidebar.write("Architecture: ResNet18")
st.sidebar.write("Dataset: The Oxford-IIIT Pet Dataset")
st.sidebar.write(f"Classes: {num_classes}")
st.sidebar.write("Input Size: 128x128")
st.sidebar.write("Device: CPU")

st.title("🐶🐱 Pet Classifier")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    col1, col2 = st.columns(2)
    
    image = Image.open(uploaded_file).convert("RGB")
    
    with col1:
        st.image(image, caption="Uploaded Image", width=400)

    input_tensor = transform(image).unsqueeze(0)

    with st.spinner("Predicting..."):
        with torch.no_grad():
            outputs = model(input_tensor)
            probs = torch.softmax(outputs, dim=1)
            top_probs, top_idxs = torch.topk(probs, 1)

    with col2:
        st.subheader("Prediction Result")
        
        best_class = idx_to_class[top_idxs[0][0].item()]
        best_conf = top_probs[0][0].item() * 100
        
        st.write("Prediction:", best_class)
        st.write("Confidence:", f"{best_conf:.2f}%")

st.sidebar.markdown("""
    <style>
    .copy-right {
        position: fixed;
        bottom: 0;
        left: 0;
        right: 0;
        text-align: left;
        margin-left: 20px;
        font-size: 14px;
        color: #888;
        background-color: inherit;
    }
    </style>
    <div class="copy-right">
        <p>© 2026 DuckYuh</p>
    </div>
""", unsafe_allow_html=True)
