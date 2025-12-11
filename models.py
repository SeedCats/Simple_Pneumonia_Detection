"""
Model and Image Processing Module
"""

import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from torchvision.models import ResNet50_Weights
import numpy as np
import cv2
from PIL import Image
from tkinter import messagebox
from config import IMAGE_SIZE, NORM_MEAN, NORM_STD


def create_model(num_classes):
    """Create ResNet50 model with custom classifier"""
    model = models.resnet50(weights=ResNet50_Weights.DEFAULT)
    model.fc = nn.Sequential(nn.Dropout(0.5), nn.Linear(model.fc.in_features, num_classes))
    return model


def get_transform():
    """Get image transformation pipeline"""
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(NORM_MEAN, NORM_STD)
    ])


def load_trained_model(model_path, num_classes, device):
    """Load trained model from checkpoint"""
    try:
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at: {model_path}")

        model = create_model(num_classes)
        state_dict = torch.load(model_path, map_location=device)
        # Handle DataParallel wrapped models
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict)
        model.to(device).eval()
        print(f"Model loaded successfully from {model_path}")
        return model
    except Exception as e:
        messagebox.showerror("Model Loading Error", f"Failed to load model: {e}")
        return None


def compute_grad_cam(model, target_layer, input_tensor, predicted_idx, device):
    """Compute Grad-CAM heatmap for model interpretation"""
    activations, gradients = [], []

    hook_f = target_layer.register_forward_hook(lambda m, i, o: activations.append(o))
    hook_b = target_layer.register_full_backward_hook(lambda m, gi, go: gradients.append(go[0]))

    input_tensor.requires_grad_(True)
    output = model(input_tensor.unsqueeze(0).to(device))
    model.zero_grad()
    output[0, predicted_idx].backward(retain_graph=True)

    hook_f.remove()
    hook_b.remove()

    # Compute weighted activation map
    weights = np.mean(gradients[0].cpu().data.numpy(), axis=(2, 3))[0]
    cam = np.sum(weights[:, None, None] * activations[0].cpu().data.numpy()[0], axis=0)
    cam = np.maximum(cam, 0)  # ReLU
    cam = cv2.resize(cam, (IMAGE_SIZE, IMAGE_SIZE))

    # Normalize
    if cam.max() > 0:
        cam = (cam - cam.min()) / cam.max()
    return cam


def analyze_image_features(image_path):
    """Perform basic image analysis using OpenCV"""
    try:
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            return None

        hist = cv2.calcHist([image], [0], None, [256], [0, 256])
        hist_norm = hist / hist.sum()
        edges = cv2.Canny(image, 50, 150)

        return {
            'mean_intensity': np.mean(image),
            'std_intensity': np.std(image),
            'histogram_entropy': -np.sum(hist_norm * np.log2(hist_norm + 1e-10)),
            'edge_density': np.sum(edges > 0) / image.size,
            'texture_complexity': np.std(cv2.Sobel(image, cv2.CV_64F, 1, 1, ksize=5))
        }
    except Exception as e:
        print(f"Image analysis error: {e}")
        return None


def predict_pneumonia_advanced(image_path, model, transform, device, class_names):
    """Enhanced prediction with feature analysis and Grad-CAM"""
    if not os.path.exists(image_path):
        return None, None, None, None, "Image file not found."

    try:
        image = Image.open(image_path).convert('RGB')
        original_image = image.copy()
    except Exception as e:
        return None, None, None, None, f"Error opening image: {e}"

    input_tensor = transform(image)

    # Prediction
    with torch.no_grad():
        output = model(input_tensor.unsqueeze(0).to(device))
        probs = torch.softmax(output, dim=1)
        confidence, pred_idx = torch.max(probs, 1)
        predicted_class = class_names[pred_idx.item()]

    # Grad-CAM
    try:
        grad_cam = compute_grad_cam(model, model.layer4, input_tensor, pred_idx.item(), device)
    except Exception as e:
        print(f"Grad-CAM error: {e}")
        grad_cam = None

    return predicted_class, confidence.item(), original_image, {
        'grad_cam_heatmap': grad_cam,
        'image_analysis': analyze_image_features(image_path),
        'raw_probabilities': probs.cpu().numpy()[0]
    }, None

