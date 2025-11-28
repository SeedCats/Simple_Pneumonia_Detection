"""
Model and Image Processing Module
Handles model creation, loading, predictions, and image analysis
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

from config import IMAGE_SIZE, NORM_MEAN, NORM_STD, NUM_CLASSES, CLASS_NAMES


def create_model(num_classes):
    """Create ResNet50 model with custom classifier"""
    model = models.resnet50(weights=ResNet50_Weights.DEFAULT)
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_ftrs, num_classes)
    )
    return model


def get_transform():
    """Get image transformation pipeline"""
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(NORM_MEAN, NORM_STD)
    ])
    return transform


def load_trained_model(model_path, num_classes, device):
    """Load trained model from checkpoint"""
    try:
        model = create_model(num_classes)
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at: {model_path}")

        state_dict = torch.load(model_path, map_location=device)
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                new_state_dict[k[7:]] = v
            else:
                new_state_dict[k] = v

        model.load_state_dict(new_state_dict)
        model.to(device)
        model.eval()
        print(f"Model loaded successfully from {model_path} and set to evaluation mode.")
        return model
    except Exception as e:
        messagebox.showerror("Model Loading Error", f"Failed to load model: {e}")
        return None


def compute_grad_cam(model, target_layer, input_tensor, predicted_idx, device):
    """
    Compute Grad-CAM for a given model and input
    Showing the regions in the image that influenced the model's decision
    (How model arrived at its prediction)

    Args:
        model (nn.Module): The classification model
        target_layer (nn.Module): The last convolutional layer to extract features from
        input_tensor (torch.Tensor): Preprocessed image tensor (1, C, H, W)
        predicted_idx (int): The index of the predicted class
        device (torch.device): The device (cpu/cuda) where the model is

    Returns:
        np.ndarray: The Grad-CAM heatmap, resized to the input image size
    """
    model.eval()

    # Store activations and gradients
    activations = []
    gradients = []

    def save_activation(module, input, output):
        activations.append(output)

    def save_gradient(module, grad_input, grad_output):
        gradients.append(grad_output[0])

    # Register hooks
    hook_handle_f = target_layer.register_forward_hook(save_activation)
    hook_handle_b = target_layer.register_full_backward_hook(save_gradient)

    # Perform forward pass
    input_tensor.requires_grad_(True)
    output = model(input_tensor.unsqueeze(0).to(device))

    # Zero gradients
    model.zero_grad()

    # Compute gradients of the predicted class score
    score = output[0, predicted_idx]
    score.backward(retain_graph=True)

    # Remove hooks
    hook_handle_f.remove()
    hook_handle_b.remove()

    # Get activations and gradients
    activations_np = activations[0].cpu().data.numpy()  # (1, C, H_feature, W_feature)
    gradients_np = gradients[0].cpu().data.numpy()  # (1, C, H_feature, W_feature)

    # Global Average Pooling of gradients
    weights = np.mean(gradients_np, axis=(2, 3))[0]  # (C,)

    # Weighted combination of activations
    cam = np.zeros(activations_np.shape[2:], dtype=np.float32)  # (H_feature, W_feature)
    for i, w in enumerate(weights):
        cam += w * activations_np[0, i, :, :]

    # Apply ReLU
    cam = np.maximum(cam, 0)

    # Normalize and resize
    cam = cv2.resize(cam, (IMAGE_SIZE, IMAGE_SIZE))
    cam = cam - np.min(cam)
    if np.max(cam) > 0:
        cam = cam / np.max(cam)
    else:
        cam = np.zeros_like(cam)

    return cam


def analyze_image_features(image_path):
    """Perform basic image analysis using OpenCV"""
    try:
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            return None

        analysis = {}

        # Basic statistics
        analysis['mean_intensity'] = np.mean(image)
        analysis['std_intensity'] = np.std(image)
        analysis['min_intensity'] = np.min(image)
        analysis['max_intensity'] = np.max(image)

        # Histogram analysis
        hist = cv2.calcHist([image], [0], None, [256], [0, 256])
        normalized_hist = hist / np.sum(hist)
        analysis['histogram_entropy'] = -np.sum(normalized_hist * np.log2(normalized_hist + 1e-10))

        # Edge detection analysis
        edges = cv2.Canny(image, 50, 150)
        analysis['edge_density'] = np.sum(edges > 0) / (image.shape[0] * image.shape[1])

        # Texture analysis using GLCM-like features (simplified)
        analysis['texture_complexity'] = np.std(cv2.Sobel(image, cv2.CV_64F, 1, 1, ksize=5))

        return analysis
    except Exception as e:
        print(f"Image analysis error: {e}")
        return None


def predict_pneumonia_advanced(image_path, model, transform, device, class_names):
    """
    Enhanced prediction with feature analysis and Grad-CAM

    Args:
        image_path: Path to the input image
        model: The trained model
        transform: Image transformation pipeline
        device: torch device
        class_names: List of class names

    Returns:
        Tuple of (predicted_class, confidence, original_image, analysis_data, error)
    """
    if not os.path.exists(image_path):
        return None, None, None, None, "Image file not found."

    try:
        image = Image.open(image_path).convert('RGB')
        original_image = image.copy()
    except Exception as e:
        return None, None, None, None, f"Error opening or processing image: {e}"

    input_tensor = transform(image)

    # Perform prediction
    with torch.no_grad():
        output = model(input_tensor.unsqueeze(0).to(device))
        probabilities = torch.softmax(output, dim=1)
        predicted_prob, predicted_idx_tensor = torch.max(probabilities, 1)
        predicted_idx = predicted_idx_tensor.item()
        predicted_class = class_names[predicted_idx]
        confidence = predicted_prob.item()

    # Compute Grad-CAM
    try:
        grad_cam_heatmap = compute_grad_cam(model, model.layer4, input_tensor, predicted_idx, device)
    except Exception as e:
        print(f"Error computing Grad-CAM: {e}")
        grad_cam_heatmap = None

    # Get image feature analysis
    image_analysis = analyze_image_features(image_path)

    return predicted_class, confidence, original_image, {
        'grad_cam_heatmap': grad_cam_heatmap,
        'image_analysis': image_analysis,
        'raw_probabilities': probabilities.cpu().numpy()[0]
    }, None

