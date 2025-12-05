#!/usr/bin/env python3
"""
Image Utilities for Image Classification

This module provides utility functions for:
- Loading and preprocessing images
- Model loading and prediction
- File handling
"""

import os
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from transformers import AutoImageProcessor, AutoModelForImageClassification


class ModelWrapper(nn.Module):
    """Wrapper to ensure model returns logits for Captum compatibility."""
    def __init__(self, model):
        super().__init__()
        self.model = model
        
    def forward(self, x):
        output = self.model(x)
        if isinstance(output, torch.Tensor):
            return output
        return output.logits


def load_model(model_path):
    """
    Load a Hugging Face image classification model and return wrapped model.
    
    Args:
        model_path: Path or name of the Hugging Face model (e.g., 'microsoft/resnet-50')
    
    Returns:
        Tuple of (wrapped_model, processor, device)
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"Loading model: {model_path}")
    
    processor = AutoImageProcessor.from_pretrained(model_path)
    base_model = AutoModelForImageClassification.from_pretrained(model_path)
    base_model.to(device)
    base_model.eval()
    
    model = ModelWrapper(base_model)
    model.eval()
    
    return model, processor, device


def get_image_files(data_path, num_samples=None):
    """
    Get list of image files from a directory or single file.
    
    Args:
        data_path: Path to directory containing images or path to a single image file
        num_samples: Maximum number of images to return (only applies to directories).
                     If None, all images in the directory are returned.
    
    Returns:
        List of image file paths
    """
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif')
    
    # Check if data_path is a file or directory
    if os.path.isfile(data_path):
        # Single file provided
        if data_path.lower().endswith(image_extensions):
            print(f"Single image file detected: {data_path}")
            return [data_path]
        else:
            print(f"Error: File '{data_path}' is not a supported image format.")
            print(f"Supported formats: {', '.join(image_extensions)}")
            return []
    elif os.path.isdir(data_path):
        # Directory provided
        print(f"Directory detected: {data_path}")
        image_files = [os.path.join(data_path, f) for f in os.listdir(data_path) 
                       if f.lower().endswith(image_extensions)]
        if num_samples is not None:
            image_files = image_files[:num_samples]
        return image_files
    else:
        print(f"Error: Path '{data_path}' does not exist.")
        return []


def create_transform(processor):
    """
    Create image preprocessing transform.
    
    Args:
        processor: AutoImageProcessor from Hugging Face model
    
    Returns:
        torchvision.transforms.Compose object
    """
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=processor.image_mean, std=processor.image_std)
    ])
    return transform


def load_and_preprocess_image(image_path, transform, device):
    """
    Load and preprocess a single image.
    
    Args:
        image_path: Path to image file
        transform: Preprocessing transform
        device: Device to move tensor to (cuda/cpu)
    
    Returns:
        Tuple of (original_image_pil, input_tensor)
    """
    original_image = Image.open(image_path).convert('RGB')
    input_tensor = transform(original_image).unsqueeze(0).to(device)
    input_tensor.requires_grad = True
    return original_image, input_tensor


def get_model_prediction(model, input_tensor):
    """
    Get model prediction for an input tensor.
    
    Args:
        model: The wrapped model
        input_tensor: Input tensor
    
    Returns:
        Tuple of (pred_class, pred_prob)
    """
    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.nn.functional.softmax(output, dim=1)
        pred_class = output.argmax(dim=1).item()
        pred_prob = probs[0, pred_class].item()
    return pred_class, pred_prob
