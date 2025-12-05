#!/usr/bin/env python3
"""
Image Classification Explainability with Captum

This module focuses on explainability functions for image classification:
- Computing attribution maps (GradientShap, IntegratedGradients, Saliency)
- Visualizing and saving attribution results
- Orchestrating complete explainability workflow

Utility functions for model loading and data processing are in helpers/image_utils.py
"""

import torch
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

from captum.attr import GradientShap, IntegratedGradients, Saliency

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Import helper functions for image utilities
from helpers.image_utils import (
    load_model,
    get_image_files,
    create_transform,
    load_and_preprocess_image,
    get_model_prediction
)


def compute_gradient_shap(model, input_tensor, pred_class, device, n_samples=50):
    """
    Compute GradientShap attributions for an input image.
    
    Args:
        model: The wrapped model
        input_tensor: Input tensor (batch_size, channels, height, width)
        pred_class: Target class for attribution
        device: Device to run on (cuda/cpu)
        n_samples: Number of samples for GradientShap
    
    Returns:
        Dictionary with attributions and statistics, or None if error occurs
    """
    print("Computing GradientShap attributions...")
    try:
        gradient_shap = GradientShap(model)
        baselines = torch.randn(10, *input_tensor.shape[1:]).to(device) * 0.1
        
        gs_attr = gradient_shap.attribute(input_tensor, 
                                         baselines=baselines,
                                         target=pred_class,
                                         n_samples=n_samples)
        gs_attr_np = gs_attr.squeeze().cpu().detach().numpy()
        gs_attr_sum = np.sum(np.abs(gs_attr_np), axis=0)
        
        print(f"GradientShap attribution range: [{gs_attr_sum.min():.4f}, {gs_attr_sum.max():.4f}]")
        print(f"GradientShap mean absolute attribution: {np.mean(np.abs(gs_attr_sum)):.4f}")
        
        return {
            'attributions': gs_attr_sum,
            'min': gs_attr_sum.min(),
            'max': gs_attr_sum.max(),
            'mean': np.mean(np.abs(gs_attr_sum)),
            'method': 'GradientShap'
        }
    except Exception as e:
        print(f"Error computing GradientShap: {e}")
        return None


def compute_integrated_gradients(model, input_tensor, pred_class, device, n_steps=50):
    """
    Compute Integrated Gradients attributions for an input image.
    
    Args:
        model: The wrapped model
        input_tensor: Input tensor (batch_size, channels, height, width)
        pred_class: Target class for attribution
        device: Device to run on (cuda/cpu)
        n_steps: Number of integration steps
    
    Returns:
        Dictionary with attributions and statistics, or None if error occurs
    """
    print("Computing Integrated Gradients attributions...")
    try:
        integrated_gradients = IntegratedGradients(model)
        baseline = torch.zeros_like(input_tensor)
        
        ig_attr = integrated_gradients.attribute(input_tensor,
                                                baselines=baseline,
                                                target=pred_class,
                                                n_steps=n_steps)
        ig_attr_np = ig_attr.squeeze().cpu().detach().numpy()
        ig_attr_sum = np.sum(np.abs(ig_attr_np), axis=0)
        
        print(f"IntegratedGradients attribution range: [{ig_attr_sum.min():.4f}, {ig_attr_sum.max():.4f}]")
        print(f"IntegratedGradients mean absolute attribution: {np.mean(np.abs(ig_attr_sum)):.4f}")
        
        return {
            'attributions': ig_attr_sum,
            'min': ig_attr_sum.min(),
            'max': ig_attr_sum.max(),
            'mean': np.mean(np.abs(ig_attr_sum)),
            'method': 'Integrated Gradients'
        }
    except Exception as e:
        print(f"Error computing Integrated Gradients: {e}")
        return None


def compute_saliency(model, input_tensor, pred_class, device):
    """
    Compute Saliency attributions for an input image.
    
    Args:
        model: The wrapped model
        input_tensor: Input tensor (batch_size, channels, height, width)
        pred_class: Target class for attribution
        device: Device to run on (cuda/cpu)
    
    Returns:
        Dictionary with attributions and statistics, or None if error occurs
    """
    print("Computing Saliency attributions...")
    try:
        saliency = Saliency(model)
        sal_attr = saliency.attribute(input_tensor, target=pred_class)
        sal_attr_np = sal_attr.squeeze().cpu().detach().numpy()
        sal_attr_sum = np.sum(np.abs(sal_attr_np), axis=0)
        
        print(f"Saliency attribution range: [{sal_attr_sum.min():.4f}, {sal_attr_sum.max():.4f}]")
        print(f"Saliency mean absolute attribution: {np.mean(np.abs(sal_attr_sum)):.4f}")
        
        return {
            'attributions': sal_attr_sum,
            'min': sal_attr_sum.min(),
            'max': sal_attr_sum.max(),
            'mean': np.mean(np.abs(sal_attr_sum)),
            'method': 'Saliency'
        }
    except Exception as e:
        print(f"Error computing Saliency: {e}")
        return None


def save_attribution_visualization(original_image, attributions, method_name, output_path, pred_class, pred_prob):
    """
    Save individual attribution visualization.
    
    Args:
        original_image: PIL Image of original image
        attributions: Attribution heatmap (numpy array)
        method_name: Name of the method (e.g., 'GradientShap')
        output_path: Path to save the visualization
        pred_class: Predicted class index
        pred_prob: Prediction probability
    """
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    
    # Original image
    ax[0].imshow(original_image)
    ax[0].set_title(f'Original Image\nPred: Class {pred_class} ({pred_prob:.2%})')
    ax[0].axis('off')
    
    # Attribution heatmap
    im = ax[1].imshow(attributions, cmap='hot')
    ax[1].set_title(f'{method_name} Attribution')
    ax[1].axis('off')
    plt.colorbar(im, ax=ax[1], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    Saved: {output_path}")


def save_comparison_visualization(original_image, attributions_dict, output_path, pred_class, pred_prob, image_name):
    """
    Save comparison visualization with all attribution methods.
    
    Args:
        original_image: PIL Image of original image
        attributions_dict: Dictionary with keys as method names and values as attribution arrays
        output_path: Path to save the comparison
        pred_class: Predicted class index
        pred_prob: Prediction probability
        image_name: Name of the image file
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # Original image
    axes[0, 0].imshow(original_image)
    axes[0, 0].set_title(f'Original Image\nPred: Class {pred_class} ({pred_prob:.2%})', fontsize=12, fontweight='bold')
    axes[0, 0].axis('off')
    
    # Attributions
    positions = [(0, 1), (1, 0), (1, 1)]
    for (row, col), (method_name, attr_data) in zip(positions, attributions_dict.items()):
        im = axes[row, col].imshow(attr_data, cmap='hot')
        axes[row, col].set_title(f'{method_name} Attribution', fontsize=12, fontweight='bold')
        axes[row, col].axis('off')
        plt.colorbar(im, ax=axes[row, col], fraction=0.046, pad=0.04)
    
    plt.suptitle(f'Explainability Methods Comparison: {image_name}', fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def explain_model_predictions(model_path, data_path, output_dir='./outputs', num_samples=None, methods=None):
    """
    Apply explainability methods (GradientShap, IntegratedGradients, Saliency) to a model.
    
    Orchestrates the use of individual explainability functions to analyze model predictions.
    
    Args:
        model_path: Path or name of the Hugging Face model (e.g., 'microsoft/resnet-50')
        data_path: Path to directory containing images OR path to a single image file
        output_dir: Directory to save visualization outputs
        num_samples: Number of images to analyze (only applies when data_path is a directory).
                     If None, all images in the directory are analyzed.
        methods: List of methods to use. Options: ['gradientshap', 'integratedgradients', 'saliency']
                 If None, all methods are used. Examples: ['gradientshap'], ['saliency', 'integratedgradients']
    """
    # Default to all methods if none specified
    if methods is None:
        methods = ['gradientshap', 'integratedgradients', 'saliency']
    
    # Normalize method names to lowercase
    methods = [m.lower() for m in methods]
    
    # Validate methods
    valid_methods = ['gradientshap', 'integratedgradients', 'saliency']
    for method in methods:
        if method not in valid_methods:
            print(f"Error: Unknown method '{method}'. Valid options: {valid_methods}")
            sys.exit(1)
    
    print(f"Selected methods: {', '.join(methods)}")
    
    # Load model
    model, processor, device = load_model(model_path)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get image files and create preprocessing transform
    image_files = get_image_files(data_path, num_samples)
    
    if not image_files:
        print(f"Error: No images found in {data_path}")
        print("Program stopping.")
        sys.exit(1)
    
    transform = create_transform(processor)
    
    print(f"\nFound {len(image_files)} images to analyze")
    
    print("\n" + "="*80)
    print("EXPLAINABILITY ANALYSIS RESULTS")
    print("="*80)
    
    # Process each image
    for idx, image_path in enumerate(image_files):
        image_name = os.path.basename(image_path)
        print(f"\n--- Image {idx+1}/{len(image_files)}: {image_name} ---")
        
        # Load and preprocess image
        original_image, input_tensor = load_and_preprocess_image(image_path, transform, device)
        
        # Get model prediction
        pred_class, pred_prob = get_model_prediction(model, input_tensor)
        print(f"Predicted class: {pred_class} (confidence: {pred_prob:.4f})")
        
        # Compute attributions using selected methods
        results = {}
        
        if 'gradientshap' in methods:
            results['GradientShap'] = compute_gradient_shap(model, input_tensor, pred_class, device)
        
        if 'integratedgradients' in methods:
            results['Integrated Gradients'] = compute_integrated_gradients(model, input_tensor, pred_class, device)
        
        if 'saliency' in methods:
            results['Saliency'] = compute_saliency(model, input_tensor, pred_class, device)
        
        # Save individual visualizations
        for method_name, result in results.items():
            if result:
                filename_suffix = method_name.lower().replace(' ', '_')
                output_path = os.path.join(output_dir, f'{os.path.splitext(image_name)[0]}_{filename_suffix}.png')
                save_attribution_visualization(original_image, result['attributions'], 
                                              method_name, output_path, pred_class, pred_prob)
        
        # Create combined comparison visualization if multiple methods were used
        if len(results) > 1:
            print("\nCreating combined comparison visualization...")
            # Filter out None results
            valid_results = {name: res['attributions'] for name, res in results.items() if res is not None}
            if len(valid_results) > 1:
                comp_output_path = os.path.join(output_dir, f'{os.path.splitext(image_name)[0]}_comparison.png')
                save_comparison_visualization(original_image, valid_results, comp_output_path, 
                                             pred_class, pred_prob, image_name)
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"✓ Analyzed {len(image_files)} images.")
    print(f"✓ Applied {len(methods)} explainability method(s): {', '.join(methods)}.")
    print(f"✓ Generated visualizations saved to: {output_dir}.")
    print("\nKey insights:")
    if 'gradientshap' in methods:
        print("- GradientShap: Captures feature importance using gradient-based Shapley values.")
    if 'integratedgradients' in methods:
        print("- Integrated Gradients: Shows pixel-level attribution by integrating gradients along path.")
    if 'saliency' in methods:
        print("- Saliency: Highlights regions with highest gradient magnitude.")
    print(f"\nAll results saved in: {output_dir}")
