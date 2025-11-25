import argparse
import torch
import torch.nn as nn
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from transformers import AutoImageProcessor, AutoModelForImageClassification

# Add BaseSafe_RedTeaming to path for poisoningaux import
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'BaseSafe_RedTeaming'))

try:
    from poisoningaux import poison_data, load_model, inference
    POISONINGAUX_AVAILABLE = True
except ImportError:
    POISONINGAUX_AVAILABLE = False
    print("Warning: poisoningaux module not available. Poisoning attack function will not work.")
from captum.attr import GradientShap, IntegratedGradients, Saliency
from captum.attr import visualization as viz

def poisoning_attack(data_path, backdoor_path, model_path, save=False, outdir='./state_dicts', filename='poisoned_model.pt'):
    """
    Poison a Hugging Face image classifier. Apply the dirty label backdoor attack (DLBD) on the provided data and fine-tune the provided model.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    x_poison, y_poison, is_poison = poison_data(data_path, backdoor_path)
    # Use 10 classes for poisoning attack
    hf_model = load_model(model_path, num_labels=10)

    if save:
        os.makedirs(outdir, exist_ok=True)
        poison_checkpoint_path = os.path.join(outdir, filename)
        try:
            assert os.path.isfile(poison_checkpoint_path)
            hf_model.model.load_state_dict(torch.load(poison_checkpoint_path, map_location=device))
        except:
            hf_model.fit(x_poison, y_poison, nb_epochs=2)
            torch.save(hf_model.model.state_dict(), poison_checkpoint_path)
    else:
        hf_model.fit(x_poison, y_poison, nb_epochs=2)

    clean_acc = inference(x_poison[~is_poison], y_poison[~is_poison], hf_model)
    print(f'clean accuracy: {clean_acc}')

    poison_acc = inference(x_poison[is_poison], y_poison[is_poison], hf_model)
    print(f'poison success rate: {poison_acc}')


def explain_model_predictions(model_path, data_path, output_dir='./explainability_outputs', num_samples=5):
    """
    Apply explainability methods (GradientShap, IntegratedGradients, Saliency) to a ResNet model.
    
    Args:
        model_path: Path or name of the Hugging Face model (e.g., 'microsoft/resnet-50')
        data_path: Path to the directory containing images
        output_dir: Directory to save visualization outputs
        num_samples: Number of images to analyze
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load model using poisoningaux with default num_labels=None for full class support
    if not POISONINGAUX_AVAILABLE:
        raise ImportError("poisoningaux module is required for explain_model_predictions function")
    
    print(f"\nLoading model: {model_path}")
    hf_model = load_model(model_path)  # Uses default num_labels=None for original classes
    base_model = hf_model.model
    base_model.to(device)
    base_model.eval()
    
    # Create a wrapper model that returns logits for Captum compatibility
    class ModelWrapper(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model
            
        def forward(self, x):
            output = self.model(x)
            # Check if output is already a tensor (logits) or needs .logits attribute
            if isinstance(output, torch.Tensor):
                return output
            return output.logits
    
    model = ModelWrapper(base_model)
    model.eval()
    
    # Get image files
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif')
    image_files = [os.path.join(data_path, f) for f in os.listdir(data_path) 
                   if f.lower().endswith(image_extensions)][:num_samples]
    
    if not image_files:
        print(f"No images found in {data_path}")
        return
    
    print(f"\nFound {len(image_files)} images to analyze")
    
    # Preprocessing transform
    processor = AutoImageProcessor.from_pretrained(model_path)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=processor.image_mean, std=processor.image_std)
    ])
    
    # Initialize explainability methods
    gradient_shap = GradientShap(model)
    integrated_gradients = IntegratedGradients(model)
    saliency = Saliency(model)
    
    print("\n" + "="*80)
    print("EXPLAINABILITY ANALYSIS RESULTS")
    print("="*80)
    
    # Process each image
    for idx, image_path in enumerate(image_files):
        image_name = os.path.basename(image_path)
        print(f"\n--- Image {idx+1}/{len(image_files)}: {image_name} ---")
        
        # Load and preprocess image
        original_image = Image.open(image_path).convert('RGB')
        input_tensor = transform(original_image).unsqueeze(0).to(device)
        input_tensor.requires_grad = True
        
        # Get model prediction
        with torch.no_grad():
            output = model(input_tensor)
            probs = torch.nn.functional.softmax(output, dim=1)
            pred_class = output.argmax(dim=1).item()
            pred_prob = probs[0, pred_class].item()
        
        print(f"Predicted class: {pred_class} (confidence: {pred_prob:.4f})")
        
        # Create baseline (black image)
        baseline = torch.zeros_like(input_tensor)
        
        # Method 1: GradientShap
        print("\nComputing GradientShap attributions...")
        try:
            # Generate random baselines for GradientShap
            baselines = torch.randn(10, *input_tensor.shape[1:]).to(device) * 0.1
            gs_attr = gradient_shap.attribute(input_tensor, 
                                             baselines=baselines,
                                             target=pred_class,
                                             n_samples=50)
            gs_attr_np = gs_attr.squeeze().cpu().detach().numpy()
            gs_attr_sum = np.sum(np.abs(gs_attr_np), axis=0)
            
            print(f"GradientShap attribution range: [{gs_attr_sum.min():.4f}, {gs_attr_sum.max():.4f}]")
            print(f"GradientShap mean absolute attribution: {np.mean(np.abs(gs_attr_sum)):.4f}")
            
            # Save GradientShap visualization
            gs_output_path = os.path.join(output_dir, f'{os.path.splitext(image_name)[0]}_gradientshap.png')
            fig, ax = plt.subplots(1, 2, figsize=(12, 5))
            
            # Original image
            ax[0].imshow(original_image)
            ax[0].set_title(f'Original Image\nPred: Class {pred_class} ({pred_prob:.2%})')
            ax[0].axis('off')
            
            # GradientShap heatmap
            im = ax[1].imshow(gs_attr_sum, cmap='hot')
            ax[1].set_title('GradientShap Attribution')
            ax[1].axis('off')
            plt.colorbar(im, ax=ax[1], fraction=0.046, pad=0.04)
            
            plt.tight_layout()
            plt.savefig(gs_output_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"    Saved: {gs_output_path}")
            
        except Exception as e:
            print(f"Error computing GradientShap: {e}")
        
        # Method 2: Integrated Gradients
        print("\nComputing Integrated Gradients attributions...")
        try:
            ig_attr = integrated_gradients.attribute(input_tensor,
                                                    baselines=baseline,
                                                    target=pred_class,
                                                    n_steps=50)
            ig_attr_np = ig_attr.squeeze().cpu().detach().numpy()
            ig_attr_sum = np.sum(np.abs(ig_attr_np), axis=0)
            
            print(f"IntegratedGradients attribution range: [{ig_attr_sum.min():.4f}, {ig_attr_sum.max():.4f}]")
            print(f"IntegratedGradients mean absolute attribution: {np.mean(np.abs(ig_attr_sum)):.4f}")
            
            # Save Integrated Gradients visualization
            ig_output_path = os.path.join(output_dir, f'{os.path.splitext(image_name)[0]}_integrated_gradients.png')
            fig, ax = plt.subplots(1, 2, figsize=(12, 5))
            
            # Original image
            ax[0].imshow(original_image)
            ax[0].set_title(f'Original Image\nPred: Class {pred_class} ({pred_prob:.2%})')
            ax[0].axis('off')
            
            # Integrated Gradients heatmap
            im = ax[1].imshow(ig_attr_sum, cmap='hot')
            ax[1].set_title('Integrated Gradients Attribution')
            ax[1].axis('off')
            plt.colorbar(im, ax=ax[1], fraction=0.046, pad=0.04)
            
            plt.tight_layout()
            plt.savefig(ig_output_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"Saved: {ig_output_path}")
            
        except Exception as e:
            print(f"Error computing Integrated Gradients: {e}")
        
        # Method 3: Saliency
        print("\nComputing Saliency attributions...")
        try:
            sal_attr = saliency.attribute(input_tensor, target=pred_class)
            sal_attr_np = sal_attr.squeeze().cpu().detach().numpy()
            sal_attr_sum = np.sum(np.abs(sal_attr_np), axis=0)
            
            print(f"Saliency attribution range: [{sal_attr_sum.min():.4f}, {sal_attr_sum.max():.4f}]")
            print(f"Saliency mean absolute attribution: {np.mean(np.abs(sal_attr_sum)):.4f}")
            
            # Save Saliency visualization
            sal_output_path = os.path.join(output_dir, f'{os.path.splitext(image_name)[0]}_saliency.png')
            fig, ax = plt.subplots(1, 2, figsize=(12, 5))
            
            # Original image
            ax[0].imshow(original_image)
            ax[0].set_title(f'Original Image\nPred: Class {pred_class} ({pred_prob:.2%})')
            ax[0].axis('off')
            
            # Saliency heatmap
            im = ax[1].imshow(sal_attr_sum, cmap='hot')
            ax[1].set_title('Saliency Attribution')
            ax[1].axis('off')
            plt.colorbar(im, ax=ax[1], fraction=0.046, pad=0.04)
            
            plt.tight_layout()
            plt.savefig(sal_output_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"    Saved: {sal_output_path}")
            
        except Exception as e:
            print(f"Error computing Saliency: {e}")
        
        # Create combined comparison visualization
        print("\nCreating combined comparison visualization...")
        try:
            comp_output_path = os.path.join(output_dir, f'{os.path.splitext(image_name)[0]}_comparison.png')
            fig, axes = plt.subplots(2, 2, figsize=(14, 12))
            
            # Original image
            axes[0, 0].imshow(original_image)
            axes[0, 0].set_title(f'Original Image\nPred: Class {pred_class} ({pred_prob:.2%})', fontsize=12, fontweight='bold')
            axes[0, 0].axis('off')
            
            # GradientShap
            im1 = axes[0, 1].imshow(gs_attr_sum, cmap='hot')
            axes[0, 1].set_title('GradientShap Attribution', fontsize=12, fontweight='bold')
            axes[0, 1].axis('off')
            plt.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04)
            
            # Integrated Gradients
            im2 = axes[1, 0].imshow(ig_attr_sum, cmap='hot')
            axes[1, 0].set_title('Integrated Gradients Attribution', fontsize=12, fontweight='bold')
            axes[1, 0].axis('off')
            plt.colorbar(im2, ax=axes[1, 0], fraction=0.046, pad=0.04)
            
            # Saliency
            im3 = axes[1, 1].imshow(sal_attr_sum, cmap='hot')
            axes[1, 1].set_title('Saliency Attribution', fontsize=12, fontweight='bold')
            axes[1, 1].axis('off')
            plt.colorbar(im3, ax=axes[1, 1], fraction=0.046, pad=0.04)
            
            plt.suptitle(f'Explainability Methods Comparison: {image_name}', fontsize=14, fontweight='bold', y=0.98)
            plt.tight_layout()
            plt.savefig(comp_output_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"Saved: {comp_output_path}")
            
        except Exception as e:
            print(f"Error creating comparison: {e}")
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"✓ Analyzed {len(image_files)} images.")
    print(f"✓ Applied 3 explainability methods: GradientShap, Integrated Gradients, Saliency.")
    print(f"✓ Generated visualizations saved to: {output_dir}.")
    print("\nKey insights:")
    print("- GradientShap: Captures feature importance using gradient-based Shapley values.")
    print("- Integrated Gradients: Shows pixel-level attribution by integrating gradients along path.")
    print("- Saliency: Highlights regions with highest gradient magnitude.")
    print(f"\nAll results saved in: {output_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    # Subcommands for different operations
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Poisoning attack command
    poison_parser = subparsers.add_parser('poison', help='Run poisoning attack')
    poison_parser.add_argument('-data', type=str, default="./data/imagenette2-320/train")
    poison_parser.add_argument('-backdoor', type=str, default="./data/backdoors/baby-on-board.png")
    poison_parser.add_argument('-model', type=str, default="microsoft/resnet-50")
    poison_parser.add_argument('-save', action='store_true')
    poison_parser.add_argument('-outdir', type=str, default='./state_dicts')
    poison_parser.add_argument('-filename', type=str, default='poisoned_model.pt')
    
    # Explainability command
    explain_parser = subparsers.add_parser('explain', help='Run explainability analysis')
    explain_parser.add_argument('-data', type=str, default="./data/imagenette2-320/train", required=True, help='Path to directory containing images')
    explain_parser.add_argument('-model', type=str, default='microsoft/resnet-50', help='Hugging Face model name or path')
    explain_parser.add_argument('-outdir', type=str, default='./explainability_outputs', help='Output directory for visualizations')
    explain_parser.add_argument('-num_samples', type=int, default=5, help='Number of images to analyze')
    
    args = parser.parse_args()
    
    if args.command == 'poison':
        poisoning_attack(args.data,
                        args.backdoor,
                        args.model,
                        save=args.save,
                        outdir=args.outdir,
                        filename=args.filename)
    elif args.command == 'explain':
        explain_model_predictions(args.model,
                                 args.data,
                                 output_dir=args.outdir,
                                 num_samples=args.num_samples)
    else:
        parser.print_help()