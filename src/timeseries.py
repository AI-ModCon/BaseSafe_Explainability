#!/usr/bin/env python3
"""
Time-Series Electricity Consumption Prediction with SHAP Explanations

This module focuses on explainability functions for time-series models:
- Training regression models
- Computing SHAP explanations
- Visualizing and analyzing SHAP results
- Individual prediction explanations

Data generation and feature engineering are in helpers/timeseries_data.py
"""

import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Import data generation and feature engineering helpers
from helpers.timeseries_data import (
    generate_synthetic_electricity_data,
    create_time_series_features,
    prepare_train_test_split
)

# Import model training helpers
from helpers.model_training import (
    create_models,
    train_model
)


def compute_shap_values(model, model_name, X_train, X_test, num_samples=100):
    """
    Compute SHAP values for a trained model.
    
    Args:
        model: Trained sklearn model
        model_name (str): Name of the model
        X_train (pd.DataFrame): Training data for background
        X_test (pd.DataFrame): Test data to explain
        num_samples (int): Number of test samples to explain
        
    Returns:
        dict: {
            'explainer': SHAP explainer,
            'shap_values': numpy array of SHAP values,
            'X_sample': DataFrame of explained samples
        } or None on error
    """
    print(f"  Creating SHAP explainer for {model_name}...")
    
    try:
        if model_name in ['Random Forest', 'Gradient Boosting']:
            # Use TreeExplainer for tree-based models (faster and more accurate)
            explainer = shap.TreeExplainer(model)
            sample_size = min(num_samples, len(X_test))
            shap_values = explainer.shap_values(X_test.iloc[:sample_size])
        else:
            # Use KernelExplainer for other models
            # Use a smaller background dataset for faster computation
            background_size = min(100, len(X_train))
            explainer = shap.KernelExplainer(model.predict, X_train.iloc[:background_size])
            sample_size = min(num_samples // 2, len(X_test))  # Smaller for KernelExplainer
            shap_values = explainer.shap_values(X_test.iloc[:sample_size])
        
        return {
            'explainer': explainer,
            'shap_values': shap_values,
            'X_sample': X_test.iloc[:len(shap_values)]
        }
    except Exception as e:
        print(f"  Error computing SHAP values: {e}")
        return None

def compute_feature_importance(shap_values, feature_names):
    """
    Compute feature importance from SHAP values.
    
    Args:
        shap_values (np.ndarray): SHAP values array
        feature_names (list): List of feature names
        
    Returns:
        pd.DataFrame: Feature importance sorted by importance
    """
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': np.mean(np.abs(shap_values), axis=0)
    }).sort_values('importance', ascending=False)
    
    return feature_importance

def get_shap_summary_stats(shap_values):
    """
    Get summary statistics for SHAP values.
    
    Args:
        shap_values (np.ndarray): SHAP values array
        
    Returns:
        dict: Summary statistics
    """
    return {
        'shape': shap_values.shape,
        'mean_abs': np.mean(np.abs(shap_values)),
        'max': np.max(shap_values),
        'min': np.min(shap_values),
        'std': np.std(shap_values)
    }

def save_shap_summary_plot(shap_values, X_sample, output_path, model_name):
    """
    Create and save SHAP summary plot.
    
    Args:
        shap_values (np.ndarray): SHAP values
        X_sample (pd.DataFrame): Sample data
        output_path (str): Path to save plot
        model_name (str): Name of the model
        
    Returns:
        str: Path to saved plot or None on error
    """
    try:
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, X_sample, show=False)
        plt.title(f'SHAP Summary Plot - {model_name}')
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        return output_path
    except Exception as e:
        print(f"Error creating SHAP summary plot: {e}")
        return None

def save_shap_waterfall_plot(explainer, shap_values, X_sample, output_path, sample_idx=0):
    """
    Create and save SHAP waterfall plot for a single prediction.
    
    Args:
        explainer: SHAP explainer object
        shap_values (np.ndarray): SHAP values
        X_sample (pd.DataFrame): Sample data
        output_path (str): Path to save plot
        sample_idx (int): Index of sample to explain
        
    Returns:
        str: Path to saved plot or None on error
    """
    try:
        plt.figure(figsize=(10, 6))
        
        # For SHAP v0.40+, use Explanation object
        if hasattr(shap, 'Explanation'):
            explanation = shap.Explanation(
                values=shap_values[sample_idx],
                base_values=explainer.expected_value if hasattr(explainer, 'expected_value') else np.mean(shap_values),
                data=X_sample.iloc[sample_idx].values,
                feature_names=X_sample.columns.tolist()
            )
            shap.waterfall_plot(explanation, show=False)
        else:
            # Fallback for older SHAP versions
            shap.waterfall_plot(
                shap.Explanation(
                    values=shap_values[sample_idx],
                    data=X_sample.iloc[sample_idx].values
                ),
                show=False
            )
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        return output_path
    except Exception as e:
        print(f"Error creating SHAP waterfall plot: {e}")
        return None

def save_feature_importance_plot(feature_importance, output_path, top_n=15):
    """
    Create and save feature importance bar plot.
    
    Args:
        feature_importance (pd.DataFrame): Feature importance dataframe
        output_path (str): Path to save plot
        top_n (int): Number of top features to show
        
    Returns:
        str: Path to saved plot or None on error
    """
    try:
        plt.figure(figsize=(10, 6))
        top_features = feature_importance.head(top_n)
        plt.barh(range(len(top_features)), top_features['importance'])
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.xlabel('Mean |SHAP value|')
        plt.title(f'Top {top_n} Feature Importance')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        return output_path
    except Exception as e:
        print(f"Error creating feature importance plot: {e}")
        return None

def get_prediction_explanation(model, shap_values, X_sample, y_sample, sample_idx=0, top_n=5):
    """
    Get explanation for a single prediction.
    
    Args:
        model: Trained model
        shap_values (np.ndarray): SHAP values
        X_sample (pd.DataFrame): Sample data
        y_sample (pd.Series): True values
        sample_idx (int): Index of sample to explain
        top_n (int): Number of top features to return
        
    Returns:
        dict: {
            'actual': actual value,
            'predicted': predicted value,
            'error': prediction error,
            'top_features': DataFrame of top contributing features
        }
    """
    actual = y_sample.iloc[sample_idx]
    predicted = model.predict(X_sample.iloc[sample_idx:sample_idx+1])[0]
    
    # Get feature contributions for this prediction
    feature_contributions = pd.DataFrame({
        'feature': X_sample.columns,
        'value': X_sample.iloc[sample_idx].values,
        'shap_value': shap_values[sample_idx]
    }).sort_values('shap_value', key=abs, ascending=False)
    
    return {
        'actual': actual,
        'predicted': predicted,
        'error': abs(actual - predicted),
        'top_features': feature_contributions.head(top_n)
    }

def save_results_summary(results, output_path):
    """
    Save results summary to a text file.
    
    Args:
        results (dict): Results dictionary with keys:
                       - 'models': list of model names
                       - 'training_results': dict of training metrics
                       - 'shap_results': dict of SHAP statistics
                       - 'feature_importance': dict of feature importance dataframes
        output_path (str): Path to save summary
        
    Returns:
        str: Path to saved file or None on error
    """
    try:
        with open(output_path, 'w') as f:
            f.write("="*60 + "\n")
            f.write("TIME-SERIES SHAP ANALYSIS RESULTS\n")
            f.write("="*60 + "\n\n")
            
            # Check if this is the new format (from notebook) or old format (from explain_timeseries_predictions)
            # New format has 'models' as a list, old format has model names as top-level keys
            if 'models' in results and isinstance(results['models'], list):
                # New format from notebook
                models = results['models']
                training_results = results['training_results']
                shap_results = results.get('shap_results', {})
                feature_importance = results.get('feature_importance', {})
                
                for model_name in models:
                    f.write(f"\n--- {model_name} ---\n")
                    
                    # Metrics
                    try:
                        if model_name in training_results:
                            metrics = training_results[model_name]['metrics']
                            f.write(f"\nPerformance Metrics:\n")
                            f.write(f"  Train MSE: {metrics['train_mse']:.2f}, R²: {metrics['train_r2']:.4f}\n")
                            f.write(f"  Test MSE: {metrics['test_mse']:.2f}, R²: {metrics['test_r2']:.4f}\n")
                    except Exception as e:
                        f.write(f"\nError writing metrics: {e}\n")
                    
                    # SHAP statistics
                    try:
                        if model_name in shap_results:
                            stats = shap_results[model_name]
                            f.write(f"\nSHAP Values Summary:\n")
                            f.write(f"  Shape: {stats['shape']}\n")
                            f.write(f"  Mean absolute SHAP value: {stats['mean_abs']:.4f}\n")
                            f.write(f"  Standard deviation: {stats['std']:.4f}\n")
                    except Exception as e:
                        f.write(f"\nError writing SHAP stats: {e}\n")
                    
                    # Feature importance
                    try:
                        if model_name in feature_importance:
                            feat_imp = feature_importance[model_name]
                            f.write(f"\nTop 10 Most Important Features:\n")
                            # Convert dict back to DataFrame if needed
                            if isinstance(feat_imp, dict):
                                import pandas as pd
                                # The dict has 'feature' and 'importance' as keys mapping to lists
                                feat_df = pd.DataFrame(feat_imp)
                                if 'feature' in feat_df.columns and 'importance' in feat_df.columns:
                                    feat_df = feat_df.sort_values('importance', ascending=False)
                                    for idx in range(min(10, len(feat_df))):
                                        row = feat_df.iloc[idx]
                                        f.write(f"  {row['feature']}: {row['importance']:.6f}\n")
                            else:
                                f.write(feat_imp.head(10).to_string(index=False))
                                f.write("\n")
                    except Exception as e:
                        f.write(f"\nError writing feature importance: {e}\n")
                        import traceback
                        f.write(f"{traceback.format_exc()}\n")
                    
                    f.write("\n" + "="*60 + "\n")
            else:
                # Old format from explain_timeseries_predictions
                for model_name, result in results.items():
                    f.write(f"\n--- {model_name} ---\n")
                    
                    # Metrics
                    metrics = result['metrics']
                    f.write(f"\nPerformance Metrics:\n")
                    f.write(f"  Train MSE: {metrics['train_mse']:.2f}, R²: {metrics['train_r2']:.4f}\n")
                    f.write(f"  Test MSE: {metrics['test_mse']:.2f}, R²: {metrics['test_r2']:.4f}\n")
                    
                    if result.get('shap_values') is not None:
                        shap_values = result['shap_values']
                        X_sample = result['X_sample']
                        
                        # Feature importance
                        feature_importance_df = compute_feature_importance(shap_values, X_sample.columns)
                        f.write(f"\nTop 10 Most Important Features:\n")
                        f.write(feature_importance_df.head(10).to_string(index=False))
                        f.write("\n")
                        
                        # SHAP statistics
                        stats = get_shap_summary_stats(shap_values)
                        f.write(f"\nSHAP Values Summary:\n")
                        f.write(f"  Shape: {stats['shape']}\n")
                        f.write(f"  Mean absolute SHAP value: {stats['mean_abs']:.2f}\n")
                        f.write(f"  Max SHAP value: {stats['max']:.2f}\n")
                        f.write(f"  Min SHAP value: {stats['min']:.2f}\n")
                    
                    f.write("\n" + "="*60 + "\n")
        
        return output_path
    except Exception as e:
        print(f"Error saving results summary: {e}")
        return None

def explain_timeseries_predictions(
    data_path=None,
    output_dir='./timeseries_shap_outputs',
    n_days=365,
    models=None,
    num_samples=100,
    test_size=0.2,
    target_col='consumption'
):
    """
    Complete workflow: generate/load data, train models, compute SHAP explanations.
    
    Args:
        data_path (str): Path to CSV file with time-series data (None = generate synthetic)
        output_dir (str): Directory to save outputs
        n_days (int): Number of days for synthetic data generation
        models (list): List of model names ['randomforest', 'gradientboosting', 'linear']
        num_samples (int): Number of test samples to explain
        test_size (float): Proportion of data for testing
        target_col (str): Name of target column
        
    Returns:
        dict: Results for each model with trained model, SHAP values, and metrics
    """
    print("Time-Series Electricity Consumption Prediction with SHAP")
    print("="*60)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load or generate data
    if data_path and os.path.exists(data_path):
        print(f"Loading data from {data_path}...")
        df = pd.read_csv(data_path)
        if 'datetime' in df.columns:
            df['datetime'] = pd.to_datetime(df['datetime'])
    else:
        if data_path:
            print(f"Warning: {data_path} not found. Generating synthetic data...")
        df = generate_synthetic_electricity_data(n_days=n_days)
    
    print(f"Dataset: {len(df)} hourly electricity consumption records")
    
    # Create time-series features
    df_features = create_time_series_features(df, target_col=target_col)
    print(f"Features: {len(df_features)} samples with {len(df_features.columns)-2} features")
    
    # Prepare train/test split
    X_train, X_test, y_train, y_test, feature_cols = prepare_train_test_split(
        df_features, target_col=target_col, test_size=test_size
    )
    
    # Create models
    model_dict = create_models(models)
    print(f"\nTraining {len(model_dict)} model(s)...")
    
    # Train models and compute SHAP values
    results = {}
    for model_name, model in model_dict.items():
        # Train model
        train_result = train_model(model, X_train, y_train, X_test, y_test, model_name)
        
        # Compute SHAP values
        shap_result = compute_shap_values(
            train_result['model'],
            model_name,
            X_train,
            X_test,
            num_samples=num_samples
        )
        
        # Combine results
        results[model_name] = {
            'model': train_result['model'],
            'metrics': train_result['metrics'],
            'y_test': y_test
        }
        
        if shap_result:
            results[model_name].update({
                'explainer': shap_result['explainer'],
                'shap_values': shap_result['shap_values'],
                'X_sample': shap_result['X_sample'],
                'y_sample': y_test.iloc[:len(shap_result['shap_values'])]
            })
            
            # Create visualizations
            model_output_dir = os.path.join(output_dir, model_name.replace(' ', '_').lower())
            os.makedirs(model_output_dir, exist_ok=True)
            
            # Save SHAP summary plot
            summary_plot_path = os.path.join(model_output_dir, 'shap_summary.png')
            save_shap_summary_plot(
                shap_result['shap_values'],
                shap_result['X_sample'],
                summary_plot_path,
                model_name
            )
            
            # Save feature importance plot
            feature_importance = compute_feature_importance(
                shap_result['shap_values'],
                shap_result['X_sample'].columns
            )
            importance_plot_path = os.path.join(model_output_dir, 'feature_importance.png')
            save_feature_importance_plot(feature_importance, importance_plot_path)
            
            # Save waterfall plot for first prediction
            waterfall_plot_path = os.path.join(model_output_dir, 'shap_waterfall_sample_0.png')
            save_shap_waterfall_plot(
                shap_result['explainer'],
                shap_result['shap_values'],
                shap_result['X_sample'],
                waterfall_plot_path,
                sample_idx=0
            )
            
            print(f"  Saved visualizations to {model_output_dir}/")
    
    # Save results summary
    summary_path = os.path.join(output_dir, 'results_summary.txt')
    save_results_summary(results, summary_path)
    print(f"\nResults summary saved to {summary_path}")
    
    print(f"\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"✓ Processed {len(df_features)} time-series samples")
    print(f"✓ Trained {len(results)} model(s)")
    print(f"✓ Generated SHAP explanations")
    print(f"✓ Saved outputs to {output_dir}/")