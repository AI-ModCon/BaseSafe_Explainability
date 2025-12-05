#!/usr/bin/env python3
"""
Model Training and Management Helpers

This module provides functions for:
- Creating regression models
- Training and evaluating models
"""

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


def create_models(models=None):
    """
    Create regression models for training.
    
    Args:
        models (list): List of model names to create. 
                      Options: 'randomforest', 'gradientboosting', 'linear'
                      None = all models
                      
    Returns:
        dict: Dictionary of {name: model} pairs
    """
    available_models = {
        'randomforest': ('Random Forest', RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)),
        'gradientboosting': ('Gradient Boosting', GradientBoostingRegressor(n_estimators=100, random_state=42)),
        'linear': ('Linear Regression', LinearRegression())
    }
    
    if models is None:
        models = list(available_models.keys())
    
    # Validate model names
    models = [m.lower() for m in models]
    invalid = [m for m in models if m not in available_models]
    if invalid:
        print(f"Warning: Invalid model names ignored: {invalid}")
        models = [m for m in models if m in available_models]
    
    if not models:
        print("No valid models specified. Using all models.")
        models = list(available_models.keys())
    
    return {available_models[m][0]: available_models[m][1] for m in models}


def train_model(model, X_train, y_train, X_test, y_test, model_name):
    """
    Train a single model and evaluate performance.
    
    Args:
        model: Sklearn model instance
        X_train, y_train: Training data
        X_test, y_test: Test data
        model_name (str): Name of the model
        
    Returns:
        dict: {
            'model': trained model,
            'metrics': dict of performance metrics
        }
    """
    print(f"\nTraining {model_name}...")
    
    # Train model
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    # Calculate metrics
    train_mse = mean_squared_error(y_train, y_pred_train)
    test_mse = mean_squared_error(y_test, y_pred_test)
    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)
    
    print(f"  Train MSE: {train_mse:.2f}, R²: {train_r2:.4f}")
    print(f"  Test MSE: {test_mse:.2f}, R²: {test_r2:.4f}")
    
    return {
        'model': model,
        'metrics': {
            'train_mse': train_mse,
            'test_mse': test_mse,
            'train_r2': train_r2,
            'test_r2': test_r2
        }
    }
