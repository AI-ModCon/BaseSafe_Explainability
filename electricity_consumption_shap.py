#!/usr/bin/env python3
"""
Time-Series Electricity Consumption Prediction with SHAP Explanations
This example demonstrates how to use SHAP to explain predictions from regression models
trained on time-series electricity consumption data.
"""

import shap
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

def generate_synthetic_electricity_data(n_days=365, start_date='2023-01-01'):
    """
    Generate synthetic electricity consumption time-series data
    """
    print("Generating synthetic electricity consumption data...")
    
    # Create date range
    dates = pd.date_range(start=start_date, periods=n_days*24, freq='H')
    
    # Create base consumption pattern
    np.random.seed(42)
    
    # Seasonal patterns
    day_of_year = dates.dayofyear
    hour_of_day = dates.hour
    day_of_week = dates.dayofweek
    
    # Base consumption with seasonal trends
    seasonal_factor = 1 + 0.3 * np.sin(2 * np.pi * day_of_year / 365)  # Yearly seasonality
    daily_pattern = 0.8 + 0.4 * np.sin(2 * np.pi * hour_of_day / 24)  # Daily pattern
    weekly_pattern = 1 + 0.1 * np.sin(2 * np.pi * day_of_week / 7)    # Weekly pattern
    
    # Temperature effect (synthetic)
    temp_base = 20 + 10 * np.sin(2 * np.pi * day_of_year / 365) + np.random.normal(0, 3, len(dates))
    temp_effect = 1 + 0.02 * np.abs(temp_base - 22)  # Higher consumption when temp deviates from 22°C
    
    # Economic activity (weekday vs weekend)
    economic_activity = np.where(day_of_week < 5, 1.2, 0.8)  # Higher on weekdays
    
    # Base consumption
    base_consumption = 1000
    
    # Combine all factors
    consumption = (base_consumption * 
                  seasonal_factor * 
                  daily_pattern * 
                  weekly_pattern * 
                  temp_effect * 
                  economic_activity +
                  np.random.normal(0, 50, len(dates)))  # Add noise
    
    # Create DataFrame
    df = pd.DataFrame({
        'datetime': dates,
        'consumption': consumption,
        'temperature': temp_base,
        'hour': hour_of_day,
        'day_of_week': day_of_week,
        'day_of_year': day_of_year,
        'month': dates.month,
        'is_weekend': (day_of_week >= 5).astype(int)
    })
    
    return df

def create_time_series_features(df, target_col='consumption', lags=[1, 2, 3, 24, 48, 168]):
    """
    Create time-series features including lags, rolling averages, and temporal features
    """
    print("Creating time-series features...")
    
    df_features = df.copy()
    
    # Lag features
    for lag in lags:
        df_features[f'lag_{lag}h'] = df[target_col].shift(lag)
    
    # Rolling averages
    for window in [6, 12, 24, 48]:
        df_features[f'rolling_mean_{window}h'] = df[target_col].rolling(window=window).mean()
        df_features[f'rolling_std_{window}h'] = df[target_col].rolling(window=window).std()
    
    # Temporal features
    df_features['hour_sin'] = np.sin(2 * np.pi * df_features['hour'] / 24)
    df_features['hour_cos'] = np.cos(2 * np.pi * df_features['hour'] / 24)
    df_features['day_of_week_sin'] = np.sin(2 * np.pi * df_features['day_of_week'] / 7)
    df_features['day_of_week_cos'] = np.cos(2 * np.pi * df_features['day_of_week'] / 7)
    df_features['month_sin'] = np.sin(2 * np.pi * df_features['month'] / 12)
    df_features['month_cos'] = np.cos(2 * np.pi * df_features['month'] / 12)
    
    # Temperature features
    df_features['temp_squared'] = df_features['temperature'] ** 2
    df_features['temp_cooling_degree'] = np.maximum(0, df_features['temperature'] - 22)
    df_features['temp_heating_degree'] = np.maximum(0, 22 - df_features['temperature'])
    
    # Drop rows with NaN values (due to lag/rolling features)
    df_features = df_features.dropna()
    
    return df_features

def train_and_explain_models(df_features, target_col='consumption'):
    """
    Train regression models and create SHAP explanations
    """
    print("\nTraining regression models...")
    
    # Prepare features and target
    feature_cols = [col for col in df_features.columns 
                   if col not in ['datetime', target_col]]
    
    X = df_features[feature_cols]
    y = df_features[target_col]
    
    # Split data (preserving temporal order)
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    print(f"Training set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}")
    
    # Train multiple models
    models = {
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
        'Linear Regression': LinearRegression()
    }
    
    results = {}
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        
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
        
        # Create SHAP explainer
        print(f"  Creating SHAP explainer for {name}...")
        
        if name in ['Random Forest', 'Gradient Boosting']:
            # Use TreeExplainer for tree-based models (faster and more accurate)
            explainer = shap.TreeExplainer(model)
            # Use a subset for faster computation
            shap_values = explainer.shap_values(X_test.iloc[:100])
        else:
            # Use KernelExplainer for other models
            # Use a smaller background dataset for faster computation
            explainer = shap.KernelExplainer(model.predict, X_train.iloc[:100])
            shap_values = explainer.shap_values(X_test.iloc[:50])
        
        results[name] = {
            'model': model,
            'explainer': explainer,
            'shap_values': shap_values,
            'X_test_sample': X_test.iloc[:len(shap_values)],
            'y_test_sample': y_test.iloc[:len(shap_values)],
            'metrics': {
                'train_mse': train_mse,
                'test_mse': test_mse,
                'train_r2': train_r2,
                'test_r2': test_r2
            }
        }
    
    return results, X_test, y_test

def analyze_shap_results(results):
    """
    Analyze and display SHAP results
    """
    print("\n" + "="*60)
    print("SHAP ANALYSIS RESULTS")
    print("="*60)
    
    for model_name, result in results.items():
        print(f"\n--- {model_name} ---")
        
        shap_values = result['shap_values']
        X_sample = result['X_test_sample']
        
        # Feature importance (mean absolute SHAP values)
        feature_importance = pd.DataFrame({
            'feature': X_sample.columns,
            'importance': np.mean(np.abs(shap_values), axis=0)
        }).sort_values('importance', ascending=False)
        
        print("\nTop 10 Most Important Features:")
        print(feature_importance.head(10).to_string(index=False))
        
        # Summary statistics
        print(f"\nSHAP Values Summary:")
        print(f"  Shape: {shap_values.shape}")
        print(f"  Mean absolute SHAP value: {np.mean(np.abs(shap_values)):.2f}")
        print(f"  Max SHAP value: {np.max(shap_values):.2f}")
        print(f"  Min SHAP value: {np.min(shap_values):.2f}")

def demonstrate_predictions_with_explanations(results, model_name='Random Forest'):
    """
    Demonstrate individual predictions with SHAP explanations
    """
    if model_name not in results:
        print(f"Model {model_name} not found in results")
        return
    
    result = results[model_name]
    model = result['model']
    shap_values = result['shap_values']
    X_sample = result['X_test_sample']
    y_sample = result['y_test_sample']
    
    print(f"\n" + "="*60)
    print(f"INDIVIDUAL PREDICTION EXPLANATIONS - {model_name}")
    print("="*60)
    
    # Show explanations for first 5 predictions
    for i in range(min(5, len(shap_values))):
        actual = y_sample.iloc[i]
        predicted = model.predict(X_sample.iloc[i:i+1])[0]
        
        print(f"\nSample {i+1}:")
        print(f"  Actual consumption: {actual:.2f} kWh")
        print(f"  Predicted consumption: {predicted:.2f} kWh")
        print(f"  Prediction error: {abs(actual - predicted):.2f} kWh")
        
        # Get feature contributions for this prediction
        feature_contributions = pd.DataFrame({
            'feature': X_sample.columns,
            'value': X_sample.iloc[i].values,
            'shap_value': shap_values[i]
        }).sort_values('shap_value', key=abs, ascending=False)
        
        print(f"  Top 5 contributing features:")
        for j in range(min(5, len(feature_contributions))):
            row = feature_contributions.iloc[j]
            direction = "increases" if row['shap_value'] > 0 else "decreases"
            print(f"    {row['feature']}: {row['value']:.2f} -> {direction} prediction by {abs(row['shap_value']):.2f}")

def main():
    print("Time-Series Electricity Consumption Prediction with SHAP")
    print("="*60)
    
    # Generate synthetic data
    df = generate_synthetic_electricity_data(n_days=365)
    print(f"Generated {len(df)} hourly electricity consumption records")
    
    # Create time-series features
    df_features = create_time_series_features(df)
    print(f"Created dataset with {len(df_features)} samples and {len(df_features.columns)-2} features")
    
    # Train models and create explanations
    results, X_test, y_test = train_and_explain_models(df_features)
    
    # Analyze SHAP results
    analyze_shap_results(results)
    
    # Demonstrate individual predictions
    demonstrate_predictions_with_explanations(results, 'Random Forest')
    
    print(f"\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print("✓ Successfully created time-series electricity consumption dataset")
    print("✓ Engineered temporal features (lags, rolling statistics, cyclical features)")
    print("✓ Trained multiple regression models")
    print("✓ Generated SHAP explanations for model predictions")
    print("✓ Identified key features driving electricity consumption predictions")
    print("\nKey insights:")
    print("- Lag features (recent consumption) are typically most important")
    print("- Temperature-based features significantly impact consumption")
    print("- Time-of-day and day-of-week patterns are crucial")
    print("- SHAP helps identify which factors drive high/low consumption predictions")

if __name__ == "__main__":
    main()