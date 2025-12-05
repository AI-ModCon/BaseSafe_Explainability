#!/usr/bin/env python3
"""
Time-Series Data Generation and Feature Engineering Helpers

This module provides helper functions for:
- Generating synthetic time-series electricity consumption data
- Creating time-series features (lags, rolling statistics, temporal features)
- Preparing train/test splits preserving temporal order
"""

import numpy as np
import pandas as pd


def generate_synthetic_electricity_data(n_days=365, start_date='2023-01-01'):
    """
    Generate synthetic electricity consumption time-series data.
    
    Args:
        n_days (int): Number of days to generate
        start_date (str): Start date for the time series
        
    Returns:
        pd.DataFrame: DataFrame with hourly electricity consumption data
    """
    print("Generating synthetic electricity consumption data...")
    
    # Create date range using numpy
    dates = pd.to_datetime(np.arange(n_days*24, dtype='timedelta64[h]') + np.datetime64(start_date))
    
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
    Create time-series features including lags, rolling averages, and temporal features.
    
    Args:
        df (pd.DataFrame): Input dataframe with time-series data
        target_col (str): Name of target column to create lag features for
        lags (list): List of lag values to create
        
    Returns:
        pd.DataFrame: DataFrame with engineered features
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
    
    # Temporal features (cyclical encoding)
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


def prepare_train_test_split(df_features, target_col='consumption', test_size=0.2):
    """
    Prepare features and target, split data preserving temporal order.
    
    Args:
        df_features (pd.DataFrame): DataFrame with features
        target_col (str): Name of target column
        test_size (float): Proportion of data for testing
        
    Returns:
        tuple: (X_train, X_test, y_train, y_test, feature_cols)
    """
    # Prepare features and target
    feature_cols = [col for col in df_features.columns 
                   if col not in ['datetime', target_col]]
    
    X = df_features[feature_cols]
    y = df_features[target_col]
    
    # Split data (preserving temporal order)
    split_idx = int(len(X) * (1 - test_size))
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    print(f"Training set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}")
    
    return X_train, X_test, y_train, y_test, feature_cols
