"""Helper functions for time-series and image processing, feature engineering, and model training."""

# Time-series data helpers
from .timeseries_data import (
    generate_synthetic_electricity_data,
    create_time_series_features,
    prepare_train_test_split
)

# Image utilities helpers
from .image_utils import (
    ModelWrapper,
    load_model,
    get_image_files,
    create_transform,
    load_and_preprocess_image,
    get_model_prediction
)

# Lazy imports for model training to avoid importing sklearn if not needed
def __getattr__(name):
    if name in ['create_models', 'train_model']:
        from .model_training import create_models, train_model
        if name == 'create_models':
            return create_models
        return train_model
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    # Time-series helpers
    'generate_synthetic_electricity_data',
    'create_time_series_features',
    'prepare_train_test_split',
    # Image helpers
    'ModelWrapper',
    'load_model',
    'get_image_files',
    'create_transform',
    'load_and_preprocess_image',
    'get_model_prediction',
    # Model training helpers
    'create_models',
    'train_model'
]
