# Explainability Analysis with SHAP and Captum

## Setup

```bash
# Navigate to the explainability directory
cd path/to/code-safe/BaseSafe_Explainability

# Activate virtual environment (located in parent directory)
source ../venv/bin/activate  # On Windows: ..\venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Running Examples

### Electricity Consumption SHAP Analysis
```bash
jupyter notebook electricity_consumption_shap.ipynb
```

### Captum GradientSHAP Example

```bash
jupyter notebook captum_gradientshap_example.ipynb
```

Configuration:
- Set `DATA_SOURCE = 'dataset'` and `DATASET_TYPE = 'materials'`
- Run download helper cell to populate `data/materials_samples/` and `data/generic_samples/`
- Switch models: Set `MODEL_TYPE = "huggingface"` (ViT) or `"torchvision"` (ResNet18)

## Files

- `electricity_consumption_shap.ipynb` - Time-series SHAP analysis
- `captum_gradientshap_example.ipynb` - Deep learning GradientSHAP with ViT/ResNet18
- `electricity_consumption_shap.py` - Python script version
- `requirements.txt` - All dependencies

## Troubleshooting

```bash
# Check Python version
python --version

# Verify environment
which python

# Check packages
pip list | grep -E "(shap|torch|captum)"

# Reinstall
pip install --force-reinstall -r requirements.txt
```

SHAP installation issues:
```bash
pip cache purge
pip install numba
pip install shap
pip install -r requirements.txt
```
