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

### Python 3.13 Compatibility Issue (IMPORTANT)

**Problem:** scipy 1.15.x/1.16.x precompiled binaries for Python 3.13 on macOS ARM have a known bug with numpy < 2.0:
```
ValueError: All ufuncs must have type `numpy.ufunc`. Received (<ufunc 'sph_legendre_p'>, ...)
```

This prevents importing:
- `sklearn` (scikit-learn) - required for time-series notebooks
- `transformers` - required for image notebooks  

**Root Cause:**
- Captum 0.8.0 requires `numpy < 2.0`
- scipy 1.15/1.16 binaries were compiled with numpy 2.x type checking
- Type mismatch causes scipy internal modules to fail

**Recommended Solutions:**

1. **Use Python 3.11 or 3.12** (Easiest - No issues!)
   ```bash
   # Create new venv with Python 3.12
   python3.12 -m venv venv_py312
   source venv_py312/bin/activate
   pip install -r requirements.txt
   ```

2. **Build scipy from source** (Works on Python 3.13, takes ~15 min)
   ```bash
   # Requires Xcode Command Line Tools on macOS
   xcode-select --install
   
   # Uninstall precompiled scipy
   pip uninstall scipy
   
   # Build from source (takes 10-15 minutes)
   pip install scipy --no-binary scipy
   ```

3. **Wait for Captum update** (When Captum supports numpy 2.0, scipy will work)

**Current Status:**
- Python version in use: **3.13.5** ⚠️
- scipy version: 1.15.3 (precompiled binary)
- numpy version: 1.26.4 (locked by Captum < 2.0)
- **Image notebook (`explain_images.ipynb`)**: ❌ Blocked by this issue
- **Timeseries notebook (`explain_timeseries.ipynb`)**: ❌ Blocked by this issue
- **Python modules (`src/*.py`)**: ✅ Work fine (import error only at runtime)

### Other Common Issues

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
