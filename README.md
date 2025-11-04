# Electricity Consumption Analysis with SHAP

This directory contains a comprehensive electricity consumption prediction analysis with SHAP explanations. The main analysis is in the Jupyter notebook `electricity_consumption_shap.ipynb`.

## 🎯 Main Features

- **Time-series electricity consumption prediction** with 96.4% R² accuracy
- **SHAP explanations** for model interpretability and individual prediction analysis
- **Alternative feature analysis** for Python 3.14+ compatibility
- **Advanced feature engineering** with 29+ temporal features including lags, rolling statistics, and cyclical encodings
- **Multiple ML models** comparison (Random Forest, Gradient Boosting, Linear Regression)
- **Interactive visualizations** including waterfall plots, summary plots, and correlation analysis

## 🔧 Quick Setup (New Users)

The fastest way to get started is using the provided `requirements.txt` file:

```bash
# 1. Clone or download this directory
# 2. Navigate to the project directory
cd path/to/electricity_consumption_analysis

# 3. Create virtual environment (Python 3.11-3.13 recommended for full SHAP support)
python3.13 -m venv electricity_env
source electricity_env/bin/activate  # On Windows: electricity_env\Scripts\activate

# 4. Install all dependencies from requirements.txt
pip install -r requirements.txt

# 5. Launch Jupyter notebook
jupyter notebook electricity_consumption_shap.ipynb
```

**That's it!** The notebook will automatically detect SHAP availability and provide the appropriate analysis.

## 📊 Quick Results
- **Best Model**: Random Forest (96.43% R²)
- **Key Feature**: `lag_168h` (7 days ago) - 94.77% importance
- **Strong weekly patterns** in electricity consumption

## 🚀 Running the Analysis

### For Full SHAP Analysis (Recommended)

Use the virtual environment that has SHAP properly configured:

#### Method 1: Using Terminal Commands

1. Navigate to this directory:
   ```bash
   cd path/to/electricity_consumption_shap
   ```

2. Activate the virtual environment:
   ```bash
   source venv/bin/activate
   ```

3. Launch the notebook with full SHAP functionality:
   ```bash
   jupyter notebook electricity_consumption_shap.ipynb
   ```
   
   Or run Python files:
   ```bash
   python test_shap.py
   python electricity_consumption_shap.py
   ```

4. When finished, deactivate the environment:
   ```bash
   deactivate
   ```

#### Method 2: VS Code Integration

If you're using VS Code, the workspace should automatically use the virtual environment. If not:

1. Open VS Code in this directory
2. Press `Cmd+Shift+P` (on macOS) to open the command palette
3. Type "Python: Select Interpreter" and select it
4. Choose the interpreter at: `./venv/bin/python`
5. Open the notebook: `electricity_consumption_shap.ipynb`

### Alternative: Without SHAP (Python 3.14+)

If you can't use the virtual environment, the notebook still provides comprehensive analysis:
- Traditional feature importance analysis
- Correlation analysis and visualizations
- Individual prediction breakdowns
- All the same insights with different explanation methods

## 📦 Environment Setup Options

### Option 1: Use requirements.txt (Recommended for New Setups)

A complete `requirements.txt` file is provided with all exact versions needed:

```bash
# Create fresh virtual environment
python3.13 -m venv electricity_env
source electricity_env/bin/activate  # Windows: electricity_env\Scripts\activate

# Install all dependencies with exact versions
pip install -r requirements.txt

# Verify installation
python -c "import shap; print(f'SHAP {shap.__version__} ready!')"

# Launch analysis
jupyter notebook electricity_consumption_shap.ipynb
```

### Option 2: Use Existing Virtual Environment (If Available)

If the `venv/` directory exists and works for you:

```bash
# Activate existing environment
source venv/bin/activate

# Run analysis
jupyter notebook electricity_consumption_shap.ipynb
```

### Option 3: Minimal Installation (Python 3.14+ or SHAP Issues)

For systems where SHAP won't install, use core packages only:

```bash
# Install essential packages
pip install numpy pandas matplotlib scikit-learn jupyter seaborn

# The notebook will automatically switch to alternative analysis mode
jupyter notebook electricity_consumption_shap.ipynb
```

### Core Dependencies
- **numpy** (≥1.24.0) - Numerical computing and array operations
- **pandas** (≥2.0.0) - Data manipulation and time-series analysis
- **matplotlib** (≥3.7.0) - Plotting and visualization
- **scikit-learn** (≥1.3.0) - Machine learning algorithms (Random Forest, Gradient Boosting, etc.)

### SHAP Dependencies (Python 3.11-3.13)
- **SHAP** (≥0.41.0) - Model explanations and interpretability
- **numba** (≥0.56.0) - JIT compilation for SHAP performance
- **cloudpickle** (≥2.0.0) - Serialization support for SHAP
- **tqdm** (≥4.64.0) - Progress bars for SHAP calculations

### Optional Dependencies
- **jupyter** - Notebook environment
- **seaborn** - Enhanced statistical visualizations
- **ipywidgets** - Interactive notebook widgets

### What's in requirements.txt

The `requirements.txt` file includes carefully tested versions for maximum compatibility:

```txt
# Core data science stack
numpy>=1.24.0,<2.0.0      # Numerical computing
pandas>=2.0.0,<3.0.0       # Data manipulation  
matplotlib>=3.7.0,<4.0.0   # Plotting
scikit-learn>=1.3.0,<2.0.0 # Machine learning

# SHAP and dependencies (Python 3.11-3.13)
shap>=0.41.0,<0.50.0       # Model explanations
numba>=0.56.0,<1.0.0       # JIT compilation for performance
cloudpickle>=2.0.0,<3.0.0  # Serialization
tqdm>=4.64.0,<5.0.0        # Progress bars

# Jupyter ecosystem
jupyter>=1.0.0,<2.0.0      # Notebook environment
ipykernel>=6.0.0,<7.0.0    # Kernel support
ipywidgets>=8.0.0,<9.0.0   # Interactive widgets

# Optional enhancements
seaborn>=0.12.0,<1.0.0     # Statistical visualizations
```

### Version Strategy
- **Lower bounds** ensure required features are available
- **Upper bounds** prevent breaking changes from newer versions
- **SHAP ecosystem** versions tested together for compatibility
- **Python 3.11-3.13** explicitly supported for full functionality

## ✅ Testing the Environment

### Test SHAP functionality:
```bash
source venv/bin/activate
python test_shap.py
```

### Run the main analysis:
```bash
source venv/bin/activate
jupyter notebook electricity_consumption_shap.ipynb
```

You should see:
- ✅ SHAP working perfectly with full analysis capabilities
- 📊 Rich visualizations and explanations
- 🎯 96%+ model accuracy with interpretable predictions

## 🧠 SHAP vs Alternative Analysis

| **With Virtual Environment** | **Without SHAP (Python 3.14+)** |
|------------------------------|-----------------------------------|
| ✅ Individual prediction explanations | ✅ Traditional feature importance |
| ✅ SHAP waterfall plots | ✅ Correlation analysis |
| ✅ Feature interaction analysis | ✅ Weighted contribution charts |
| ✅ TreeExplainer for tree models | ✅ Model-specific importance |
| 🎯 **Recommended for full insights** | 🎯 **Still very comprehensive!** |

## 🆘 Troubleshooting

### Environment Issues
```bash
# Check Python version
python --version  # Should be 3.11-3.13 for SHAP

# Verify virtual environment is active
which python  # Should point to your venv/bin/python

# Check installed packages
pip list | grep -E "(shap|numpy|pandas|sklearn)"

# Reinstall if needed
pip install --force-reinstall -r requirements.txt
```

### SHAP Installation Problems
```bash
# Clear pip cache and reinstall
pip cache purge
pip install --no-cache-dir -r requirements.txt

# Alternative: Install SHAP separately first
pip install numba  # Install numba first
pip install shap   # Then SHAP
pip install -r requirements.txt  # Then everything else
```

### VS Code Integration
- **Import errors in VS Code but terminal works**: Restart VS Code or reload window (`Cmd+Shift+P` → "Developer: Reload Window")
- **Wrong Python interpreter**: `Cmd+Shift+P` → "Python: Select Interpreter" → Choose your virtual environment
- **Kernel issues**: In notebook, click kernel name → "Select Another Kernel" → Choose your environment

### Fallback Options
If SHAP won't install on your system:
1. **Use Python 3.11-3.13** in a clean virtual environment
2. **Try conda instead of pip**: `conda install -c conda-forge shap`
3. **Use minimal setup**: The notebook provides excellent traditional analysis without SHAP

### Common Error Messages
- `"No module named 'shap'"` → Virtual environment not activated or SHAP not installed
- `"DLL load failed"` (Windows) → Try reinstalling numba: `pip install --force-reinstall numba`
- `"Python version not supported"` → Use Python 3.11-3.13 for full SHAP support

### Getting Help
- Check notebook output for automatic SHAP detection messages
- Run `python test_shap.py` to verify SHAP installation
- The notebook gracefully falls back to traditional analysis if SHAP unavailable