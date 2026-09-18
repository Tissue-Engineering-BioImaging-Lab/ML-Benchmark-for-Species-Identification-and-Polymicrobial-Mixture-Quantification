# TEB GRAD-CAM and CNN Notebook

This README documents `TEB GRAD_CAM and CNN copy.ipynb`, which trains 1D CNN models on Raman spectra and generates Grad-CAM explanations for binary and multi-class classification.

## What this notebook does

- Loads spectral CSV data and labels.
- Standardizes features and encodes labels.
- Tunes CNN hyperparameters with Optuna (`n_trials=5`).
- Trains:
- Binary CNN (`best_binary_cnn.keras`)
- Multiclass CNN (`best_multiclass_cnn.keras`)
- Evaluates models with confusion matrix, accuracy, precision, recall, F1, classification report, and ROC curves.
- Runs class-averaged 1D Grad-CAM and prints wavenumber importance values.

## Expected data format

- Input is a CSV where:
- Columns `[:-1]` are spectral features (wavenumbers).
- Last column is the class label.
- The notebook drops the first row (`data.drop(index=0)`), assuming it is not a training sample.
- The Grad-CAM x-axis is read from CSV column headers (`df.columns[:-1].astype(float)`).

## Environment

Use Python 3.10+ and install:

```bash
pip install numpy pandas scikit-learn tensorflow matplotlib seaborn optuna
```

## Before you run

Update these variables in the notebook:

1. `data_path` for binary workflow (default: `Binary_Test_ANN_CNN_data.csv`).
2. `data_path_multi` for multiclass workflow (default: `Test_ANN_CNN_data.csv`).
3. `label_map` dictionaries for your real class names.
4. Optional: `conv_layer_name` in Grad-CAM calls if you want a specific Conv1D layer.

Note: if `conv_layer_name` is not provided, the notebook now auto-selects the last Conv1D layer in the model.

## Notebook workflow

1. Import libraries and define helpers (`prepare_data`, `prepare_data_binary`).
2. Binary model:
- Build + Optuna objective.
- Run `study_binary.optimize(...)`.
- Train best model and save `best_binary_cnn.keras`.
- Evaluate and plot metrics.
- Run binary class-averaged Grad-CAM.
3. Multiclass model:
- Build + Optuna objective.
- Run `study_multi.optimize(...)`.
- Train best model and save `best_multiclass_cnn.keras`.
- Evaluate and plot metrics.
- Run multiclass class-averaged Grad-CAM for classes 1-6.

## Outputs

- Saved models:
- `best_binary_cnn.keras`
- `best_multiclass_cnn.keras`
- Plots:
- Confusion matrices
- ROC curves
- Training vs validation accuracy curves
- Per-class Grad-CAM plots over wavelength
- Console output:
- Best Optuna hyperparameters
- Classification metrics
- Wavenumber-importance pairs for each class

## Notes and caveats

- The binary pipeline now uses `prepare_data_binary(...)` in optimization and final training.
- `n_trials=5` is fast but usually not enough for stable hyperparameter tuning. Increase trials for final experiments.
- For reproducibility, consider adding fixed seeds for NumPy/TensorFlow/Optuna.

## Related files in this project

- `TEB GRAD_CAM and CNN copy.ipynb`
- `best_binary_cnn.keras`
- `best_multiclass_cnn.keras`
- `best_resnet_binary.keras`
- `best_resnet_multiclass.keras`
- `best_transformer_multiclass.keras`
- `CNN Training Data/`
