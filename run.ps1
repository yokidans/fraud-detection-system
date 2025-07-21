<#
.SYNOPSIS
    Fraud Detection System - Main Execution Script

.DESCRIPTION
    This script provides a command-line interface for the fraud detection system,
    allowing users to preprocess data, train models, and run the web application.

.PARAMETER Action
    The action to perform: 'preprocess', 'train', or 'run'

.PARAMETER Dataset
    The dataset to use: 'ecommerce' or 'creditcard'

.EXAMPLE
    .\run.ps1 -Action preprocess -Dataset ecommerce
    .\run.ps1 -Action train -Dataset creditcard
    .\run.ps1 -Action run
#>

param(
    [Parameter(Mandatory=$false)]
    [ValidateSet("preprocess", "train", "run")]
    [string]$Action = "run",
    
    [Parameter(Mandatory=$false)]
    [ValidateSet("ecommerce", "creditcard")]
    [string]$Dataset = "ecommerce"
)

# Import required Python modules
$projectRoot = (Get-Item -Path $PSScriptRoot).FullName
$pythonScriptPath = Join-Path -Path $projectRoot -ChildPath "src"
$env:PYTHONPATH = "$pythonScriptPath;$($env:PYTHONPATH)"

# Load configuration
$config = @{
    FRAUD_DATA_PATH = Join-Path -Path $projectRoot -ChildPath "data\raw\Fraud_Data.csv"
    IP_MAPPING_PATH = Join-Path -Path $projectRoot -ChildPath "data\raw\IpAddress_to_Country.csv"
    CREDITCARD_DATA_PATH = Join-Path -Path $projectRoot -ChildPath "data\raw\creditcard.csv"
    MODEL_PATH = Join-Path -Path $projectRoot -ChildPath "models\best_model.pkl"
    PREPROCESSOR_PATH = Join-Path -Path $projectRoot -ChildPath "models\preprocessor.pkl"
}

function Invoke-Preprocess {
    param(
        [string]$datasetType
    )
    
    try {
        Write-Host "Starting data preprocessing for $datasetType..." -ForegroundColor Cyan
        
        # Create temporary Python script
        $tempScript = Join-Path -Path $env:TEMP -ChildPath "fraud_preprocess_$(Get-Date -Format 'yyyyMMddHHmmss').py"
        
        # Write the Python script with all required imports and proper handling
        @"
import sys
import os
import pandas as pd
import numpy as np
from scipy.sparse import save_npz, issparse
import joblib

# Add project root to Python path
project_root = r"$projectRoot"
src_path = os.path.join(project_root, "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

print(f"Python module search paths: {sys.path}")
print(f"Current working directory: {os.getcwd()}")

try:
    from data.preprocessing import DataPreprocessor
    print("Successfully imported DataPreprocessor")
except ImportError as e:
    print(f"ImportError: {e}")
    print("Current sys.path:", sys.path)
    raise

# Configuration
dataset_type = "$datasetType"
fraud_data_path = os.path.normpath(r"$($config.FRAUD_DATA_PATH)")
output_root = r"$projectRoot"

# Data loading
print("\nLoading data from:", fraud_data_path)
try:
    df = pd.read_csv(fraud_data_path)
    print("Data loaded successfully. Shape:", df.shape)
except Exception as e:
    print("Error loading data:", str(e))
    raise

# Data processing
try:
    preprocessor = DataPreprocessor({
        'missing_values_strategy': 'median',
        'random_state': 42,
        'sampling_strategy': 'SMOTE'
    })
    X, y, feature_names = preprocessor.preprocess_data(df, 'class')
    print("\nData processed successfully.")
    print("Features shape:", X.shape)
    print("Target shape:", y.shape)
    print("Number of features:", len(feature_names))
    
    # Save preprocessor
    os.makedirs(os.path.join(output_root, "models"), exist_ok=True)
    preprocessor_path = os.path.join(output_root, "models", "preprocessor.pkl")
    joblib.dump(preprocessor.preprocessor, preprocessor_path)
    print("\nSaved preprocessor to:", preprocessor_path)
    
except Exception as e:
    print("Error processing data:", str(e))
    raise

# Save results
output_dir = os.path.join(output_root, "data", "processed")
os.makedirs(output_dir, exist_ok=True)

try:
    # Handle sparse matrices
    if issparse(X):
        features_path = os.path.join(output_dir, "ecommerce_features.npz")
        save_npz(features_path, X)
    else:
        features_path = os.path.join(output_dir, "ecommerce_features.csv")
        pd.DataFrame(X, columns=feature_names).to_csv(features_path, index=False)
    
    target_path = os.path.join(output_dir, "ecommerce_target.npy")
    np.save(target_path, y)
    
    feature_names_path = os.path.join(output_dir, "ecommerce_feature_names.csv")
    pd.Series(feature_names).to_csv(feature_names_path, index=False)
    
    print("\nSaved features to:", features_path)
    print("Saved target to:", target_path)
    print("Saved feature names to:", feature_names_path)
    print("\nPreprocessing completed successfully")
except Exception as e:
    print("Error saving results:", str(e))
    raise
"@ | Out-File -FilePath $tempScript -Encoding utf8

        # Execute the Python script
        Write-Host "Executing preprocessing script..." -ForegroundColor Yellow
        $output = python $tempScript 2>&1
        
        # Display output
        Write-Host $output -ForegroundColor Green
        
        # Clean up
        Remove-Item -Path $tempScript -Force
        
    } catch {
        Write-Host "Error during preprocessing: $_" -ForegroundColor Red
        if (Test-Path $tempScript) {
            Write-Host "Review the script at: $tempScript" -ForegroundColor Yellow
        }
        exit 1
    }
}

function Invoke-Train {
    param(
        [string]$datasetType
    )
    
    try {
        Write-Host "Starting model training for $datasetType..." -ForegroundColor Cyan
        
        # Create a temporary Python script file
        $tempScript = Join-Path -Path $env:TEMP -ChildPath "fraud_train_$(Get-Date -Format 'yyyyMMddHHmmss').py"
        
        # Write the Python script with proper string formatting
        @"
import sys
import os
import pandas as pd
import numpy as np
from scipy.sparse import load_npz, issparse

# Add project root to Python path
project_root = r"$($projectRoot -replace '\\', '\\')"
src_path = os.path.join(project_root, "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from models.train import FraudDetectionModel

dataset_type = "$datasetType"
model_path = os.path.join(project_root, "models", "best_model.pkl")
processed_dir = os.path.join(project_root, "data", "processed")

# Load processed data
try:
    if dataset_type == "ecommerce":
        features_path = os.path.join(processed_dir, "ecommerce_features.npz")
        if os.path.exists(features_path):
            X = load_npz(features_path)
        else:
            features_path = os.path.join(processed_dir, "ecommerce_features.csv")
            X = pd.read_csv(features_path)
        
        y = np.load(os.path.join(processed_dir, "ecommerce_target.npy"))
    else:
        features_path = os.path.join(processed_dir, "creditcard_features.csv")
        X = pd.read_csv(features_path)
        y = np.load(os.path.join(processed_dir, "creditcard_target.npy"))
    
    print(f"Loaded {dataset_type} data. Features shape: {X.shape}, Target shape: {y.shape}")
    
    # Train model
    trainer = FraudDetectionModel({'random_state': 42})
    model = trainer.train_ensemble(X, y, optimize=True)
    trainer.save_model(model, model_path)
    
    print("\nModel training completed successfully")
    print("Saved model to:", model_path)
    
except Exception as e:
    print("Error during training:", str(e))
    raise
"@ | Out-File -FilePath $tempScript -Encoding utf8

        # Execute the Python script
        Write-Host "Executing training script..." -ForegroundColor Yellow
        $output = python $tempScript 2>&1
        
        # Display output
        Write-Host $output -ForegroundColor Green
        
        # Clean up
        Remove-Item -Path $tempScript -Force
        
    } catch {
        Write-Host "Error during training: $_" -ForegroundColor Red
        if (Test-Path $tempScript) {
            Write-Host "Review the script at: $tempScript" -ForegroundColor Yellow
        }
        exit 1
    }
}

function Invoke-Run {
    try {
        Write-Host "Starting web application..." -ForegroundColor Cyan
        $mainScript = Join-Path -Path $projectRoot -ChildPath "src\main.py"
        python $mainScript
    } catch {
        Write-Host "Error running web application: $_" -ForegroundColor Red
        exit 1
    }
}

# Execute the requested action
switch ($Action) {
    "preprocess" { Invoke-Preprocess -datasetType $Dataset }
    "train" { Invoke-Train -datasetType $Dataset }
    "run" { Invoke-Run }
    default { Write-Host "Invalid action specified" -ForegroundColor Red }
}