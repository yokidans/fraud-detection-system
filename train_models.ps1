<#
.SYNOPSIS
    Fraud Detection Model Training Pipeline
.DESCRIPTION
    Trains and evaluates Logistic Regression and XGBoost models
    with hyperparameter tuning and business-aligned evaluation
#>

# Import required modules
Import-Module .\src\data\preprocessing.psm1
Import-Module .\src\models\training.psm1

# Configuration
$config = @{
    DataPath = "data/processed"
    ModelPath = "models"
    ReportsPath = "reports"
    RandomState = 42
    TestSize = 0.2
    CostMatrix = @{ FP = 10; FN = 100 } # Business cost assumptions
}

# Create required directories
New-Item -ItemType Directory -Path $config.ModelPath -Force
New-Item -ItemType Directory -Path $config.ReportsPath -Force

# 1. Data Loading
function Load-Data {
    param(
        [string]$datasetType
    )
    
    $featuresPath = Join-Path $config.DataPath "${datasetType}_features.csv"
    $targetPath = Join-Path $config.DataPath "${datasetType}_target.csv"
    
    $X = Import-Csv $featuresPath
    $y = Import-Csv $targetPath | Select-Object -ExpandProperty "class"
    
    return $X, $y
}

# 2. Model Training Pipeline
function Train-Models {
    param(
        [object]$X,
        [object]$y,
        [string]$datasetName
    )
    
    # Split data
    $split = Train-TestSplit -X $X -y $y -TestSize $config.TestSize -RandomState $config.RandomState
    
    # Initialize models
    $models = @{
        "LogisticRegression" = New-FraudLogisticModel -RandomState $config.RandomState
        "XGBoost" = New-FraudXGBoostModel -RandomState $config.RandomState
    }

    $bestParams = Optimize-FraudModel -X $split.X_train -y $split.y_train
    $metrics = Test-FraudModel -Model $model -X $split.X_test -y $split.y_test
        
    # Hyperparameter tuning for XGBoost
    $models.XGBoost = Update-XGBoost -Model $models.XGBoost -Params $bestParams
    
    # Train and evaluate
    $results = @{}
    foreach ($name in $models.Keys) {
        Write-Host "Training $name model..." -ForegroundColor Cyan
        
        $model = $models[$name]
        $model = Train-Model -Model $model -X $split.X_train -y $split.y_train
        
        $metrics = Evaluate-Model -Model $model -X $split.X_test -y $split.y_test
        $results[$name] = $metrics
        
        # Generate SHAP explanations for XGBoost
        if ($name -eq "XGBoost") {
            $shapResults = Invoke-ShapAnalysis -Model $model -X $split.X_test
            Export-ShapPlots -Results $shapResults -Path $config.ReportsPath
        }
    }
    
    # Select best model based on business costs
    $bestModel = Select-BestModel -Results $results -CostMatrix $config.CostMatrix
    Save-Model -Model $bestModel -Path (Join-Path $config.ModelPath "best_${datasetName}_model.pkl")
    
    return $results
}

# 3. Execute for both datasets
$ecommerceResults = Train-Models -X (Load-Data "ecommerce").X -y (Load-Data "ecommerce").y -datasetName "ecommerce"
$creditcardResults = Train-Models -X (Load-Data "creditcard").X -y (Load-Data "creditcard").y -datasetName "creditcard"

# 4. Generate consolidated report
$report = @{
    Timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    EcommerceResults = $ecommerceResults
    CreditCardResults = $creditcardResults
} | ConvertTo-Json -Depth 5

$report | Out-File (Join-Path $config.ReportsPath "model_performance.json")