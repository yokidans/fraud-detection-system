# Fraud Detection System for E-commerce and Bank Transactions

## Project Overview

This project aims to improve fraud detection for e-commerce and bank credit transactions using machine learning. The system analyzes transaction patterns, geolocation data, and user behavior to identify fraudulent activities accurately. Key challenges include handling class imbalance and balancing security with user experience.

### Key Features:
- **Data Analysis**: Preprocesses and merges transaction datasets with geolocation data.
- **Feature Engineering**: Creates time-based and behavioral features to enhance fraud detection.
- **Model Training**: Implements Logistic Regression and ensemble models (Random Forest or Gradient Boosting).
- **Model Explainability**: Uses SHAP to interpret model decisions and identify fraud drivers.
- **Real-time Monitoring**: Designed for efficient reporting and quick action on detected fraud.

## Project Structure

## fraud-detection-system/
### ├── data/
### │   ├── raw/                   # Original datasets (e.g., Fraud_Data.csv, creditcard.csv)
### │   ├── processed/             # Cleaned and processed data (e.g., merged datasets)
### │   └── interim/               # Intermediate files (e.g., feature-engineered data)
### ├── models/                    # Saved models (e.g., .pkl files)
### ├── notebooks/                 # Jupyter notebooks for exploratory analysis
### ├── src/
### │   ├── data/                  # Data processing modules
### │   │   ├── preprocessing.py   # Handles missing values, duplicates, and data cleaning
### │   │   ├── feature_engineering.py # Creates features like time_since_signup
### │   │   └── geolocation.py     # Maps IP addresses to countries
### │   ├── models/                # Model-related code
### │   │   ├── train.py           # Trains models (Logistic Regression, Random Forest, etc.)
### │   │   ├── evaluate.py        # Evaluates models using metrics like AUC-PR and F1-Score
### │   │   └── explainability.py  # Generates SHAP plots for model interpretation
### │   ├── visualization/         # Visualization utilities
### │   │   └── eda.py             # Creates EDA plots (e.g., histograms, correlation matrices)
### │   ├── config.py              # Configuration settings (e.g., file paths, hyperparameters)
### │   └── app/                   # Web application (Flask/Dash)
### │       ├── main.py            # Entry point for the web app
### │       └── templates/         # HTML templates for the UI
### ├── tests/                     # Unit and integration tests
### ├── docs/                      # Project documentation (e.g., reports, blog posts)
### ├── requirements.txt           # Python dependencies
### ├── Makefile                   # Build automation (e.g., running tests, preprocessing)
### └── README.md                  # Project documentation (this file)

## Setup Instructions

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/yokidans/fraud-detection-system.git
   cd fraud-detection-system
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Download the datasets:
   - Place `Fraud_Data.csv`, `IpAddress_to_Country.csv`, and `creditcard.csv` in the `data/raw/` folder.

## Execution Steps

### Data Preprocessing and Feature Engineering
1. Run the preprocessing script:
   ```bash
   python src/data/preprocessing.py
   ```

2. Generate features:
   ```bash
   python src/data/feature_engineering.py
   ```

3. Merge datasets with geolocation data:
   ```bash
   python src/data/geolocation.py
   ```

### Model Training and Evaluation
1. Train the models:
   ```bash
   python src/models/train.py
   ```

2. Evaluate model performance:
   ```bash
   python src/models/evaluate.py
   ```

3. Generate SHAP plots for explainability:
   ```bash
   python src/models/explainability.py
   ```

### Web Application (Optional)
To run the Flask/Dash app:
```bash
python src/app/main.py
```
Access the app at `http://localhost:5000`.

## Key Files and Their Roles
- **`requirements.txt`**: Lists all Python dependencies.
- **`Makefile`**: Automates tasks like testing and preprocessing.
- **`notebooks/`**: Contains exploratory analysis and prototyping.
- **`src/config.py`**: Centralizes configuration settings for easy adjustments.

## Testing
Run unit tests to ensure functionality:
```bash
make test
```
## Documentation
- **Interim Reports**: Located in `docs/interim/`.
- **Final Report**: A detailed PDF or blog post in `docs/final/`.

## References
- Dataset sources and tutorials are linked in the project documentation (`docs/references.md`).

## License
This project is licensed under the MIT License. See `LICENSE` for details.
```
