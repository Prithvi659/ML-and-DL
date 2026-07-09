"""Configuration for salary prediction model."""

import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
LOGS_DIR = PROJECT_ROOT / "logs"

# Ensure directories exist
MODELS_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)

# Data configuration
DATA_FILE = os.getenv("SALARY_DATA_PATH", str(DATA_DIR / "salary_data.csv"))
RANDOM_SEED = 42

# Train-validation-test split ratios
TRAIN_RATIO = 0.6
VALID_RATIO = 0.2
TEST_RATIO = 0.2

# Feature and target columns
FEATURE_COLUMN = "YearsExperience"
TARGET_COLUMN = "Salary"

# Model configuration
MODEL_NAME = "linear_regression_salary"
MODEL_PATH = MODELS_DIR / f"{MODEL_NAME}.pkl"
SCALER_PATH = MODELS_DIR / f"{MODEL_NAME}_scaler.pkl"

# Logging configuration
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FILE = LOGS_DIR / "salary_model.log"
