"""
Configuration file for Structural Damage Image Classification Project.
UE24CS352A Machine Learning Mini-Project | Problem Statement #39
"""

import os

# Project Directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
MODELS_DIR = os.path.join(OUTPUT_DIR, "models")
SAMPLE_DIR = os.path.join(BASE_DIR, "sample_images")
DOCS_DIR = os.path.join(BASE_DIR, "docs")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(SAMPLE_DIR, exist_ok=True)
os.makedirs(DOCS_DIR, exist_ok=True)

# Random Seed for Reproducibility
RANDOM_SEED = 42

# Image Processing Specifications
IMG_HEIGHT = 48
IMG_WIDTH = 48
IMG_SIZE = (IMG_HEIGHT, IMG_WIDTH)
INPUT_DIM = IMG_HEIGHT * IMG_WIDTH  # 2,304 continuous raw pixel features
NORM_RANGE = (-1.0, 1.0)

# Dataset Pooling and Splits
DEFAULT_DATA_DIRS = [
    os.path.expanduser("~/Downloads/task2_damage_state_1"),
    os.path.join(BASE_DIR, "data"),
    os.path.join(BASE_DIR, "task2_damage_state_1"),
]
SAMPLE_SIZE = 3000  # Balanced: 1,500 Damaged, 1,500 Undamaged
SPLIT_RATIOS = (0.70, 0.15, 0.15)  # Train, Validation, Test

# Model Hyperparameters (Strict Classical Syllabus - No CNNs, No Bayes)
LOGISTIC_PARAMS = {
    'penalty': 'l2',
    'C': 1.0,
    'solver': 'lbfgs',
    'max_iter': 1000,
    'random_state': RANDOM_SEED
}

DECISION_TREE_PARAMS = {
    'criterion': 'entropy',  # ID3 Algorithm: Shannon Entropy & Information Gain
    'max_depth': 10,
    'min_samples_split': 5,
    'random_state': RANDOM_SEED
}

SVM_PARAMS = {
    'kernel': 'rbf',
    'C': 1.0,
    'gamma': 'scale',
    'probability': True,
    'random_state': RANDOM_SEED
}

# Output Deliverables Paths
REPORT_PDF_PATH = os.path.join(OUTPUT_DIR, "Structural_Damage_Classification_Report.pdf")
CONFUSION_PLOT_PATH = os.path.join(OUTPUT_DIR, "confusion_matrices.png")
ROC_PLOT_PATH = os.path.join(OUTPUT_DIR, "roc_curves.png")
PCA_PLOT_PATH = os.path.join(OUTPUT_DIR, "pca_scatter.png")
SAMPLE_PRED_PATH = os.path.join(OUTPUT_DIR, "sample_predictions.png")
METRICS_CSV_PATH = os.path.join(OUTPUT_DIR, "model_comparison.csv")
ROC_DATA_PATH = os.path.join(MODELS_DIR, "roc_data.joblib")
