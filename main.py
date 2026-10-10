"""
Main execution script for Structural Damage Image Classification.
UE24CS352A Machine Learning Mini-Project | Problem Statement #39
Team Members: Nikhil B Menon (PES2UG24CS317) & Niharika SM (PES2UG24CS316)
Runs data loading, preprocessing, model training, evaluation, and plot generation.
"""

import os
import sys
import time
from preprocessing import load_and_preprocess_data
from train_eval import train_and_eval_all
from config import REPORT_PDF_PATH, OUTPUT_DIR


def main():
    print("=" * 75)
    print("  UE24CS352A Machine Learning Mini-Project - Problem Statement #39")
    print("  Structural Damage Image Classification Using Classical ML Models")
    print("  Team: Nikhil B Menon (PES2UG24CS317) & Niharika SM (PES2UG24CS316)")
    print("=" * 75)

    start_time = time.time()

    # Step 1: Preprocessing & Data Loading
    print("\n[Step 1/3] Loading and Preprocessing Dataset...")
    data_dict = load_and_preprocess_data()

    # Step 2: Training & Model Evaluation
    print("\n[Step 2/3] Training Models & Computing Performance Metrics...")
    df_results, eval_dict, models = train_and_eval_all(data_dict)

    # Step 3: Verification & Deliverables Summary
    elapsed_time = time.time() - start_time
    print("\n[Step 3/3] Pipeline Execution Completed Successfully!")
    print(f"Total Execution Time: {elapsed_time:.2f} seconds")
    print(f"Output Directory:    {OUTPUT_DIR}")
    if os.path.exists(REPORT_PDF_PATH):
        print(f"PDF Summary Report:  {REPORT_PDF_PATH}")
    print("=" * 75)


if __name__ == "__main__":
    main()
