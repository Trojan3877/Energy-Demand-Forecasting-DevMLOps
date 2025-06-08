"""
build_features.py

Module: Feature Engineering for Energy Demand Forecasting Pipeline
Author: Corey Leath

Description:
- Loads processed combined data
- Creates rolling window features
- Outputs engineered features CSV

Input:
- data/processed/combined.csv

Output:
- data/processed/features.csv
"""

import pandas as pd
import os

def main():
    # Define input path
    input_path = 'data/processed/combined.csv'

    # Define output path
    output_path = 'data/processed/features.csv'

    # Load combined data
    print(f"Loading combined data from {input_path}...")
    df = pd.read_csv(input_path, parse_dates=['timestamp'])

    # Set timestamp as index
    df.set_index('timestamp', inplace=True)

    # Create rolling window features
    print("Creating rolling window features (3-hour moving avera

git add src/features/build_features.py
git commit -m "Add feature engineering module"
git push
