import pandas as pd
import numpy as np
import matplotlib as mpl

# Set global font to Times New Roman
mpl.rcParams['font.family'] = 'Times New Roman'

# 1. Load dataset
# Make sure the path matches your local environment
file_path = r'dataset\final_dataset_with_features.csv'
df = pd.read_csv(file_path, parse_dates=["Date"])

# 2. Define the features used in the Mahalanobis model
features = [
    'Avg_rainfall', 'Avg_smlvl_at15cm', 'Annual_Temp', 'Temp_Seasonal',
    'Rainfall_lag_1d', 'Rainfall_lag_3d', 'Rainfall_lag_7d',
    'Rainfall_7d_sum', 'Rainfall_7d_avg', 'SoilMoisture_7d_avg',
    'day_of_year', 'week_of_year', 'month_sin', 'month_cos'
]

# --- PRE-CLEANING DATA CHECK ---

print("="*50)
print("PRE-CLEANING MISSING VALUE ANALYSIS")
print("="*50)

# 1. Calculate missing percentage for every column in the dataframe
missing_percentage = (df.isnull().sum() / len(df)) * 100

print("\n--- [All Columns] Percentage of Missing Values ---")
# Filter to show only columns that actually have missing values
missing_only = missing_percentage[missing_percentage > 0].sort_values(ascending=False)
if not missing_only.empty:
    print(missing_only)
else:
    print("No missing values found in the entire dataset.")

# 2. Specifically check the features you are using for the model
print("\n--- [Model Features] Missing Values in Selected Features ---")
features_missing = (df[features].isnull().sum() / len(df)) * 100
print(features_missing.sort_values(ascending=False))

print("\n" + "="*50)
print("End")