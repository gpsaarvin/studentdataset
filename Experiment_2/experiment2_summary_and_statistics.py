# ==============================================================================
# ML / DATA SCIENCE PRACTICAL EXPERIMENT 2
# Title: Display Summary and Statistics of a Dataset using Pandas and NumPy
# Dataset: Student Performance Dataset (StudentsPerformance.csv)
# ==============================================================================

# Task 1: Import required Python libraries (Pandas and NumPy)
import os
import pandas as pd
import numpy as np

# Set pandas display options for clean terminal output
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

print("=" * 75)
print(" EXPERIMENT 2: SUMMARY AND STATISTICAL ANALYSIS OF DATASET")
print("=" * 75)

# Task 2: Load the uploaded Student Performance Dataset CSV file
possible_paths = [
    "StudentsPerformance.csv",
    os.path.join("..", "StudentsPerformance.csv"),
    os.path.join("..", "archive", "StudentsPerformance.csv"),
    os.path.join("archive", "StudentsPerformance.csv")
]

file_path = None
for path in possible_paths:
    if os.path.exists(path):
        file_path = path
        break

if file_path is None:
    file_path = os.path.join("..", "archive", "StudentsPerformance.csv")

print(f"\n[INFO] Loading dataset from path: '{file_path}'...\n")
df = pd.read_csv(file_path)
print("[SUCCESS] Dataset loaded successfully into pandas DataFrame!\n")

# Task 9: Display statistical summary of the dataset using describe()
print("--- 1. Statistical Summary of the Dataset (describe()) ---")
print(df.describe())
print()

# Task 10: Identify only the numerical columns automatically
numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
print("--- 2. Automatically Identified Numerical Columns ---")
print(f"Numerical Columns: {numerical_cols}")
print()

# Task 11: Calculate and display the mean of all numerical columns
print("--- 3. Mean of Numerical Columns ---")
print(df[numerical_cols].mean())
print()

# Task 12: Calculate and display the median of all numerical columns
print("--- 4. Median of Numerical Columns ---")
print(df[numerical_cols].median())
print()

# Task 13: Calculate and display the mode where applicable
print("--- 5. Mode of Numerical Columns ---")
print(df[numerical_cols].mode().iloc[0])
print()

# Task 14: Display the minimum value of numerical columns
print("--- 6. Minimum Values of Numerical Columns ---")
print(df[numerical_cols].min())
print()

# Task 15: Display the maximum value of numerical columns
print("--- 7. Maximum Values of Numerical Columns ---")
print(df[numerical_cols].max())
print()

# Task 16: Display the standard deviation of numerical columns
print("--- 8. Standard Deviation of Numerical Columns ---")
print(df[numerical_cols].std())
print()

# Task 17: Check and display missing/null values in every column
print("--- 9. Missing/Null Values in Every Column ---")
print(df.isnull().sum())
print()

# Task 18: Display the total number of missing values in the dataset
total_missing = df.isnull().sum().sum()
print("--- 10. Total Number of Missing Values in Dataset ---")
print(f"Total Missing/Null Values: {total_missing}")
print()

print("=" * 75)
print(" EXPERIMENT 2 COMPLETED SUCCESSFULLY!")
print("=" * 75)
