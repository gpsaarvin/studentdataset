# ==============================================================================
# ML / DATA SCIENCE PRACTICAL EXPERIMENTS
# Dataset: Student Performance Dataset (StudentsPerformance.csv)
# Experiment 1: Import, Load, and View Dataset
# Experiment 2: Summary and Statistical Analysis of Dataset
# ==============================================================================

# Task 1: Import required Python libraries (Pandas and NumPy)
import os
import pandas as pd
import numpy as np

# Set pandas display options for clean terminal formatting
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

# ==============================================================================
# EXPERIMENT 1: IMPORT, LOAD, AND VIEW DATASET
# ==============================================================================
print("=" * 75)
print(" EXPERIMENT 1: IMPORT, LOAD, AND VIEW DATASET")
print("=" * 75)

# Task 2: Load the uploaded Student Performance Dataset CSV file
# Check if file exists in root or archive directory automatically
dataset_filename = "StudentsPerformance.csv"
if os.path.exists(dataset_filename):
    file_path = dataset_filename
elif os.path.exists(os.path.join("archive", dataset_filename)):
    file_path = os.path.join("archive", dataset_filename)
else:
    file_path = dataset_filename  # Default fallback

print(f"\n[INFO] Loading dataset from: '{file_path}'...\n")
df = pd.read_csv(file_path)
print("[SUCCESS] Dataset loaded successfully into pandas DataFrame!\n")

# Task 3: Display the first 5 rows of the dataset using head()
print("--- 1. First 5 Rows of the Dataset (head()) ---")
print(df.head())
print()

# Task 4: Display the last 5 rows of the dataset using tail()
print("--- 2. Last 5 Rows of the Dataset (tail()) ---")
print(df.tail())
print()

# Task 5: Display the number of rows and columns using shape
print("--- 3. Shape of the Dataset (Rows, Columns) ---")
print(f"Number of Rows    : {df.shape[0]}")
print(f"Number of Columns : {df.shape[1]}")
print(f"Shape (Rows, Cols): {df.shape}")
print()

# Task 6: Display all column names
print("--- 4. All Column Names ---")
for index, col in enumerate(df.columns, start=1):
    print(f" {index}. {col}")
print()

# Task 7: Display the data type of each column
print("--- 5. Data Type of Each Column ---")
print(df.dtypes)
print()

# Task 8: Display complete basic information about the dataset using info()
print("--- 6. Complete Basic Information (info()) ---")
df.info()
print()


# ==============================================================================
# EXPERIMENT 2: DISPLAY SUMMARY AND STATISTICS OF DATASET
# ==============================================================================
print("=" * 75)
print(" EXPERIMENT 2: DISPLAY SUMMARY AND STATISTICS OF DATASET")
print("=" * 75)

# Task 9: Display statistical summary of the dataset using describe()
print("\n--- 1. Statistical Summary of the Dataset (describe()) ---")
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
print(" PROGRAM COMPLETED SUCCESSFULLY!")
print("=" * 75)
