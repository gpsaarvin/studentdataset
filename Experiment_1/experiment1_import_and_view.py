# ==============================================================================
# ML / DATA SCIENCE PRACTICAL EXPERIMENT 1
# Title: Import, Load, and View a Dataset using Pandas and NumPy
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
print(" EXPERIMENT 1: IMPORT, LOAD, AND VIEW DATASET")
print("=" * 75)

# Task 2: Load the uploaded Student Performance Dataset CSV file
# Search for dataset file in common relative directories
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

print("=" * 75)
print(" EXPERIMENT 1 COMPLETED SUCCESSFULLY!")
print("=" * 75)
