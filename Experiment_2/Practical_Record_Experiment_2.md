# PRACTICAL RECORD: EXPERIMENT 2

## 1. Title
**Create a Python Program to Display Summary and Statistics of a Dataset**

---

## 2. Aim
To write a beginner-friendly Python program using **Pandas** and **NumPy** to:
1. Display the statistical summary of the dataset using `describe()`.
2. Automatically identify numerical feature columns.
3. Compute summary statistics: Mean, Median, Mode, Minimum, Maximum, and Standard Deviation for all numerical columns.
4. Detect and count missing/null values column-by-column and in total across the dataset.

---

## 3. Requirements
- **Language:** Python 3.x
- **Libraries:** `pandas`, `numpy`, `os`
- **Dataset File:** `StudentsPerformance.csv` (Located in `archive/` directory)

---

## 4. Dataset Description
- **Filename:** `StudentsPerformance.csv`
- **Numerical Columns:** `math score`, `reading score`, `writing score`
- **Categorical Columns:** `gender`, `race/ethnicity`, `parental level of education`, `lunch`, `test preparation course`

---

## 5. Algorithm
1. **Start**
2. Import `pandas as pd`, `numpy as np`, and `os`.
3. Load `StudentsPerformance.csv` into DataFrame `df`.
4. Generate general descriptive statistics using `df.describe()`.
5. Automatically isolate numerical columns using `df.select_dtypes(include=[np.number]).columns`.
6. Calculate **Mean** using `df[numerical_cols].mean()`.
7. Calculate **Median** using `df[numerical_cols].median()`.
8. Calculate **Mode** using `df[numerical_cols].mode()`.
9. Calculate **Min** using `df[numerical_cols].min()`.
10. Calculate **Max** using `df[numerical_cols].max()`.
11. Calculate **Standard Deviation** using `df[numerical_cols].std()`.
12. Count missing values column-wise using `df.isnull().sum()`.
13. Compute total missing values using `df.isnull().sum().sum()`.
14. **Stop**

---

## 6. Python Code
```python
import os
import pandas as pd
import numpy as np

# Load dataset
file_path = os.path.join("..", "archive", "StudentsPerformance.csv")
df = pd.read_csv(file_path)

# Experiment 2 Operations
print("--- 1. Describe Summary ---")
print(df.describe())

# Auto-identify numerical columns
numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
print(f"\n--- 2. Numerical Columns: {numerical_cols} ---")

print("\n--- 3. Mean ---")
print(df[numerical_cols].mean())

print("\n--- 4. Median ---")
print(df[numerical_cols].median())

print("\n--- 5. Mode ---")
print(df[numerical_cols].mode().iloc[0])

print("\n--- 6. Minimum ---")
print(df[numerical_cols].min())

print("\n--- 7. Maximum ---")
print(df[numerical_cols].max())

print("\n--- 8. Standard Deviation ---")
print(df[numerical_cols].std())

print("\n--- 9. Missing Values per Column ---")
print(df.isnull().sum())

print(f"\n--- 10. Total Missing Values: {df.isnull().sum().sum()} ---")
```

---

## 7. Explanation of the Code
- **`df.describe()`**: Produces count, mean, std, min, percentiles (25%, 50%, 75%), and max for numerical columns.
- **`select_dtypes(include=[np.number])`**: Filters columns automatically so categorical strings do not cause errors during numeric calculation.
- **`mean()` / `median()` / `mode()`**: Measures of central tendency.
- **`min()` / `max()` / `std()`**: Measures of dispersion and score boundaries.
- **`isnull().sum()`**: Returns counts of null/missing values per attribute.

---

## 8. Expected Output
```text
===========================================================================
 EXPERIMENT 2: DISPLAY SUMMARY AND STATISTICS OF DATASET
===========================================================================

--- 1. Statistical Summary of the Dataset (describe()) ---
       math score  reading score  writing score
count  1000.00000    1000.000000    1000.000000
mean     66.08900      69.169000      68.054000
std      15.16308      14.600192      15.195657
min       0.00000      17.000000      10.000000
25%      57.00000      59.000000      57.750000
50%      66.00000      70.000000      69.000000
75%      77.00000      79.000000      79.000000
max     100.00000     100.000000     100.000000

--- 2. Automatically Identified Numerical Columns ---
Numerical Columns: ['math score', 'reading score', 'writing score']

--- 3. Mean of Numerical Columns ---
math score       66.089
reading score    69.169
writing score    68.054
dtype: float64

--- 4. Median of Numerical Columns ---
math score       66.0
reading score    70.0
writing score    69.0
dtype: float64

--- 5. Mode of Numerical Columns ---
math score       65
reading score    72
writing score    74
Name: 0, dtype: int64

--- 6. Minimum Values of Numerical Columns ---
math score        0
reading score    17
writing score    10
dtype: int64

--- 7. Maximum Values of Numerical Columns ---
math score       100
reading score    100
writing score    100
dtype: int64

--- 8. Standard Deviation of Numerical Columns ---
math score       15.163080
reading score    14.600192
writing score    15.195657
dtype: float64

--- 9. Missing/Null Values in Every Column ---
gender                         0
race/ethnicity                 0
parental level of education    0
lunch                          0
test preparation course        0
math score                     0
reading score                  0
writing score                  0
dtype: int64

--- 10. Total Number of Missing Values in Dataset ---
Total Missing/Null Values: 0
```

---

## 9. Result
Experiment 2 was successfully executed. The statistical summary was displayed, numerical columns were identified (`math score`, `reading score`, `writing score`), measures of central tendency and dispersion were calculated, and the dataset was verified to have 0 missing values.
