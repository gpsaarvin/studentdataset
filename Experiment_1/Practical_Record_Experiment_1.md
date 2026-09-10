# PRACTICAL RECORD: EXPERIMENT 1

## 1. Title
**Devise a Python Program to Import, Load, and View a Dataset**

---

## 2. Aim
To write a beginner-friendly Python program using **Pandas** and **NumPy** libraries to:
1. Load the Kaggle **Student Performance Dataset** (`StudentsPerformance.csv`).
2. View sample dataset records using `head()` and `tail()`.
3. Determine dataset shape (row and column counts).
4. Extract all column headers and data types.
5. Display complete structural information using `info()`.

---

## 3. Requirements
- **Language:** Python 3.x
- **Libraries:** `pandas`, `numpy`, `os`
- **Dataset File:** `StudentsPerformance.csv` (Located in `archive/` directory)

---

## 4. Dataset Description
- **Filename:** `StudentsPerformance.csv`
- **Total Records:** 1,000 rows
- **Total Columns:** 8 attributes (5 categorical, 3 numerical integer scores)

| S.No | Column Name | Data Type | Kind |
| :--- | :--- | :--- | :--- |
| 1 | `gender` | `str` (`object`) | Categorical |
| 2 | `race/ethnicity` | `str` (`object`) | Categorical |
| 3 | `parental level of education` | `str` (`object`) | Categorical |
| 4 | `lunch` | `str` (`object`) | Categorical |
| 5 | `test preparation course` | `str` (`object`) | Categorical |
| 6 | `math score` | `int64` | Numerical |
| 7 | `reading score` | `int64` | Numerical |
| 8 | `writing score` | `int64` | Numerical |

---

## 5. Algorithm
1. **Start**
2. Import `pandas as pd`, `numpy as np`, and `os`.
3. Locate `StudentsPerformance.csv` and load it into a DataFrame `df` using `pd.read_csv()`.
4. Display the top 5 rows using `df.head()`.
5. Display the bottom 5 rows using `df.tail()`.
6. Print row and column dimensions using `df.shape`.
7. Iterate and display all column names using `df.columns`.
8. Print data type of each column using `df.dtypes`.
9. Display concise DataFrame summary including memory usage using `df.info()`.
10. **Stop**

---

## 6. Python Code
```python
import os
import pandas as pd
import numpy as np

# Load dataset
file_path = os.path.join("..", "archive", "StudentsPerformance.csv")
df = pd.read_csv(file_path)

# Experiment 1 Operations
print("--- 1. First 5 Rows (head()) ---")
print(df.head())

print("\n--- 2. Last 5 Rows (tail()) ---")
print(df.tail())

print(f"\n--- 3. Shape: {df.shape} (Rows: {df.shape[0]}, Cols: {df.shape[1]}) ---")

print("\n--- 4. Column Names ---")
print(list(df.columns))

print("\n--- 5. Column Data Types ---")
print(df.dtypes)

print("\n--- 6. Dataset Info (info()) ---")
df.info()
```

---

## 7. Explanation of the Code
- **`pd.read_csv()`**: Reads the CSV file and stores it as a Pandas DataFrame.
- **`head()`**: Displays the first 5 records (Index 0 to 4).
- **`tail()`**: Displays the last 5 records (Index 995 to 999).
- **`shape`**: Returns a tuple `(1000, 8)` representing number of rows and columns.
- **`dtypes`**: Inspects data types assigned to each column (`object` or `int64`).
- **`info()`**: Prints complete structural metadata (non-null counts, types, memory usage).

---

## 8. Expected Output
```text
===========================================================================
 EXPERIMENT 1: IMPORT, LOAD, AND VIEW DATASET
===========================================================================

[INFO] Loading dataset from path: 'archive\StudentsPerformance.csv'...

[SUCCESS] Dataset loaded successfully into pandas DataFrame!

--- 1. First 5 Rows of the Dataset (head()) ---
   gender race/ethnicity parental level of education         lunch test preparation course  math score  reading score  writing score
0  female        group B           bachelor's degree      standard                    none          72             72             74
1  female        group C                some college      standard               completed          69             90             88
2  female        group B             master's degree      standard                    none          90             95             93
3    male        group A          associate's degree  free/reduced                    none          47             57             44
4    male        group C                some college      standard                    none          76             78             75

--- 2. Last 5 Rows of the Dataset (tail()) ---
     gender race/ethnicity parental level of education         lunch test preparation course  math score  reading score  writing score
995  female        group E             master's degree      standard               completed          88             99             95
996    male        group C                 high school  free/reduced                    none          62             55             55
997  female        group C                 high school  free/reduced               completed          59             71             65
998  female        group D                some college      standard               completed          68             78             77
999  female        group D                some college  free/reduced                    none          77             86             86

--- 3. Shape of the Dataset (Rows, Columns) ---
Number of Rows    : 1000
Number of Columns : 8
Shape (Rows, Cols): (1000, 8)

--- 4. All Column Names ---
 1. gender
 2. race/ethnicity
 3. parental level of education
 4. lunch
 5. test preparation course
 6. math score
 7. reading score
 8. writing score

--- 5. Data Type of Each Column ---
gender                           str
race/ethnicity                   str
parental level of education      str
lunch                            str
test preparation course          str
math score                     int64
reading score                  int64
writing score                  int64
dtype: object

--- 6. Complete Basic Information (info()) ---
<class 'pandas.DataFrame'>
RangeIndex: 1000 entries, 0 to 999
Data columns (total 8 columns):
 #   Column                       Non-Null Count  Dtype
---  ------                       --------------  -----
 0   gender                       1000 non-null   str  
 1   race/ethnicity               1000 non-null   str  
 2   parental level of education  1000 non-null   str  
 3   lunch                        1000 non-null   str  
 4   test preparation course      1000 non-null   str  
 5   math score                   1000 non-null   int64
 6   reading score                1000 non-null   int64
 7   writing score                1000 non-null   int64
dtypes: int64(3), str(5)
memory usage: 62.6 KB
```

---

## 9. Result
Experiment 1 was successfully executed. The dataset was imported, loaded, and verified to contain 1,000 rows and 8 columns with 0 missing values.
