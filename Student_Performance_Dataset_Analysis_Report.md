# Student Performance Dataset - Comprehensive Analysis Report

**Dataset File:** `StudentsPerformance.csv`  
**Total Records:** 1,000 Students  
**Total Variables:** 8 Attributes (5 Categorical, 3 Numerical)  
**Data Completeness:** 100% (0 Missing / Null Values)  

---

## 1. Executive Summary

This report presents a thorough exploratory and statistical analysis of the Kaggle **Student Performance Dataset**. The dataset measures student performance across three core academic subjects (**Mathematics**, **Reading**, and **Writing**) alongside five key demographic and socio-environmental factors (**Gender**, **Race/Ethnicity**, **Parental Level of Education**, **Lunch Type**, and **Test Preparation Course Status**).

### Key Takeaways:
1. **Highest Core Skills:** Students perform strongest in **Reading** (Mean: 69.17) followed closely by **Writing** (Mean: 68.05), while **Math** shows the lowest mean score (66.09) and the highest variability ($\sigma = 15.16$).
2. **Strongest Correlation:** **Reading and Writing scores exhibit an exceptionally high linear correlation ($r = 0.955$)**, indicating that literacy skills are deeply intertwined.
3. **Impact of Test Preparation:** Completing a test preparation course produces a significant performance boost across all subjects (+5.6 points in Math, +7.4 points in Reading, and +9.9 points in Writing).
4. **Socioeconomic Impact (Lunch Type):** Students receiving standard lunch outperform students receiving free/reduced lunch by over **11 points in Math** (70.03 vs 58.92) and **7.8 points in Writing** (70.82 vs 63.02).
5. **Gender Performance Divergence:** Female students excel significantly in Reading (72.61) and Writing (72.47), whereas male students outperform in Mathematics (68.73 vs 63.63).

---

## 2. Dataset Structure & Data Types

The dataset contains 1,000 instances with 8 columns.

| Column Name | Data Type | Kind | Sample Values |
| :--- | :--- | :--- | :--- |
| `gender` | String (`object`) | Categorical | `female`, `male` |
| `race/ethnicity` | String (`object`) | Categorical | `group B`, `group C`, `group A`, `group D`, `group E` |
| `parental level of education` | String (`object`) | Categorical | `bachelor's degree`, `some college`, `master's degree`, `associate's degree`, `high school`, `some high school` |
| `lunch` | String (`object`) | Categorical | `standard`, `free/reduced` |
| `test preparation course` | String (`object`) | Categorical | `none`, `completed` |
| `math score` | Integer (`int64`) | Numerical | `72`, `69`, `90`, `47` |
| `reading score` | Integer (`int64`) | Numerical | `72`, `90`, `95`, `57` |
| `writing score` | Integer (`int64`) | Numerical | `74`, `88`, `93`, `44` |

---

## 3. Statistical Summary & Central Tendency

### Overall Statistical Table

| Metric | Math Score | Reading Score | Writing Score |
| :--- | :---: | :---: | :---: |
| **Count** | 1,000 | 1,000 | 1,000 |
| **Mean** | **66.089** | **69.169** | **68.054** |
| **Median (50%)** | **66.000** | **70.000** | **69.000** |
| **Mode** | **65** | **72** | **74** |
| **Std. Deviation ($\sigma$)** | 15.163 | 14.600 | 15.196 |
| **Minimum** | 0 | 17 | 10 |
| **25th Percentile (Q1)** | 57.000 | 59.000 | 57.750 |
| **75th Percentile (Q3)** | 77.000 | 79.000 | 79.000 |
| **Maximum** | 100 | 100 | 100 |

### Insights from Descriptive Statistics:
- **Symmetry:** Math score mean (66.09) and median (66.00) are virtually identical, indicating a balanced, near-normal distribution.
- **Score Ceiling:** All three subjects reached the maximum score of 100.
- **Score Floor:** Math has a minimum score of 0, while Reading min is 17 and Writing min is 10.

---

## 4. Factor Analysis & Demographic Insights

### 4.1. Gender Wise Breakdown

| Gender | Count | % Share | Math Mean | Reading Mean | Writing Mean | Overall Mean |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Female** | 518 | 51.8% | 63.63 | **72.61** | **72.47** | **69.57** |
| **Male** | 482 | 48.2% | **68.73** | 65.47 | 63.31 | 65.84 |

- **Female Advantage:** Females score **+7.14 points higher in Reading** and **+9.16 points higher in Writing**.
- **Male Advantage:** Males score **+5.10 points higher in Mathematics**.

---

### 4.2. Test Preparation Course Effectiveness

| Test Prep Course | Student Count | Math Mean | Reading Mean | Writing Mean |
| :--- | :---: | :---: | :---: | :---: |
| **Completed** | 358 (35.8%) | **69.70** | **73.89** | **74.42** |
| **None** | 642 (64.2%) | 64.08 | 66.53 | 64.50 |
| **Difference (Gain)** | -- | **+5.62** | **+7.36** | **+9.92** |

- Completing test prep boosts **Writing scores by nearly 10 full marks**.
- Only 35.8% of students complete test preparation, highlighting an opportunity for intervention.

---

### 4.3. Socioeconomic Status (Lunch Type)

| Lunch Type | Count | Math Mean | Reading Mean | Writing Mean |
| :--- | :---: | :---: | :---: | :---: |
| **Standard** | 645 (64.5%) | **70.03** | **71.65** | **70.82** |
| **Free/Reduced** | 355 (35.5%) | 58.92 | 64.65 | 63.02 |
| **Performance Gap** | -- | **-11.11** | **-7.00** | **-7.80** |

- Lunch type acts as a proxy for family income level. Students from standard lunch households demonstrate a significant advantage across all three assessment subjects.

---

### 4.4. Parental Level of Education Impact

| Parental Education Level | Count | Math Mean | Reading Mean | Writing Mean | Overall Average |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Master's Degree** | 59 | **69.75** | **75.37** | **75.68** | **73.60** |
| **Bachelor's Degree** | 118 | 69.39 | 73.00 | 73.38 | 71.92 |
| **Associate's Degree** | 222 | 67.88 | 70.93 | 69.90 | 69.57 |
| **Some College** | 226 | 67.13 | 69.46 | 68.84 | 68.48 |
| **Some High School** | 179 | 63.50 | 66.94 | 64.89 | 65.11 |
| **High School** | 196 | 62.14 | 64.70 | 62.45 | 63.10 |

- There is a direct positive correlation between higher parental academic attainment and student test scores.
- Students whose parents hold a Master's degree score **~10.5 marks higher overall** than students whose parents completed high school.

---

### 4.5. Race / Ethnicity Group Breakdown

| Group | Student Count | Math Mean | Reading Mean | Writing Mean | Overall Mean |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Group E** | 140 | **73.82** | **73.03** | **71.41** | **72.75** |
| **Group D** | 262 | 67.36 | 70.03 | 70.15 | 69.18 |
| **Group C** | 319 | 64.46 | 69.10 | 67.83 | 67.13 |
| **Group B** | 190 | 63.45 | 67.35 | 65.60 | 65.47 |
| **Group A** | 89 | 61.63 | 64.67 | 62.67 | 62.99 |

- Group E records the highest performance in Mathematics (73.82) and overall score average (72.75).

---

## 5. Subject Correlation Analysis

| Subject Pair | Pearson Correlation ($r$) | Relationship Strength |
| :--- | :---: | :--- |
| **Reading Score $\leftrightarrow$ Writing Score** | **0.9546** | Extremely Strong Positive Correlation |
| **Math Score $\leftrightarrow$ Reading Score** | **0.8176** | Strong Positive Correlation |
| **Math Score $\leftrightarrow$ Writing Score** | **0.8026** | Strong Positive Correlation |

### Correlation Takeaways:
- **Literacy Coupling:** The $0.9546$ correlation between Reading and Writing indicates that improving reading comprehension directly enhances writing capability.
- **Cross-Domain Transfer:** High math performers also tend to be strong readers ($r = 0.8176$).

---

## 6. Data Quality & Data Integrity Audit

- **Null / Missing Values:** 0 null cells across all 8,000 data cells.
- **Duplicate Records:** 0 duplicated rows.
- **Data Types:** Categorical features are string object types; numerical scores are clean non-negative integers (`int64`).
- **Outlier Note:** 1 student achieved a Math score of 0, which represents an extreme lower outlier.

---

## 7. Actionable Recommendations

1. **Mandatory Test Prep Programs:** Encourage test preparation enrollment, as it increases writing scores by ~10 points and math scores by ~5.6 points.
2. **Targeted Math Support for Females & Writing Support for Males:**
   - Provide additional math tutoring programs tailored for female students.
   - Introduce focused writing workshops for male students.
3. **Nutritional & Socioeconomic Assistance:** Ensure students on free/reduced lunch have access to academic mentorship and learning support resources to help bridge the 11-point score gap.
