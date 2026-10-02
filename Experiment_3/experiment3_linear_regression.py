# ==============================================================================
# ML / DATA SCIENCE PRACTICAL EXPERIMENT 3
# Title: Linear Regression for Student Score Prediction
# Dataset: Student Performance Dataset (StudentsPerformance.csv)
# ==============================================================================

"""Predict mathematics scores using reading and writing scores.

This experiment implements ordinary least-squares linear regression from
scratch so that it can run with Python's standard library alone.
"""

import csv
import os
from typing import List, Sequence, Tuple


FEATURE_COLUMNS = ("reading score", "writing score")
TARGET_COLUMN = "math score"


def locate_dataset() -> str:
    """Find the dataset when the script is run from common project folders."""
    script_directory = os.path.dirname(os.path.abspath(__file__))
    project_directory = os.path.dirname(script_directory)
    candidates = (
        os.path.join(project_directory, "archive", "StudentsPerformance.csv"),
        os.path.join(project_directory, "StudentsPerformance.csv"),
        os.path.join(os.getcwd(), "archive", "StudentsPerformance.csv"),
        os.path.join(os.getcwd(), "StudentsPerformance.csv"),
    )

    for path in candidates:
        if os.path.exists(path):
            return path

    raise FileNotFoundError(
        "StudentsPerformance.csv was not found. Expected it in the archive folder."
    )


def load_dataset(path: str) -> Tuple[List[List[float]], List[float]]:
    """Load the selected features and target values from the CSV file."""
    features: List[List[float]] = []
    targets: List[float] = []

    with open(path, newline="", encoding="utf-8") as dataset_file:
        reader = csv.DictReader(dataset_file)
        required_columns = set(FEATURE_COLUMNS + (TARGET_COLUMN,))
        available_columns = set(reader.fieldnames or ())
        missing_columns = required_columns - available_columns
        if missing_columns:
            raise ValueError(
                f"Dataset is missing required columns: {sorted(missing_columns)}"
            )

        for row in reader:
            try:
                features.append([float(row[column]) for column in FEATURE_COLUMNS])
                targets.append(float(row[TARGET_COLUMN]))
            except (TypeError, ValueError) as error:
                raise ValueError("Dataset contains a non-numeric score value.") from error

    if not features:
        raise ValueError("The dataset does not contain any usable rows.")
    return features, targets


def solve_linear_system(matrix: List[List[float]], values: List[float]) -> List[float]:
    """Solve a small linear system using Gaussian elimination with pivoting."""
    size = len(values)
    augmented = [row[:] + [values[index]] for index, row in enumerate(matrix)]

    for column in range(size):
        pivot = max(
            range(column, size), key=lambda row_index: abs(augmented[row_index][column])
        )
        if abs(augmented[pivot][column]) < 1e-12:
            raise ValueError("The feature matrix is singular and cannot be solved.")
        augmented[column], augmented[pivot] = augmented[pivot], augmented[column]

        pivot_value = augmented[column][column]
        augmented[column] = [
            value / pivot_value for value in augmented[column]
        ]

        for row in range(size):
            if row == column:
                continue
            factor = augmented[row][column]
            augmented[row] = [
                current - factor * pivot_value_value
                for current, pivot_value_value in zip(augmented[row], augmented[column])
            ]

    return [augmented[row][size] for row in range(size)]


def fit_linear_regression(features: Sequence[Sequence[float]], targets: Sequence[float]) -> List[float]:
    """Fit coefficients using the normal equation: beta = (X'X)^-1 X'y."""
    design_matrix = [[1.0, *row] for row in features]
    column_count = len(design_matrix[0])

    x_transpose_x = [
        [
            sum(row[left] * row[right] for row in design_matrix)
            for right in range(column_count)
        ]
        for left in range(column_count)
    ]
    x_transpose_y = [
        sum(row[column] * target for row, target in zip(design_matrix, targets))
        for column in range(column_count)
    ]
    return solve_linear_system(x_transpose_x, x_transpose_y)


def predict(coefficients: Sequence[float], features: Sequence[float]) -> float:
    """Return a prediction for one row of features."""
    return coefficients[0] + sum(
        coefficient * value
        for coefficient, value in zip(coefficients[1:], features)
    )


def calculate_metrics(actual: Sequence[float], predicted: Sequence[float]) -> Tuple[float, float, float]:
    """Calculate mean absolute error, mean squared error, and R-squared."""
    errors = [observed - estimate for observed, estimate in zip(actual, predicted)]
    mae = sum(abs(error) for error in errors) / len(errors)
    mse = sum(error * error for error in errors) / len(errors)
    mean_actual = sum(actual) / len(actual)
    total_sum_of_squares = sum((value - mean_actual) ** 2 for value in actual)
    r_squared = 1.0 - (sum(error * error for error in errors) / total_sum_of_squares)
    return mae, mse, r_squared


def main() -> None:
    print("=" * 75)
    print(" EXPERIMENT 3: LINEAR REGRESSION FOR SCORE PREDICTION")
    print("=" * 75)

    dataset_path = locate_dataset()
    print(f"\n[INFO] Loading dataset from path: '{dataset_path}'...")
    features, targets = load_dataset(dataset_path)
    print(f"[SUCCESS] Loaded {len(features)} student records.\n")

    # Use every fifth row for testing to keep the split deterministic.
    train_features = [row for index, row in enumerate(features) if index % 5 != 0]
    train_targets = [value for index, value in enumerate(targets) if index % 5 != 0]
    test_features = [row for index, row in enumerate(features) if index % 5 == 0]
    test_targets = [value for index, value in enumerate(targets) if index % 5 == 0]

    coefficients = fit_linear_regression(train_features, train_targets)
    predictions = [predict(coefficients, row) for row in test_features]
    mae, mse, r_squared = calculate_metrics(test_targets, predictions)

    print("--- 1. Model Configuration ---")
    print(f"Input Features : {', '.join(FEATURE_COLUMNS)}")
    print(f"Target Feature : {TARGET_COLUMN}")
    print(f"Training Rows  : {len(train_features)}")
    print(f"Testing Rows   : {len(test_features)}")
    print()

    print("--- 2. Learned Regression Equation ---")
    print(
        f"Predicted math score = {coefficients[0]:.4f} "
        f"+ ({coefficients[1]:.4f} x reading score) "
        f"+ ({coefficients[2]:.4f} x writing score)"
    )
    print()

    print("--- 3. Model Evaluation on Test Data ---")
    print(f"Mean Absolute Error (MAE) : {mae:.4f}")
    print(f"Mean Squared Error (MSE)  : {mse:.4f}")
    print(f"R-squared (R^2)           : {r_squared:.4f}")
    print()

    sample_features = [80.0, 85.0]
    sample_prediction = predict(coefficients, sample_features)
    print("--- 4. Example Prediction ---")
    print(f"Reading score: {sample_features[0]:.0f}")
    print(f"Writing score: {sample_features[1]:.0f}")
    print(f"Predicted math score: {sample_prediction:.2f}")
    print()

    print("=" * 75)
    print(" EXPERIMENT 3 COMPLETED SUCCESSFULLY!")
    print("=" * 75)


if __name__ == "__main__":
    main()
