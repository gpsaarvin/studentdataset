# ==============================================================================
# ML / DATA SCIENCE PRACTICAL EXPERIMENT 4
# Title: Bayesian Logistic Regression and SVM for Classification
# Dataset: Student Performance Dataset (StudentsPerformance.csv)
# ==============================================================================

"""Classify whether a student completed the test preparation course.

Both classifiers are implemented with Python's standard library:
1. Bayesian logistic regression uses a Gaussian prior and a Laplace
   approximation to estimate posterior coefficient uncertainty.
2. A linear soft-margin SVM is trained with hinge-loss stochastic gradient
   descent.
"""

import csv
import math
import os
from typing import List, Sequence, Tuple


FEATURE_COLUMNS = ("math score", "reading score", "writing score")
TARGET_COLUMN = "test preparation course"


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


def load_dataset(path: str) -> Tuple[List[List[float]], List[int]]:
    """Load score features and encode completed=1, none=0."""
    features: List[List[float]] = []
    targets: List[int] = []

    with open(path, newline="", encoding="utf-8") as dataset_file:
        reader = csv.DictReader(dataset_file)
        required_columns = set(FEATURE_COLUMNS + (TARGET_COLUMN,))
        missing_columns = required_columns - set(reader.fieldnames or ())
        if missing_columns:
            raise ValueError(
                f"Dataset is missing required columns: {sorted(missing_columns)}"
            )

        for row in reader:
            try:
                features.append([float(row[column]) for column in FEATURE_COLUMNS])
            except (TypeError, ValueError) as error:
                raise ValueError("Dataset contains a non-numeric score value.") from error
            label = row[TARGET_COLUMN].strip().lower()
            if label not in {"completed", "none"}:
                raise ValueError(f"Unexpected target value: {row[TARGET_COLUMN]!r}")
            targets.append(1 if label == "completed" else 0)

    if not features:
        raise ValueError("The dataset does not contain any usable rows.")
    return features, targets


def standardize(
    train_features: Sequence[Sequence[float]],
    test_features: Sequence[Sequence[float]],
) -> Tuple[List[List[float]], List[List[float]]]:
    """Standardize features using training-set statistics only."""
    column_count = len(train_features[0])
    means = [
        sum(row[column] for row in train_features) / len(train_features)
        for column in range(column_count)
    ]
    standard_deviations = [
        math.sqrt(
            sum((row[column] - means[column]) ** 2 for row in train_features)
            / len(train_features)
        )
        or 1.0
        for column in range(column_count)
    ]

    def transform(rows: Sequence[Sequence[float]]) -> List[List[float]]:
        return [
            [
                (value - means[column]) / standard_deviations[column]
                for column, value in enumerate(row)
            ]
            for row in rows
        ]

    return transform(train_features), transform(test_features)


def sigmoid(value: float) -> float:
    """Return a numerically stable logistic probability."""
    if value >= 0:
        exponential = math.exp(-value)
        return 1.0 / (1.0 + exponential)
    exponential = math.exp(value)
    return exponential / (1.0 + exponential)


def dot(left: Sequence[float], right: Sequence[float]) -> float:
    return sum(left_value * right_value for left_value, right_value in zip(left, right))


def fit_bayesian_logistic_regression(
    features: Sequence[Sequence[float]],
    targets: Sequence[int],
    prior_precision: float = 1.0,
    class_weights: Tuple[float, float] = (1.0, 1.0),
) -> Tuple[List[float], List[List[float]]]:
    """Fit MAP logistic regression and estimate covariance with Laplace's method."""
    weights = [0.0] * (len(features[0]) + 1)
    learning_rate = 0.08

    for _ in range(2500):
        gradient = [0.0] * len(weights)
        for row, target in zip(features, targets):
            design_row = [1.0, *row]
            sample_weight = class_weights[target]
            residual = sample_weight * (sigmoid(dot(weights, design_row)) - target)
            for index, value in enumerate(design_row):
                gradient[index] += residual * value
        gradient[0] /= len(features)
        for index in range(1, len(weights)):
            gradient[index] = (
                gradient[index] / len(features) + prior_precision * weights[index]
            )
        weights = [
            weight - learning_rate * gradient_value
            for weight, gradient_value in zip(weights, gradient)
        ]

    hessian = [[0.0] * len(weights) for _ in weights]
    for row, target in zip(features, targets):
        design_row = [1.0, *row]
        probability = sigmoid(dot(weights, design_row))
        factor = class_weights[target] * probability * (1.0 - probability)
        for left in range(len(weights)):
            for right in range(len(weights)):
                hessian[left][right] += factor * design_row[left] * design_row[right]
    for index in range(1, len(weights)):
        hessian[index][index] += prior_precision * len(features)

    covariance = invert_matrix(hessian)
    return weights, covariance


def invert_matrix(matrix: List[List[float]]) -> List[List[float]]:
    """Invert a small matrix using Gauss-Jordan elimination with pivoting."""
    size = len(matrix)
    augmented = [
        row[:] + [1.0 if column == row_index else 0.0 for column in range(size)]
        for row_index, row in enumerate(matrix)
    ]
    for column in range(size):
        pivot = max(
            range(column, size), key=lambda row_index: abs(augmented[row_index][column])
        )
        if abs(augmented[pivot][column]) < 1e-12:
            raise ValueError("The posterior Hessian is singular.")
        augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        pivot_value = augmented[column][column]
        augmented[column] = [value / pivot_value for value in augmented[column]]
        for row in range(size):
            if row == column:
                continue
            factor = augmented[row][column]
            augmented[row] = [
                current - factor * pivot_value_value
                for current, pivot_value_value in zip(augmented[row], augmented[column])
            ]
    return [row[size:] for row in augmented]


def bayesian_predict(
    weights: Sequence[float],
    covariance: Sequence[Sequence[float]],
    features: Sequence[float],
) -> Tuple[float, float]:
    """Return posterior predictive probability and logit uncertainty."""
    design_row = [1.0, *features]
    logit = dot(weights, design_row)
    variance = max(
        0.0,
        sum(
            design_row[left] * covariance[left][right] * design_row[right]
            for left in range(len(design_row))
            for right in range(len(design_row))
        ),
    )
    # A logistic-Gaussian approximation accounts for posterior uncertainty.
    adjusted_probability = sigmoid(logit / math.sqrt(1.0 + math.pi * variance / 8.0))
    return adjusted_probability, math.sqrt(variance)


def fit_svm(
    features: Sequence[Sequence[float]],
    targets: Sequence[int],
    regularization: float = 0.01,
    class_weights: Tuple[float, float] = (1.0, 1.0),
) -> List[float]:
    """Train a linear SVM with hinge-loss stochastic gradient descent."""
    weights = [0.0] * (len(features[0]) + 1)
    signed_targets = [1 if target == 1 else -1 for target in targets]
    learning_rate = 0.01

    for epoch in range(80):
        for row, target in zip(features, signed_targets):
            design_row = [1.0, *row]
            sample_weight = class_weights[1 if target == 1 else 0]
            margin = target * dot(weights, design_row)
            weights[0] *= 1.0 - learning_rate * regularization
            for index in range(1, len(weights)):
                weights[index] *= 1.0 - learning_rate * regularization
            if margin < 1.0:
                for index, value in enumerate(design_row):
                    weights[index] += learning_rate * sample_weight * target * value
        learning_rate *= 0.98
    return weights


def calculate_metrics(actual: Sequence[int], predicted: Sequence[int]) -> Tuple[float, float, float]:
    """Calculate accuracy, precision, and recall."""
    true_positive = sum(value == 1 and estimate == 1 for value, estimate in zip(actual, predicted))
    true_negative = sum(value == 0 and estimate == 0 for value, estimate in zip(actual, predicted))
    false_positive = sum(value == 0 and estimate == 1 for value, estimate in zip(actual, predicted))
    false_negative = sum(value == 1 and estimate == 0 for value, estimate in zip(actual, predicted))
    accuracy = (true_positive + true_negative) / len(actual)
    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
    return accuracy, precision, recall


def main() -> None:
    print("=" * 75)
    print(" EXPERIMENT 4: BAYESIAN LOGISTIC REGRESSION AND SVM")
    print("=" * 75)

    dataset_path = locate_dataset()
    print(f"\n[INFO] Loading dataset from path: '{dataset_path}'...")
    features, targets = load_dataset(dataset_path)
    print(f"[SUCCESS] Loaded {len(features)} student records.\n")

    train_features = [row for index, row in enumerate(features) if index % 5 != 0]
    train_targets = [value for index, value in enumerate(targets) if index % 5 != 0]
    test_features = [row for index, row in enumerate(features) if index % 5 == 0]
    test_targets = [value for index, value in enumerate(targets) if index % 5 == 0]
    scaled_train, scaled_test = standardize(train_features, test_features)
    class_counts = [train_targets.count(label) for label in (0, 1)]
    class_weights = (
        len(train_targets) / (2.0 * class_counts[0]),
        len(train_targets) / (2.0 * class_counts[1]),
    )

    bayesian_weights, covariance = fit_bayesian_logistic_regression(
        scaled_train, train_targets, class_weights=class_weights
    )
    bayesian_results = [
        bayesian_predict(bayesian_weights, covariance, row) for row in scaled_test
    ]
    bayesian_predictions = [1 if probability >= 0.5 else 0 for probability, _ in bayesian_results]

    svm_weights = fit_svm(
        scaled_train, train_targets, class_weights=class_weights
    )
    svm_predictions = [
        1 if dot([1.0, *row], svm_weights) >= 0.0 else 0 for row in scaled_test
    ]

    print("--- 1. Classification Configuration ---")
    print(f"Input Features : {', '.join(FEATURE_COLUMNS)}")
    print(f"Target Feature : {TARGET_COLUMN} (completed=1, none=0)")
    print(f"Training Rows  : {len(scaled_train)}")
    print(f"Testing Rows   : {len(scaled_test)}")
    print(f"Class Weights  : none={class_weights[0]:.4f}, completed={class_weights[1]:.4f}")
    print()

    print("--- 2. Bayesian Logistic Regression ---")
    print(f"MAP Intercept  : {bayesian_weights[0]:.4f}")
    print(f"MAP Coefficients: {[round(value, 4) for value in bayesian_weights[1:]]}")
    average_uncertainty = sum(uncertainty for _, uncertainty in bayesian_results) / len(bayesian_results)
    print(f"Average logit uncertainty: {average_uncertainty:.4f}")
    accuracy, precision, recall = calculate_metrics(test_targets, bayesian_predictions)
    print(f"Accuracy       : {accuracy:.4f}")
    print(f"Precision      : {precision:.4f}")
    print(f"Recall         : {recall:.4f}")
    print()

    print("--- 3. Linear Support Vector Machine ---")
    print(f"Intercept      : {svm_weights[0]:.4f}")
    print(f"Coefficients   : {[round(value, 4) for value in svm_weights[1:]]}")
    accuracy, precision, recall = calculate_metrics(test_targets, svm_predictions)
    print(f"Accuracy       : {accuracy:.4f}")
    print(f"Precision      : {precision:.4f}")
    print(f"Recall         : {recall:.4f}")
    print()

    sample_features = [75.0, 80.0, 78.0]
    sample_scaled, _ = standardize(train_features, [sample_features])
    probability, uncertainty = bayesian_predict(
        bayesian_weights, covariance, sample_scaled[0]
    )
    svm_score = dot([1.0, *sample_scaled[0]], svm_weights)
    print("--- 4. Example Prediction ---")
    print(f"Scores (math, reading, writing): {sample_features}")
    print(f"Bayesian probability of completion: {probability:.4f}")
    print(f"Bayesian prediction: {'completed' if probability >= 0.5 else 'none'}")
    print(f"Posterior logit uncertainty: {uncertainty:.4f}")
    print(f"SVM decision score: {svm_score:.4f}")
    print(f"SVM prediction: {'completed' if svm_score >= 0.0 else 'none'}")
    print()

    print("=" * 75)
    print(" EXPERIMENT 4 COMPLETED SUCCESSFULLY!")
    print("=" * 75)


if __name__ == "__main__":
    main()
