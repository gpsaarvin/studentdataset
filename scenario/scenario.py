# ==============================================================================
# STUDENT ACADEMIC RISK PREDICTION SCENARIO
# Task: Compare Logistic Regression and Linear SVM Classification
# ==============================================================================

"""Predict whether a student is At Risk or Safe.

Class labels:
    0 = At Risk
    1 = Safe

The models are implemented with Python's standard library so this scenario
does not require additional packages.
"""

import math
from typing import List, Sequence, Tuple


FEATURE_NAMES = ("Attendance percentage", "Internal assessment mark", "Study hours per week")
FEATURES = [
    [60.0, 40.0, 2.0],
    [65.0, 45.0, 2.0],
    [70.0, 50.0, 3.0],
    [72.0, 52.0, 3.0],
    [80.0, 65.0, 4.0],
    [85.0, 70.0, 5.0],
    [90.0, 80.0, 6.0],
    [95.0, 85.0, 7.0],
]
TARGETS = [0, 0, 0, 0, 1, 1, 1, 1]
QUERY = [88.0, 75.0, 5.0]


def standardize(
    training_features: Sequence[Sequence[float]],
    rows: Sequence[Sequence[float]],
) -> List[List[float]]:
    """Scale columns using statistics calculated from the training data."""
    means = [
        sum(row[column] for row in training_features) / len(training_features)
        for column in range(len(training_features[0]))
    ]
    deviations = [
        math.sqrt(
            sum((row[column] - means[column]) ** 2 for row in training_features)
            / len(training_features)
        )
        or 1.0
        for column in range(len(means))
    ]
    return [
        [(value - means[column]) / deviations[column] for column, value in enumerate(row)]
        for row in rows
    ]


def dot(left: Sequence[float], right: Sequence[float]) -> float:
    return sum(left_value * right_value for left_value, right_value in zip(left, right))


def sigmoid(value: float) -> float:
    if value >= 0:
        exponential = math.exp(-value)
        return 1.0 / (1.0 + exponential)
    exponential = math.exp(value)
    return exponential / (1.0 + exponential)


def fit_logistic_regression(
    features: Sequence[Sequence[float]],
    targets: Sequence[int],
) -> List[float]:
    """Fit logistic regression with gradient descent and L2 regularization."""
    weights = [0.0] * (len(features[0]) + 1)
    learning_rate = 0.15
    regularization = 0.01

    for _ in range(3000):
        gradient = [0.0] * len(weights)
        for row, target in zip(features, targets):
            design_row = [1.0, *row]
            error = sigmoid(dot(weights, design_row)) - target
            for index, value in enumerate(design_row):
                gradient[index] += error * value

        gradient = [
            gradient[0] / len(features),
            *[
                gradient[index] / len(features) + regularization * weights[index]
                for index in range(1, len(weights))
            ],
        ]
        weights = [
            weight - learning_rate * gradient_value
            for weight, gradient_value in zip(weights, gradient)
        ]
    return weights


def logistic_probabilities(
    weights: Sequence[float],
    features: Sequence[float],
) -> Tuple[float, float]:
    probability_safe = sigmoid(dot(weights, [1.0, *features]))
    return 1.0 - probability_safe, probability_safe


def fit_linear_svm(
    features: Sequence[Sequence[float]],
    targets: Sequence[int],
) -> List[float]:
    """Fit a linear soft-margin SVM with hinge-loss gradient descent."""
    weights = [0.0] * (len(features[0]) + 1)
    learning_rate = 0.02
    regularization = 0.01
    signed_targets = [1 if target == 1 else -1 for target in targets]

    for _ in range(1500):
        for row, target in zip(features, signed_targets):
            design_row = [1.0, *row]
            margin = target * dot(weights, design_row)
            weights = [
                weight * (1.0 - learning_rate * regularization)
                for weight in weights
            ]
            if margin < 1.0:
                weights = [
                    weight + learning_rate * target * value
                    for weight, value in zip(weights, design_row)
                ]
        learning_rate *= 0.998
    return weights


def class_name(label: int) -> str:
    return "Safe" if label == 1 else "At Risk"


def main() -> None:
    print("=" * 75)
    print(" STUDENT ACADEMIC RISK PREDICTION SCENARIO")
    print("=" * 75)
    print("\n--- 1. Dataset ---")
    print(f"Features: {', '.join(FEATURE_NAMES)}")
    print(f"Records : {len(FEATURES)}")
    print("Labels  : 0 = At Risk, 1 = Safe")

    scaled_features = standardize(FEATURES, FEATURES)
    scaled_query = standardize(FEATURES, [QUERY])[0]

    logistic_weights = fit_logistic_regression(scaled_features, TARGETS)
    logistic_risk_probability, logistic_safe_probability = logistic_probabilities(
        logistic_weights, scaled_query
    )
    logistic_prediction = int(logistic_safe_probability >= 0.5)

    svm_weights = fit_linear_svm(scaled_features, TARGETS)
    svm_decision_score = dot(svm_weights, [1.0, *scaled_query])
    svm_prediction = int(svm_decision_score >= 0.0)

    print("\n--- 2. Logistic Regression ---")
    print(f"Coefficients: {[round(value, 4) for value in logistic_weights]}")
    print(f"Probability of At Risk: {logistic_risk_probability:.4f}")
    print(f"Probability of Safe   : {logistic_safe_probability:.4f}")
    print(f"Prediction            : {class_name(logistic_prediction)}")

    print("\n--- 3. Linear SVM ---")
    print(f"Coefficients : {[round(value, 4) for value in svm_weights]}")
    print(f"Decision score: {svm_decision_score:.4f}")
    print(f"Prediction     : {class_name(svm_prediction)}")

    print("\n--- 4. Query Prediction ---")
    print(f"Attendance percentage : {QUERY[0]:.0f}")
    print(f"Internal mark        : {QUERY[1]:.0f}")
    print(f"Study hours per week : {QUERY[2]:.0f}")

    print("\n--- 5. Model Comparison ---")
    print(f"Logistic Regression: {class_name(logistic_prediction)}")
    print(f"SVM                : {class_name(svm_prediction)}")
    if logistic_prediction == svm_prediction:
        print(f"Result             : Both models agree - {class_name(logistic_prediction)}")
    else:
        print("Result             : The models disagree; review both scores.")

    print("\n" + "=" * 75)
    print(" SCENARIO COMPLETED SUCCESSFULLY!")
    print("=" * 75)


if __name__ == "__main__":
    main()
