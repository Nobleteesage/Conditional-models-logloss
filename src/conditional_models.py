"""Conditional probabilistic models p_theta(y | x) and their evaluation.

This module contains the reusable pieces of the project:

* Gaussian conditional models with a shared mean and different variances,
  evaluated with the average negative log-likelihood (log loss).
* Logistic regression with raw and polynomial features, evaluated with
  validation log loss and accuracy.

Author: Goriola-Obafemi Babatunde Sukanmi
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss

# ---------------------------------------------------------------------------
# Data used in the study
# ---------------------------------------------------------------------------

#: Validation data for the Gaussian experiment (Task 1).
GAUSS_X = np.array([0.0, 1.0, 2.0])
GAUSS_Y = np.array([1.0, 2.0, 4.5])

#: Training and validation data for the classification experiment (Task 2).
CLF_X_TRAIN = np.array([-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2], dtype=float)
CLF_Y_TRAIN = np.array([1, 1, 0, 0, 0, 0, 0, 1, 1])
CLF_X_VAL = np.array([-1.75, -0.75, 0.75, 1.75])
CLF_Y_VAL = np.array([1, 0, 0, 1])


# ---------------------------------------------------------------------------
# Part 1: Gaussian conditional models
# ---------------------------------------------------------------------------

def gaussian_nll(y: np.ndarray, mean: np.ndarray, var: float) -> np.ndarray:
    r"""Pointwise negative log density of N(mean, var) at y.

    .. math::
        -\log p(y) = \tfrac12 \log(2\pi v) + \frac{(y - m)^2}{2v}
    """
    y = np.asarray(y, dtype=float)
    mean = np.asarray(mean, dtype=float)
    if var <= 0:
        raise ValueError("variance must be positive")
    return 0.5 * np.log(2 * np.pi * var) + (y - mean) ** 2 / (2 * var)


def linear_mean(x: np.ndarray) -> np.ndarray:
    """Conditional mean m(x) = 1 + x shared by both Gaussian models."""
    return 1.0 + np.asarray(x, dtype=float)


def average_log_loss(x: np.ndarray, y: np.ndarray, var: float) -> float:
    """Average log loss of the model Y | X = x ~ N(1 + x, var)."""
    return float(gaussian_nll(y, linear_mean(x), var).mean())


# ---------------------------------------------------------------------------
# Part 2: Logistic regression with nonlinear features
# ---------------------------------------------------------------------------

def linear_features(x: np.ndarray) -> np.ndarray:
    """Feature map x -> [x]."""
    return np.asarray(x, dtype=float).reshape(-1, 1)


def quadratic_features(x: np.ndarray) -> np.ndarray:
    """Feature map x -> [x, x^2]."""
    x = np.asarray(x, dtype=float)
    return np.column_stack([x, x ** 2])


FEATURE_MAPS = {
    "Model 1 (linear)": linear_features,
    "Model 2 (quadratic)": quadratic_features,
}


@dataclass
class FittedModel:
    """A logistic regression model together with its feature map."""

    name: str
    features: callable
    model: LogisticRegression

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        """Return p(1 | x)."""
        return self.model.predict_proba(self.features(x))[:, 1]

    @property
    def coefficients(self) -> dict[str, float]:
        coef = {"a": float(self.model.intercept_[0])}
        for name, value in zip(["b", "c"], self.model.coef_[0]):
            coef[name] = float(value)
        return coef


def fit_logistic_models(x: np.ndarray = CLF_X_TRAIN,
                        y: np.ndarray = CLF_Y_TRAIN,
                        C: float = 1.0) -> list[FittedModel]:
    """Fit Model 1 and Model 2 with L2 regularised logistic regression."""
    fitted = []
    for name, phi in FEATURE_MAPS.items():
        clf = LogisticRegression(C=C, max_iter=1000).fit(phi(x), y)
        fitted.append(FittedModel(name, phi, clf))
    return fitted


def evaluate(model: FittedModel,
             x: np.ndarray = CLF_X_VAL,
             y: np.ndarray = CLF_Y_VAL,
             threshold: float = 0.5) -> dict[str, float]:
    """Validation log loss and accuracy of a fitted model."""
    p = model.predict_proba(x)
    return {
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "accuracy": float(accuracy_score(y, (p >= threshold).astype(int))),
    }
