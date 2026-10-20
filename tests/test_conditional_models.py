"""Checks against values derived by hand in the report."""

import numpy as np
import pytest

from src.conditional_models import (
    GAUSS_X, GAUSS_Y,
    average_log_loss, evaluate, fit_logistic_models, gaussian_nll, linear_mean,
)


def test_gaussian_nll_at_mean_is_normaliser():
    assert gaussian_nll(0.0, 0.0, 1.0) == pytest.approx(0.5 * np.log(2 * np.pi))


def test_gaussian_nll_rejects_nonpositive_variance():
    with pytest.raises(ValueError):
        gaussian_nll(0.0, 0.0, 0.0)


def test_task1_full_validation_set():
    assert average_log_loss(GAUSS_X, GAUSS_Y, 0.25) == pytest.approx(1.7258, abs=1e-4)
    assert average_log_loss(GAUSS_X, GAUSS_Y, 1.0) == pytest.approx(1.2939, abs=1e-4)


def test_task1_ranking_reverses_on_first_two_points():
    loss_a = average_log_loss(GAUSS_X[:2], GAUSS_Y[:2], 0.25)
    loss_b = average_log_loss(GAUSS_X[:2], GAUSS_Y[:2], 1.0)
    assert loss_a == pytest.approx(0.2258, abs=1e-4)
    assert loss_b == pytest.approx(0.9189, abs=1e-4)
    assert loss_a < loss_b


def test_task1_point_prediction():
    assert linear_mean(3.0) == pytest.approx(4.0)


def test_task2_quadratic_model_beats_linear():
    linear, quadratic = fit_logistic_models()
    m_lin, m_quad = evaluate(linear), evaluate(quadratic)
    assert m_quad["log_loss"] < m_lin["log_loss"]
    assert m_quad["accuracy"] == pytest.approx(1.0)
    assert m_lin["accuracy"] == pytest.approx(0.5)


def test_task2_symmetry_of_fitted_models():
    linear, quadratic = fit_logistic_models()
    assert abs(linear.coefficients["b"]) < 1e-6      # flat curve
    assert abs(quadratic.coefficients["b"]) < 1e-6   # symmetric U shape
    assert quadratic.coefficients["c"] > 0
