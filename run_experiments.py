"""Reproduce every number and figure reported in the README.

Usage:
    python run_experiments.py

Outputs:
    results/gaussian_log_loss.csv
    results/logistic_metrics.csv
    results/logistic_validation_predictions.csv
    figures/gaussian_models.png
    figures/logistic_fits.png
    figures/feature_space.png
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm

from src.conditional_models import (
    CLF_X_TRAIN, CLF_X_VAL, CLF_Y_TRAIN, CLF_Y_VAL,
    GAUSS_X, GAUSS_Y,
    average_log_loss, evaluate, fit_logistic_models, gaussian_nll, linear_mean,
)

ROOT = Path(__file__).resolve().parent
FIG = ROOT / "figures"
RES = ROOT / "results"
FIG.mkdir(exist_ok=True)
RES.mkdir(exist_ok=True)

GAUSS_MODELS = {"Model A (v = 0.25)": 0.25, "Model B (v = 1)": 1.0}
COLORS = ["#1f77b4", "#d62728"]

plt.rcParams.update({"figure.dpi": 120, "axes.grid": True, "grid.alpha": 0.3})


def gaussian_experiment() -> None:
    rows = []
    for name, v in GAUSS_MODELS.items():
        nll = gaussian_nll(GAUSS_Y, linear_mean(GAUSS_X), v)
        rows.append({
            "model": name, "variance": v,
            "nll_point_1": nll[0], "nll_point_2": nll[1], "nll_point_3": nll[2],
            "avg_log_loss_all_3": average_log_loss(GAUSS_X, GAUSS_Y, v),
            "avg_log_loss_first_2": average_log_loss(GAUSS_X[:2], GAUSS_Y[:2], v),
            "prediction_at_x3": float(linear_mean(3.0)),
        })
    df = pd.DataFrame(rows)
    df.to_csv(RES / "gaussian_log_loss.csv", index=False, float_format="%.4f")
    print("\nGaussian models\n", df.round(4).to_string(index=False))

    # Figure: predictive densities at x = 2 and pointwise log loss
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.3))
    grid = np.linspace(-0.5, 6.5, 500)
    for (name, v), col in zip(GAUSS_MODELS.items(), COLORS):
        ax1.plot(grid, norm.pdf(grid, loc=3, scale=np.sqrt(v)), color=col, label=name)
    ax1.axvline(4.5, color="k", ls="--", lw=1.2, label="observed $y_3 = 4.5$")
    ax1.axvline(3.0, color="gray", ls=":", lw=1, label="shared mean $m(2) = 3$")
    ax1.set_title("Predictive densities at $x = 2$")
    ax1.set_xlabel("$y$"); ax1.set_ylabel(r"$p(y \mid x = 2)$"); ax1.legend()

    width = 0.38
    pos = np.arange(3)
    for i, ((name, v), col) in enumerate(zip(GAUSS_MODELS.items(), COLORS)):
        vals = gaussian_nll(GAUSS_Y, linear_mean(GAUSS_X), v)
        bars = ax2.bar(pos + (i - 0.5) * width, vals, width, color=col, label=name)
        ax2.bar_label(bars, fmt="%.2f", fontsize=9)
    ax2.set_xticks(pos, [f"$(x, y) = ({x:g}, {y:g})$" for x, y in zip(GAUSS_X, GAUSS_Y)])
    ax2.set_ylabel(r"$-\log p(y_i \mid x_i)$")
    ax2.set_title("Pointwise log loss")
    ax2.legend()
    fig.tight_layout()
    fig.savefig(FIG / "gaussian_models.png", bbox_inches="tight")
    plt.close(fig)


def logistic_experiment() -> None:
    models = fit_logistic_models()

    metrics, preds = [], {"x": CLF_X_VAL, "y": CLF_Y_VAL}
    for m in models:
        row = {"model": m.name, **m.coefficients, **evaluate(m)}
        metrics.append(row)
        preds[f"p(1|x) {m.name}"] = m.predict_proba(CLF_X_VAL)
    mdf = pd.DataFrame(metrics)[["model", "a", "b", "c", "log_loss", "accuracy"]]
    pdf = pd.DataFrame(preds)
    mdf.to_csv(RES / "logistic_metrics.csv", index=False, float_format="%.4f")
    pdf.to_csv(RES / "logistic_validation_predictions.csv", index=False, float_format="%.4f")
    print("\nLogistic regression\n", mdf.round(4).to_string(index=False))
    print("\n", pdf.round(4).to_string(index=False))

    # Figure: fitted probability curves
    grid = np.linspace(-2.5, 2.5, 400)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    labels = [r"Model 1: $\sigma(a + bx)$", r"Model 2: $\sigma(a + bx + cx^2)$"]
    for m, lab, col in zip(models, labels, COLORS):
        ax.plot(grid, m.predict_proba(grid), color=col, lw=2, label=lab)
    ax.scatter(CLF_X_TRAIN, CLF_Y_TRAIN, c="k", zorder=3, label="training")
    ax.scatter(CLF_X_VAL, CLF_Y_VAL, facecolors="none", edgecolors="#2ca02c",
               s=90, lw=2, zorder=3, label="validation")
    ax.axhline(0.5, color="gray", ls=":", lw=1)
    ax.set_xlabel("$x$"); ax.set_ylabel(r"$p(1 \mid x)$")
    ax.set_title("Fitted logistic regression models")
    ax.legend(loc="upper center")
    fig.tight_layout()
    fig.savefig(FIG / "logistic_fits.png", bbox_inches="tight")
    plt.close(fig)

    # Figure: why x^2 helps (data in the (x, x^2) feature space)
    quad = models[1].coefficients
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for cls, mk, col in [(0, "o", COLORS[0]), (1, "s", COLORS[1])]:
        sel = CLF_Y_TRAIN == cls
        ax.scatter(CLF_X_TRAIN[sel], CLF_X_TRAIN[sel] ** 2, marker=mk, color=col,
                   s=60, label=f"class {cls}")
    xs = np.linspace(-2.5, 2.5, 200)
    # boundary a + b x + c z = 0  ->  z = -(a + b x) / c
    ax.plot(xs, -(quad["a"] + quad["b"] * xs) / quad["c"], "k--",
            label="Model 2 boundary (a line in feature space)")
    ax.set_xlabel("$x$"); ax.set_ylabel("$x^2$")
    ax.set_title("Classes become linearly separable in $(x, x^2)$")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG / "feature_space.png", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    gaussian_experiment()
    logistic_experiment()
    print(f"\nFigures saved to {FIG}\nResults saved to {RES}")
