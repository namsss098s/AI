import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import r2_score


def plot_actual_vs_predicted(y_test, y_pred, save_path=None):
    """
    Plot Actual vs Predicted Insurance Cost
    """

    y_test = np.array(y_test)
    y_pred = np.array(y_pred)

    r2 = r2_score(y_test, y_pred)

    plt.figure(figsize=(7, 7))
    plt.scatter(y_test, y_pred, alpha=0.6)

    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())

    plt.plot([min_val, max_val], [min_val, max_val], linestyle="--")

    plt.xlabel("Actual Insurance Cost")
    plt.ylabel("Predicted Insurance Cost")
    plt.title(f"Actual vs Predicted Insurance Cost\nR² = {r2:.4f}")

    plt.grid(True)

    plt.gca().set_aspect('equal', adjustable='box')

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")

    plt.show()


def plot_residuals(y_test, y_pred, save_path=None):
    """
    Plot Residual Scatter
    """

    y_test = np.array(y_test)
    y_pred = np.array(y_pred)

    residuals = y_test - y_pred

    plt.figure(figsize=(7, 5))
    plt.scatter(y_pred, residuals, alpha=0.6)
    plt.axhline(y=0, linestyle="--")

    plt.xlabel("Predicted Insurance Cost")
    plt.ylabel("Residual (Actual - Predicted)")
    plt.title("Residual Plot")

    plt.grid(True)

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")

    plt.show()


def plot_residual_distribution(y_test, y_pred, save_path=None):
    """
    Plot histogram of residuals
    """

    residuals = np.array(y_test) - np.array(y_pred)

    plt.figure(figsize=(7, 5))
    plt.hist(residuals, bins=30)

    plt.xlabel("Residual")
    plt.ylabel("Frequency")
    plt.title("Residual Distribution")

    plt.grid(True)

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")

    plt.show()