"""
====================================================================
AI R&D Assignment
Parameter Estimation of a Parametric Curve
====================================================================

Author      : Jothi Krishna
Language    : Python 3.x
Description :
    This program estimates the unknown parameters of a parametric
    curve using Differential Evolution optimization.

Unknown Parameters
------------------
Theta (θ)
M
X

Dataset
-------
data/xy_data.csv

Output
------
results/result.txt
results/plot.png

====================================================================
"""

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.optimize import differential_evolution


# ==========================================================
# Configuration
# ==========================================================

DATA_FILE = "data/xy_data.csv"

RESULT_FOLDER = "results"

THETA_RANGE = (0.0, 50.0)
M_RANGE = (-0.05, 0.05)
X_RANGE = (0.0, 100.0)

T_MIN = 6.0
T_MAX = 60.0

NUM_SAMPLES = 1000
PLOT_SAMPLES = 3000

MAX_ITERATIONS = 100
POPULATION_SIZE = 20
TOLERANCE = 1e-6

RANDOM_SEED = 42


os.makedirs(RESULT_FOLDER, exist_ok=True)

# ==========================================================
# Dataset Loading
# ==========================================================

def load_dataset(file_path):
    """
    Load the observed curve points from the CSV file.

    Parameters
    ----------
    file_path : str
        Path to the CSV file.

    Returns
    -------
    tuple
        Two NumPy arrays containing the observed x and y coordinates.
    """

    try:
        data = pd.read_csv(file_path)

    except FileNotFoundError:
        raise FileNotFoundError(
            f"Dataset not found: {file_path}"
        )

    required_columns = {"x", "y"}

    if not required_columns.issubset(data.columns):
        raise ValueError(
            "CSV file must contain 'x' and 'y' columns."
        )

    observed_x = data["x"].to_numpy(dtype=float)
    observed_y = data["y"].to_numpy(dtype=float)

    return observed_x, observed_y


if __name__ == "__main__":

    x, y = load_dataset(DATA_FILE)

    print(f"Loaded {len(x)} points.")

    print()

    print("First five x values:")
    print(x[:5])

    print()

    print("First five y values:")
    print(y[:5])