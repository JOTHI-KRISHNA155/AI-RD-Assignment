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


# Dataset Loading


def load_dataset(file_path):


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


def generate_curve(t_samples, theta_deg, m_value, x_offset):
    """
    Generate the predicted parametric curve using the
    mathematical model provided in the assignment.

    Parameters
    ----------
    t_samples : numpy.ndarray
        Uniformly sampled values of the parameter t.

    theta_deg : float
        Rotation angle in degrees.

    m_value : float
        Exponential growth parameter.

    x_offset : float
        Horizontal translation of the curve.

    Returns
    -------
    tuple
        Predicted x and y coordinates.
    """

    theta_rad = np.radians(theta_deg)

    oscillation = (
        np.exp(m_value * np.abs(t_samples))
        * np.sin(0.3 * t_samples)
    )

    predicted_x = (
        t_samples * np.cos(theta_rad)
        - oscillation * np.sin(theta_rad)
        + x_offset
    )

    predicted_y = (
        42
        + t_samples * np.sin(theta_rad)
        + oscillation * np.cos(theta_rad)
    )

    return predicted_x, predicted_y



# Loss Function


def compute_loss(parameters, observed_x, observed_y):
    """
    Compute the total L1 distance between the observed
    curve and the predicted curve.

    Parameters
    ----------
    parameters : tuple
        Candidate values of (theta, M, X).

    observed_x : numpy.ndarray
        Observed x coordinates.

    observed_y : numpy.ndarray
        Observed y coordinates.

    Returns
    -------
    float
        Total L1 loss.
    """

    theta_deg, m_value, x_offset = parameters

    
    t_samples = np.linspace(
        T_MIN,
        T_MAX,
        NUM_SAMPLES,
    )

    
    predicted_x, predicted_y = generate_curve(
        t_samples,
        theta_deg,
        m_value,
        x_offset,
    )

    total_loss = 0.0

    
    for x_real, y_real in zip(observed_x, observed_y):

        distances = (
            np.abs(predicted_x - x_real)
            + np.abs(predicted_y - y_real)
        )

        total_loss += np.min(distances)

    return total_loss



def estimate_parameters(observed_x, observed_y):
    """
    Estimate the unknown parameters using Differential Evolution.

    Parameters
    ----------
    observed_x : numpy.ndarray
        Observed x coordinates.

    observed_y : numpy.ndarray
        Observed y coordinates.

    Returns
    -------
    OptimizeResult
        Result returned by scipy.optimize.differential_evolution.
    """

    parameter_bounds = [

        THETA_RANGE,

        M_RANGE,

        X_RANGE,

    ]

    result = differential_evolution(

        func=compute_loss,

        bounds=parameter_bounds,

        args=(observed_x, observed_y),

        maxiter=MAX_ITERATIONS,

        popsize=POPULATION_SIZE,

        tol=TOLERANCE,

        seed=RANDOM_SEED,

        polish=True,

    )

    return result




def plot_results(
    observed_x,
    observed_y,
    theta_deg,
    m_value,
    x_offset,
):
    """
    Plot the observed data and the predicted curve.

    The figure is automatically saved inside the
    results directory.
    """

    t_samples = np.linspace(
        T_MIN,
        T_MAX,
        PLOT_SAMPLES,
    )

    predicted_x, predicted_y = generate_curve(
        t_samples,
        theta_deg,
        m_value,
        x_offset,
    )

    plt.figure(figsize=(8,8))

    plt.scatter(
        observed_x,
        observed_y,
        s=10,
        label="Observed Data",
    )

    plt.plot(
        predicted_x,
        predicted_y,
        color="red",
        linewidth=2,
        label="Predicted Curve",
    )

    plt.xlabel("X Coordinate")

    plt.ylabel("Y Coordinate")

    plt.title("Observed vs Predicted Parametric Curve")

    plt.axis("equal")

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            RESULT_FOLDER,
            "plot.png",
        ),
        dpi=300,
    )

    plt.close()



def save_results(
    theta_deg,
    m_value,
    x_offset,
    loss_value,
):
    """
    Save the estimated parameters into a text file.
    """

    output_path = os.path.join(
        RESULT_FOLDER,
        "result.txt",
    )

    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as file:

        file.write(
            "Parameter Estimation Results\n"
        )

        file.write(
            "=" * 35 + "\n\n"
        )

        file.write(
            f"Theta : {theta_deg:.6f} degrees\n"
        )

        file.write(
            f"M     : {m_value:.6f}\n"
        )

        file.write(
            f"X     : {x_offset:.6f}\n\n"
        )

        file.write(
            f"Final L1 Loss : {loss_value:.6f}\n"
        )




def main():

    print("=" * 60)
    print("AI R&D Assignment")
    print("Parameter Estimation")
    print("=" * 60)

    observed_x, observed_y = load_dataset(DATA_FILE)

    result = estimate_parameters(
        observed_x,
        observed_y,
    )

    theta_deg, m_value, x_offset = result.x

    plot_results(
        observed_x,
        observed_y,
        theta_deg,
        m_value,
        x_offset,
    )

    save_results(
        theta_deg,
        m_value,
        x_offset,
        result.fun,
    )

    print("\nEstimated Parameters\n")

    print(f"Theta : {theta_deg:.6f}")

    print(f"M     : {m_value:.6f}")

    print(f"X     : {x_offset:.6f}")

    print(f"\nFinal L1 Loss : {result.fun:.6f}")

    print("\nResults saved inside the 'results' folder.")


if __name__ == "__main__":
    main()