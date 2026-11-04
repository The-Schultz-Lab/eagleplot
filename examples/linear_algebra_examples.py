"""Examples for linear algebra functions.

Demonstrates compute_eigenvalues(), solve_linear_system(), and matrix_decomposition()
with physically motivated examples.

Run from the repo root:
    python examples/linear_algebra_examples.py
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib

matplotlib.use("Agg")

sys.path.insert(0, str(Path(__file__).parent.parent))

from plottle.math import compute_eigenvalues, solve_linear_system, matrix_decomposition
from plottle.plotting import heatmap, save_figure

OUTPUT_DIR = Path(__file__).parent / "plots"
OUTPUT_DIR.mkdir(exist_ok=True)

CHECK = "[OK]"


def example_eigenvalues():
    """Example 1: Eigenvalues of a symmetric matrix (vibrational modes)."""
    print("\n" + "=" * 70)
    print("Example 1: Eigenvalues and Eigenvectors")
    print("=" * 70)

    # Force constant matrix (symmetric — real eigenvalues guaranteed)
    K = np.array(
        [
            [2.0, -1.0, 0.0],
            [-1.0, 2.0, -1.0],
            [0.0, -1.0, 2.0],
        ]
    )

    result = compute_eigenvalues(K)

    print("  Eigenvalues (vibrational frequencies^2):")
    for i, ev in enumerate(result["eigenvalues"]):
        print(f"    lambda_{i + 1} = {ev:.4f}")
    print("  Eigenvectors (columns are normal modes):")
    print(f"    {result['eigenvectors'].round(4)}")
    print(f"{CHECK} Eigenvalue decomposition complete")


def example_solve_linear_system():
    """Example 2: Solve a linear system Ax = b."""
    print("\n" + "=" * 70)
    print("Example 2: Solve Linear System Ax = b")
    print("=" * 70)

    # Equilibrium concentrations in a three-component system
    A = np.array(
        [
            [3.0, 1.0, 0.0],
            [1.0, 4.0, 1.0],
            [0.0, 1.0, 2.0],
        ]
    )
    b = np.array([9.0, 16.0, 5.0])

    x = solve_linear_system(A, b)

    print(f"  Solution x = {x.round(4)}")
    print(f"  Residual ||Ax - b|| = {np.linalg.norm(A @ x - b):.2e}")
    print(f"{CHECK} Linear system solved")


def example_qr_decomposition():
    """Example 3: QR decomposition."""
    print("\n" + "=" * 70)
    print("Example 3: QR Decomposition")
    print("=" * 70)

    np.random.seed(0)
    A = np.random.rand(4, 4)

    result = matrix_decomposition(A, method="qr")

    Q, R = result["Q"], result["R"]
    print(f"  Q shape: {Q.shape}  (orthonormal columns)")
    print(f"  R shape: {R.shape}  (upper triangular)")
    print(f"  ||Q^T Q - I|| = {np.linalg.norm(Q.T @ Q - np.eye(4)):.2e}")
    print(f"  ||QR - A||    = {np.linalg.norm(Q @ R - A):.2e}")
    print(f"{CHECK} QR decomposition complete")


def example_svd():
    """Example 4: SVD for low-rank approximation of a data matrix."""
    print("\n" + "=" * 70)
    print("Example 4: SVD — Low-Rank Approximation")
    print("=" * 70)

    np.random.seed(1)
    A = np.random.rand(8, 6)

    result = matrix_decomposition(A, method="svd")
    U, S, Vt = result["U"], result["S"], result["Vt"]

    # Rank-2 approximation
    rank = 2
    A_approx = U[:, :rank] @ np.diag(S[:rank]) @ Vt[:rank, :]
    error = np.linalg.norm(A - A_approx, "fro") / np.linalg.norm(A, "fro")

    print(f"  Singular values: {S.round(4)}")
    print(f"  Rank-{rank} relative error: {error:.4f}")

    fig, ax = heatmap(
        A_approx, title=f"Rank-{rank} SVD Approximation", xlabel="Column", ylabel="Row"
    )
    out = OUTPUT_DIR / "svd_approximation.png"
    save_figure(fig, out, dpi=150)
    print(f"  Saved: {out}")
    print(f"{CHECK} SVD complete")


if __name__ == "__main__":
    print("Plottle — Linear Algebra Examples")
    print("=" * 70)
    example_eigenvalues()
    example_solve_linear_system()
    example_qr_decomposition()
    example_svd()
    print("\nAll linear algebra examples complete.")
