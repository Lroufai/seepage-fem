"""
01_column_homogeneous.py
========================
Steady saturated flow through a homogeneous 1D soil column.

    d/dz ( K dh/dz ) = 0,   h(0) = h0,   h(L) = hL

The problem is solved twice:
  (a) with a hand-written finite element assembly (2 x 2 element matrices),
  (b) with the scikit-fem library,
and both are compared with the exact (linear) solution.

Expected result: the error is at machine precision (~1e-15 m), because for this
equation linear elements are exact at the nodes in 1D.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from skfem import MeshLine, ElementLineP1, Basis, BilinearForm, asm, condense, solve
from skfem.helpers import dot, grad

import plotstyle

os.makedirs("figures", exist_ok=True)

# ---------------------------------------------------------------- input data
L = 5.0        # column height (m)
K = 1e-4       # hydraulic conductivity (m/s)
h0 = 3.0       # prescribed head at the bottom, z = 0 (m)
hL = 0.0       # prescribed head at the top, z = L (m)
n_elem = 20    # number of elements


# ------------------------------------------------- (a) hand-written assembly
def solve_manual(L, K, h0, hL, n_elem):
    """Assemble the global matrix element by element and solve for the heads."""
    z = np.linspace(0.0, L, n_elem + 1)          # node elevations
    dz = z[1] - z[0]                              # element length
    N = n_elem + 1                                # number of nodes

    A = np.zeros((N, N))
    Ke = (K / dz) * np.array([[1.0, -1.0],        # element matrix: Darcy flux
                              [-1.0, 1.0]])       # between the element's two nodes
    for e in range(n_elem):
        nodes = [e, e + 1]                        # local -> global node numbers
        for i in range(2):
            for j in range(2):
                A[nodes[i], nodes[j]] += Ke[i, j] # "+=": shared nodes accumulate

    b = np.zeros(N)                               # no sources or sinks

    # Dirichlet conditions: keep the interior equations, move known heads to the RHS
    A_int = A[1:-1, 1:-1]
    b_int = b[1:-1] - A[1:-1, 0] * h0 - A[1:-1, -1] * hL

    h = np.zeros(N)
    h[0], h[-1] = h0, hL
    h[1:-1] = np.linalg.solve(A_int, b_int)
    return z, h, A


# ------------------------------------------------------------ (b) scikit-fem
def solve_skfem(L, K, h0, hL, n_elem):
    """Same problem with scikit-fem: the weak form is written as on paper."""
    mesh = MeshLine(np.linspace(0.0, L, n_elem + 1))
    basis = Basis(mesh, ElementLineP1())          # piecewise-linear "hat" functions

    @BilinearForm
    def darcy(h, v, w):
        return K * dot(grad(h), grad(v))          # integral of K h' v'

    A = asm(darcy, basis)
    b = np.zeros(basis.N)

    z = mesh.p[0]
    bottom = np.where(np.isclose(z, 0.0))[0]
    top = np.where(np.isclose(z, L))[0]
    h = np.zeros(basis.N)
    h[bottom], h[top] = h0, hL
    D = np.concatenate([bottom, top])             # nodes with prescribed head

    h = solve(*condense(A, b, x=h, D=D))
    return z, h


if __name__ == "__main__":
    plotstyle.apply()

    z_a, h_a, A_small = solve_manual(3.0, 1.0, 3.0, 0.0, 3)
    print("Global matrix for the 3-element example (K = 1, dz = 1):")
    print(A_small)
    print("Heads:", h_a, "\n")

    z_m, h_m, _ = solve_manual(L, K, h0, hL, n_elem)
    z_s, h_s = solve_skfem(L, K, h0, hL, n_elem)

    exact = lambda z: h0 + (hL - h0) * z / L
    err_m = np.max(np.abs(h_m - exact(z_m)))
    err_s = np.max(np.abs(h_s - exact(z_s)))
    order = np.argsort(z_s)
    diff = np.max(np.abs(h_m - h_s[order]))

    print(f"Max error, hand-written assembly : {err_m:.2e} m")
    print(f"Max error, scikit-fem            : {err_s:.2e} m")
    print(f"Max difference between the two   : {diff:.2e} m")

    # Darcy flux in every element: must be identical everywhere (mass conservation)
    q = -K * np.diff(h_m) / np.diff(z_m)
    print(f"Darcy flux per element: min {q.min():.6e}, max {q.max():.6e} m/s")

    fig, ax = plt.subplots(figsize=(4.8, 5.2))
    zz = np.linspace(0, L, 200)
    ax.plot(exact(zz), zz, color=plotstyle.C2, label="Analytical solution")
    ax.plot(h_m, z_m, "o", color=plotstyle.C1, mfc="none", mew=1.6,
            label="FEM, hand-written assembly")
    ax.plot(h_s[order], z_s[order], "+", color=plotstyle.INK, ms=7,
            label="FEM, scikit-fem")
    ax.set_xlabel("Hydraulic head h (m)")
    ax.set_ylabel("Elevation z (m)")
    ax.set_title("Homogeneous column, K = 1e-4 m/s")
    ax.legend(loc="upper right")
    fig.savefig("figures/01_column_homogeneous.png")
    print("Figure saved: figures/01_column_homogeneous.png")
