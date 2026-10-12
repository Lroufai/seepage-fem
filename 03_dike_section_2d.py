"""
03_dike_section_2d.py
=====================
Vertical 2D cross-section under a river dike during a flood (steady state).

Geometry (x horizontal, z vertical, datum at the base of the aquifer):

    river          dike (impermeable)            polder
  h = h_river   |========================|  h = h_polder at ground surface
  ~~~~~~~~~~~~~~+------------------------+------------------------------------
                                         |   clay blanket (d = 1.5 m, K_b)
  ---------------------------------------+------------------------------------
                  sand aquifer (D = 10 m, K_a)
  ----------------------------------------------------------------------------
                          impermeable base (no flow)

Boundary conditions
  * river bed (x <= X_RIVER, top of the sand) : h = h_river
  * polder surface (x >= X_TOE, top of blanket): h = h_polder
  * everything else (base, sides, dike footprint): no flow, imposed "for free"
    by the weak form (natural boundary condition).

Outputs
  1. head field and flow direction in the section,
  2. head under the blanket along the polder, compared with the classic
     leakage-length (blanket) theory used in Dutch dike assessment,
  3. uplift factor of safety along the polder,
  4. mass balance (inflow from the river = outflow through the blanket),
  5. mesh-refinement check of the head at the landward toe.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from skfem import MeshTri, ElementTriP1, Basis, BilinearForm, asm, condense, solve
from skfem.helpers import dot, grad

import plotstyle

os.makedirs("figures", exist_ok=True)

# ---------------------------------------------------------------- input data
D = 10.0           # aquifer thickness (m)
d = 1.5            # blanket thickness (m)
K_A = 1e-4         # aquifer conductivity (m/s)
K_B = 1e-7         # blanket conductivity (m/s)
X_RIVER = 50.0     # end of the exposed river bed (m)
X_TOE = 90.0       # landward toe of the dike (m): seepage length L = 40 m
X_END = 900.0      # end of the polder in the model (m)
H_POLDER = D + d   # polder water table at ground surface (m)
DH = 4.0           # river level above the polder level (m)
H_RIVER = H_POLDER + DH
GAMMA_W = 9.81     # kN/m3
GAMMA_SAT = 17.5   # saturated unit weight of the blanket (kN/m3)


def build_mesh(refine=1):
    """Structured triangular mesh, refined around the toe, aligned with the layers."""
    r = refine
    x = np.unique(np.concatenate([
        np.linspace(0.0, X_RIVER, int(25 * r) + 1),
        np.linspace(X_RIVER, X_TOE - 10, int(30 * r) + 1),
        np.linspace(X_TOE - 10, X_TOE + 10, int(80 * r) + 1),   # 0.25 m around the toe
        np.linspace(X_TOE + 10, 300.0, int(95 * r) + 1),
        np.linspace(300.0, X_END, int(60 * r) + 1),
    ]))
    z = np.unique(np.concatenate([
        np.linspace(0.0, D, int(20 * r) + 1),
        np.linspace(D, D + d, int(6 * r) + 1),                  # 0.25 m in the blanket
    ]))
    mesh = MeshTri.init_tensor(x, z)

    # remove the blanket where there is none: river side and under the dike
    centroids = mesh.p[:, mesh.t].mean(axis=1)
    no_blanket = (centroids[1] > D) & (centroids[0] < X_TOE)
    return mesh.remove_elements(np.nonzero(no_blanket)[0])


def solve_section(mesh):
    basis = Basis(mesh, ElementTriP1())

    @BilinearForm
    def darcy(h, v, w):
        K = np.where(w.x[1] < D, K_A, K_B)   # sand below z = D, clay above
        return K * dot(grad(h), grad(v))

    A = asm(darcy, basis)
    x, z = mesh.p
    river = np.nonzero(np.isclose(z, D) & (x <= X_RIVER + 1e-9))[0]
    polder = np.nonzero(np.isclose(z, D + d) & (x >= X_TOE - 1e-9))[0]

    h = np.zeros(basis.N)
    h[river] = H_RIVER
    h[polder] = H_POLDER
    D_nodes = np.concatenate([river, polder])
    h = solve(*condense(A, np.zeros(basis.N), x=h, D=D_nodes))

    # Mass balance from the reaction forces: (A h)_i is the flow entering at node i
    r = A @ h
    q_in = r[river].sum()            # m3/s per metre of dike, entering from the river
    q_out = -r[polder].sum()         # leaving through the blanket
    return basis, h, q_in, q_out


def leakage_theory(x):
    """Classic blanket (leakage-length) theory, 1D, used in Dutch piping assessment.

    Leakage length   lambda = sqrt(K_A * D * d / K_B)
    Under the dike   flow K_A * D * (h_river - h_toe) / L
    Under the polder head decays exponentially towards the polder level.
    """
    lam = np.sqrt(K_A * D * d / K_B)
    L = X_TOE - X_RIVER
    phi_toe = H_POLDER + DH * lam / (L + lam)
    return H_POLDER + (phi_toe - H_POLDER) * np.exp(-(x - X_TOE) / lam), lam, phi_toe


def head_under_blanket(mesh, h, xs):
    """Head along z = D (base of the blanket), interpolated from the nodes."""
    x, z = mesh.p
    on_line = np.nonzero(np.isclose(z, D))[0]
    order = np.argsort(x[on_line])
    return np.interp(xs, x[on_line][order], h[on_line][order])


if __name__ == "__main__":
    plotstyle.apply()

    mesh = build_mesh(refine=1)
    basis, h, q_in, q_out = solve_section(mesh)
    print(f"Mesh: {mesh.p.shape[1]} nodes, {mesh.t.shape[1]} triangles")

    # ------------------------------------------------------- mass balance
    print("\n=== Mass balance (per metre of dike) ===")
    print(f"Inflow from the river   : {q_in:.4e} m3/s/m")
    print(f"Outflow through blanket : {q_out:.4e} m3/s/m")
    print(f"Relative imbalance      : {abs(q_in - q_out) / q_in:.1e}")

    # ------------------------------------- head under the blanket vs theory
    xs = np.linspace(X_TOE, X_END, 800)
    h_fem = head_under_blanket(mesh, h, xs)
    h_th, lam, phi_toe = leakage_theory(xs)
    h_toe = h_fem[0]
    print("\n=== Head under the blanket at the landward toe ===")
    print(f"Leakage length lambda   : {lam:.1f} m")
    print(f"FEM                     : {h_toe:.3f} m")
    print(f"Leakage-length theory   : {phi_toe:.3f} m")
    print(f"Difference              : {h_toe - phi_toe:+.3f} m")
    rel_diff = np.max(np.abs(h_fem - h_th)) / DH
    print(f"Max difference along the polder, as a fraction of dH: {rel_diff:.1%}")

    # ------------------------------------------------ uplift along the polder
    u = GAMMA_W * (h_fem - D)                  # pore pressure under the blanket (kPa)
    fs = GAMMA_SAT * d / u
    i_blanket = (h_fem - H_POLDER) / d
    unsafe = xs[fs < 1.0]
    print("\n=== Uplift of the blanket ===")
    print(f"FS uplift at the toe    : {fs[0]:.2f}")
    print(f"Vertical gradient at toe: {i_blanket[0]:.2f}")
    if unsafe.size:
        print(f"FS < 1 from the toe up to {unsafe.max() - X_TOE:.0f} m into the polder")

    # ---------------------------------------------------- mesh refinement
    print("\n=== Mesh refinement: head at the toe ===")
    toe_heads = {}
    for r in (1, 2):
        m_r = build_mesh(refine=r)
        _, h_r, qi, qo = solve_section(m_r)
        toe_heads[r] = head_under_blanket(m_r, h_r, np.array([X_TOE]))[0]
        print(f"  refine x{r}: {m_r.p.shape[1]:6d} nodes, h_toe = {toe_heads[r]:.4f} m, "
              f"Q = {qi:.4e} m3/s/m")
    print(f"Change in h_toe after refinement: {abs(toe_heads[2] - toe_heads[1]) * 1000:.1f} mm")

    # ============================================================== figures
    x, z = mesh.p
    tri = mtri.Triangulation(x, z, mesh.t.T)

    # Figure 1: head field (vertical exaggeration on the first 250 m)
    fig, ax = plt.subplots(figsize=(10, 3.8))
    levels = np.linspace(H_POLDER, H_RIVER, 17)
    cf = ax.tricontourf(tri, h, levels=levels, cmap="Blues")
    ax.tricontour(tri, h, levels=levels[::2], colors="white", linewidths=0.6)
    # flow paths: streamlines of the Darcy velocity in the aquifer, started at the river bed
    interp = mtri.LinearTriInterpolator(tri, h)
    xg, zg = np.meshgrid(np.linspace(0, 250, 500), np.linspace(0.02, D - 0.02, 60))
    dhdx, dhdz = interp.gradient(xg, zg)
    starts = np.column_stack([np.linspace(4, 46, 8), np.full(8, D - 0.05)])
    ax.streamplot(xg, zg, -np.asarray(dhdx), -np.asarray(dhdz), start_points=starts,
                  color=plotstyle.INK_2, linewidth=0.9, arrowsize=0.9,
                  integration_direction="forward", broken_streamlines=False)
    ax.plot([X_RIVER, X_TOE], [D, D], color=plotstyle.INK, lw=4, solid_capstyle="butt")
    ax.plot([X_TOE, 250], [D + d, D + d], color=plotstyle.C2, lw=1.2)
    ax.plot([X_TOE, X_TOE], [D, D + d], color=plotstyle.C2, lw=1.2)
    ax.text(X_RIVER / 2, D + 0.35, "river bed\nh = h_river", ha="center",
            va="bottom", color=plotstyle.INK_2, fontsize=9)
    ax.text((X_RIVER + X_TOE) / 2, D + 0.35, "dike (impermeable)", ha="center",
            va="bottom", color=plotstyle.INK, fontsize=9)
    ax.text(170, D + d + 0.25, "clay blanket, polder surface h = h_polder",
            ha="center", va="bottom", color=plotstyle.C2, fontsize=9)
    ax.set_xlim(0, 250)
    ax.set_ylim(0, D + d + 2.2)
    ax.set_xlabel("Distance x (m)")
    ax.set_ylabel("Elevation z (m)")
    ax.set_title("Head field and flow paths under the dike, 4 m flood (first 250 m shown)")
    ax.grid(False)
    cb = fig.colorbar(cf, ax=ax, pad=0.01)
    cb.set_label("Hydraulic head h (m)")
    fig.savefig("figures/03_head_field.png")

    # Figure 2: head under the blanket, FEM vs theory
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.plot(xs - X_TOE, h_th, color=plotstyle.C2, label="Leakage-length theory (1D)")
    ax.plot(xs[::16] - X_TOE, h_fem[::16], "o", color=plotstyle.C1, mfc="none",
            mew=1.4, label="FEM, 2D section")
    ax.axhline(H_POLDER, color=plotstyle.INK_2, lw=1, ls=":")
    ax.text(X_END - X_TOE, H_POLDER + 0.12, "polder level", ha="right",
            color=plotstyle.INK_2, fontsize=9)
    ax.set_xlabel("Distance from the landward toe (m)")
    ax.set_ylabel("Head under the blanket (m)")
    ax.set_title(f"Head under the blanket decays over lambda = {lam:.0f} m")
    ax.legend(loc="upper right")
    fig.savefig("figures/03_head_under_blanket.png")

    # Figure 3: uplift factor of safety along the polder
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    ax.plot(xs - X_TOE, fs, color=plotstyle.C1)
    ax.axhline(1.0, color=plotstyle.CRIT, lw=1.2, ls="--")
    ax.text(X_END - X_TOE, 1.03, "FS = 1", color=plotstyle.CRIT, ha="right", va="bottom")
    if unsafe.size:
        ax.axvspan(0, unsafe.max() - X_TOE, color=plotstyle.CRIT, alpha=0.06, lw=0)
        ax.text((unsafe.max() - X_TOE) / 2, 0.15,
                f"uplift expected\nover {unsafe.max() - X_TOE:.0f} m",
                ha="center", color=plotstyle.CRIT, fontsize=9)
    ax.set_ylim(0, None)
    ax.set_xlabel("Distance from the landward toe (m)")
    ax.set_ylabel("Factor of safety (-)")
    ax.set_title("Uplift factor of safety of the blanket along the polder")
    fig.savefig("figures/03_uplift_safety.png")
    print("\nFigures saved in figures/")
