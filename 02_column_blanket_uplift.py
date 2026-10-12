"""
02_column_blanket_uplift.py
===========================
Vertical column through the landward toe of a dike during a flood:
a sand aquifer (4 m) covered by a clay blanket (d = 1 m).

    d/dz ( K(z) dh/dz ) = 0
    bottom (aquifer, z = 0)    : h = h_top + dH   (excess head transmitted from the river)
    top (polder surface, z = H): h = h_top        (water table at ground level)

Datum: z = 0 at the base of the aquifer, so h is an elevation head (m).

The script
  1. solves the column with scikit-fem, K varying with depth,
  2. checks it against the exact two-layer solution (resistances in series),
  3. checks mass conservation (identical Darcy flux in every element),
  4. computes the two classic uplift / heave checks for the blanket,
  5. runs a parametric study on blanket thickness,
  6. runs a mesh-convergence test with the interface placed inside an element.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from skfem import MeshLine, ElementLineP1, Basis, BilinearForm, asm, condense, solve
from skfem.helpers import dot, grad

import plotstyle

os.makedirs("figures", exist_ok=True)

# ---------------------------------------------------------------- input data
D_SAND = 4.0        # aquifer thickness (m)
D_CLAY = 1.0        # blanket thickness (m)
K_SAND = 1e-4       # m/s
K_CLAY = 1e-7       # m/s
DH = 3.0            # excess head in the aquifer during the flood (m)
GAMMA_W = 9.81      # unit weight of water (kN/m3)
GAMMA_SAT = 17.5    # saturated unit weight of the clay blanket (kN/m3)


def solve_column(d_sand, d_clay, k_sand, k_clay, dh, n_elem=50, z_interface=None):
    """Finite element solution of the layered column. Returns (z, h, h_top, z_int)."""
    H = d_sand + d_clay
    z_int = d_sand if z_interface is None else z_interface
    h_top = H                          # polder water table at the ground surface
    h_bot = h_top + dh

    mesh = MeshLine(np.linspace(0.0, H, n_elem + 1))
    basis = Basis(mesh, ElementLineP1())

    @BilinearForm
    def darcy(h, v, w):
        # K is evaluated at the quadrature points w.x[0]: sand below, clay above
        K = np.where(w.x[0] < z_int, k_sand, k_clay)
        return K * dot(grad(h), grad(v))

    A = asm(darcy, basis)
    z = mesh.p[0]
    bottom = np.where(np.isclose(z, 0.0))[0]
    top = np.where(np.isclose(z, H))[0]
    h = np.zeros(basis.N)
    h[bottom], h[top] = h_bot, h_top
    h = solve(*condense(A, np.zeros(basis.N), x=h, D=np.concatenate([bottom, top])))

    order = np.argsort(z)
    return z[order], h[order], h_top, z_int


def exact_column(z, d_sand, d_clay, k_sand, k_clay, dh, z_int=None):
    """Exact solution: two resistances in series carry the same Darcy flux q."""
    H = d_sand + d_clay
    z_int = d_sand if z_int is None else z_int
    L1, L2 = z_int, H - z_int
    h_top, h_bot = H, H + dh
    q = (h_bot - h_top) / (L1 / k_sand + L2 / k_clay)
    h_int = h_bot - q * L1 / k_sand
    h = np.where(z <= z_int, h_bot - q * z / k_sand, h_int - q * (z - z_int) / k_clay)
    return h, q, h_int


def blanket_checks(h_int, z_int, h_top, d_clay):
    """Uplift (vertical equilibrium) and heave (gradient) checks of the blanket."""
    u = GAMMA_W * (h_int - z_int)              # pore pressure under the blanket (kPa)
    sigma_v = GAMMA_SAT * d_clay               # total vertical stress of the blanket (kPa)
    fs_uplift = sigma_v / u
    i_blanket = (h_int - h_top) / d_clay       # vertical gradient across the blanket
    i_crit = (GAMMA_SAT - GAMMA_W) / GAMMA_W   # critical gradient
    fs_heave = i_crit / i_blanket
    return dict(u=u, sigma_v=sigma_v, fs_uplift=fs_uplift,
                i=i_blanket, i_crit=i_crit, fs_heave=fs_heave)


if __name__ == "__main__":
    plotstyle.apply()

    # ---------------------------------------------------- 1-3. reference case
    z, h, h_top, z_int = solve_column(D_SAND, D_CLAY, K_SAND, K_CLAY, DH, n_elem=50)
    h_ex, q_ex, h_int_ex = exact_column(z, D_SAND, D_CLAY, K_SAND, K_CLAY, DH)
    i_node = np.argmin(np.abs(z - z_int))
    h_int = h[i_node]

    print("=== Reference case: 4 m sand + 1 m clay, dH = 3 m ===")
    print(f"Max error vs exact solution      : {np.max(np.abs(h - h_ex)):.2e} m")
    print(f"Head at the sand-clay interface  : {h_int:.4f} m  (exact {h_int_ex:.4f} m)")
    print(f"Share of the head loss in clay   : {(h_int - h_top) / DH * 100:.1f} %")

    zm = 0.5 * (z[1:] + z[:-1])
    K_elem = np.where(zm < z_int, K_SAND, K_CLAY)
    q_elem = -K_elem * np.diff(h) / np.diff(z)
    print(f"Darcy flux, exact                : {q_ex:.6e} m/s")
    print(f"Darcy flux per element, min/max  : {q_elem.min():.6e} / {q_elem.max():.6e} m/s")
    print(f"Relative spread of the flux      : {(q_elem.max() - q_elem.min()) / q_ex:.1e}")

    # ------------------------------------------------------ 4. blanket checks
    c = blanket_checks(h_int, z_int, h_top, D_CLAY)
    print("\n=== Blanket checks ===")
    print(f"Pore pressure under the blanket  : {c['u']:.1f} kPa")
    print(f"Weight of the blanket            : {c['sigma_v']:.1f} kPa")
    print(f"FS uplift = sigma_v / u          : {c['fs_uplift']:.2f}")
    print(f"Vertical gradient in the blanket : {c['i']:.2f}  (critical {c['i_crit']:.2f})")
    print(f"FS heave = i_crit / i            : {c['fs_heave']:.2f}")
    print("-> FS < 1: the blanket is expected to lift or crack; this is the first stage"
          " of backward erosion piping (uplift -> heave -> pipe).")

    fig, ax = plt.subplots(figsize=(5.4, 5.2))
    ax.axhspan(0, z_int, color=plotstyle.C1, alpha=0.07, lw=0)
    ax.axhspan(z_int, D_SAND + D_CLAY, color=plotstyle.C2, alpha=0.10, lw=0)
    zz = np.linspace(0, D_SAND + D_CLAY, 400)
    ax.plot(exact_column(zz, D_SAND, D_CLAY, K_SAND, K_CLAY, DH)[0], zz,
            color=plotstyle.C2, label="Analytical solution")
    ax.plot(h[::2], z[::2], "o", color=plotstyle.C1, mfc="none", mew=1.5,
            label="FEM (scikit-fem)")
    ax.text(h_top + 0.08, 4.5, "clay blanket\nK = 1e-7 m/s", color=plotstyle.INK_2, va="center")
    ax.text(h_top + 0.08, 2.0, "sand aquifer\nK = 1e-4 m/s", color=plotstyle.INK_2, va="center")
    ax.set_xlabel("Hydraulic head h (m)")
    ax.set_ylabel("Elevation z (m)")
    ax.set_title("Almost all the head is lost in the blanket")
    ax.legend(loc="lower left")
    fig.savefig("figures/02_column_blanket_profile.png")

    # ------------------------------------------- 5. blanket thickness study
    d_values = np.linspace(0.5, 6.0, 23)
    fs_up, fs_hv = [], []
    for d in d_values:
        n = int(round((D_SAND + d) / 0.05))
        z_d, h_d, ht_d, zi_d = solve_column(D_SAND, d, K_SAND, K_CLAY, DH, n_elem=n)
        hi_d = np.interp(zi_d, z_d, h_d)
        cd = blanket_checks(hi_d, zi_d, ht_d, d)
        fs_up.append(cd["fs_uplift"])
        fs_hv.append(cd["fs_heave"])
    fs_up, fs_hv = np.array(fs_up), np.array(fs_hv)
    d_crit = np.interp(1.0, fs_up, d_values)
    print(f"\nBlanket thickness needed for FS uplift = 1 (dH = {DH} m): {d_crit:.2f} m")

    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    ax.plot(d_values, fs_up, color=plotstyle.C1, label="FS uplift (vertical equilibrium)")
    ax.plot(d_values, fs_hv, color=plotstyle.C2, label="FS heave (critical gradient)")
    ax.axhline(1.0, color=plotstyle.CRIT, lw=1.2, ls="--")
    ax.text(d_values[-1], 1.04, "FS = 1", color=plotstyle.CRIT, ha="right", va="bottom")
    ax.axvline(d_crit, color=plotstyle.INK_2, lw=1, ls=":")
    ax.text(d_crit + 0.08, 0.15, f"{d_crit:.1f} m", color=plotstyle.INK_2)
    ax.set_xlabel("Blanket thickness d (m)")
    ax.set_ylabel("Factor of safety (-)")
    ax.set_ylim(0, None)
    ax.set_title(f"Blanket stability for an excess head of {DH:.0f} m")
    ax.legend(loc="upper left")
    fig.savefig("figures/02_blanket_thickness_study.png")

    # ------------------------------ 6. mesh convergence, interface inside an element
    # Interface moved to z = 4.25 m. With n = 10 * 3^k elements (size 0.5 / 3^k m)
    # it always falls exactly in the middle of an element, so the meshes are comparable.
    z_off = 4.25
    n_list = [10, 30, 90, 270, 810]
    sizes, errors = [], []
    for n in n_list:
        z_n, h_n, _, _ = solve_column(D_SAND, D_CLAY, K_SAND, K_CLAY, DH,
                                      n_elem=n, z_interface=z_off)
        h_e = exact_column(z_n, D_SAND, D_CLAY, K_SAND, K_CLAY, DH, z_int=z_off)[0]
        sizes.append((D_SAND + D_CLAY) / n)
        errors.append(np.max(np.abs(h_n - h_e)))
    sizes, errors = np.array(sizes), np.array(errors)
    slope = np.polyfit(np.log(sizes), np.log(errors), 1)[0]
    print("\n=== Mesh convergence, interface at z = 4.25 m (middle of an element) ===")
    for s_, e in zip(sizes, errors):
        print(f"  element size {s_:7.4f} m  ->  max error {e:.2e} m")
    print(f"Observed convergence order: {slope:.2f}")
    z_a, h_a, _, _ = solve_column(D_SAND, D_CLAY, K_SAND, K_CLAY, DH, n_elem=20, z_interface=z_off)
    h_ae = exact_column(z_a, D_SAND, D_CLAY, K_SAND, K_CLAY, DH, z_int=z_off)[0]
    print(f"Same interface on a node (20 elements of 0.25 m): max error "
          f"{np.max(np.abs(h_a - h_ae)):.2e} m  -> the mesh should follow the layers.")

    fig, ax = plt.subplots(figsize=(5.6, 4.2))
    ax.loglog(sizes, errors, "o-", color=plotstyle.C1, label="FEM, interface mid-element")
    ref = errors[0] * (sizes / sizes[0]) ** 1
    ax.loglog(sizes, ref, ls="--", lw=1.2, color=plotstyle.INK_2, label="Slope 1 (reference)")
    ax.set_xlabel("Element size (m)")
    ax.set_ylabel("Max head error (m)")
    ax.set_title(f"Mesh convergence, observed order {slope:.2f}")
    ax.legend(loc="lower right")
    fig.savefig("figures/02_mesh_convergence.png")
    print("\nFigures saved in figures/")
