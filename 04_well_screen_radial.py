"""
04_well_screen_radial.py
========================
Steady flow towards the screen of a production well (axisymmetric), as a first
step towards sand production in geothermal wells.

Geometry (r radial distance from the well axis, z vertical, confined reservoir):

    z = D  ====================== impermeable top (no flow) ======================
           |screen|
           | h=hw |  -> -> -> -> radial flow towards the screen
           |      |                                               h = 0 at r = R
    z_tip  +------+   <- screen tip (partial penetration only)
           | wall |   (no flow below the screen)
    z = 0  ===================== impermeable base (no flow) ======================
          r = r_w                                                       r = R

Weak form in cylindrical coordinates (axisymmetric, no source term):

    integral of  K grad(h) . grad(v)  r dr dz  = 0      for all v

It is the weak form of the 2D section with one extra factor r: a ring of soil at
radius r has a volume proportional to r. The factor 2*pi is taken out.

Two cases, both producing the same rate Q (150 m3/h, a typical Dutch geothermal
doublet):
  A. Screen over the full reservoir thickness. Verified against the Thiem
     solution, on a geometric and on a uniform radial mesh.
  B. Screen over the upper half only. Flow converges on the screen tip.

Erosion criterion: the coupled hydro-mechanical erosion model of Scheperboer,
Suiker, Bosco and Clemens (Acta Geotechnica, 2022) erodes the soil where the
Darcy flux exceeds a critical value. Here an illustrative threshold q_c is used
to map where that would happen; q_c is not a measured value.
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
R_W = 0.1            # well (screen) radius (m)
R_OUT = 500.0        # outer radius, head kept at its initial value (m)
D = 50.0             # reservoir thickness (m)
K = 5e-5             # hydraulic conductivity of the reservoir sand (m/s)
Q_PROD = 150 / 3600  # production rate (m3/s), 150 m3/h
SCREEN_PARTIAL = 25.0            # screen length in case B (m), upper half
Z_TIP = D - SCREEN_PARTIAL       # elevation of the screen tip in case B (m)
Q_C = 1e-3           # illustrative critical Darcy flux for erosion (m/s)
BAND = 0.5           # length over which the flux at the screen tip is averaged (m)


@BilinearForm
def darcy_axi(h, v, w):
    r = w.x[0]                       # radial coordinate of the quadrature point
    return K * dot(grad(h), grad(v)) * r


def graded(a, b, x0, h0, growth):
    """Nodes between a and b, spacing h0 at x0, growing by a constant factor away from it."""
    pts, s, x = [x0], h0, x0
    while x - s > a:
        x -= s
        pts.append(x)
        s *= growth
    s, x = h0, x0
    while x + s < b:
        x += s
        pts.append(x)
        s *= growth
    return np.unique(np.array(pts + [a, b]))


def build_mesh(full_screen, refine=1, n_r=120, radial="geometric", n_z=None):
    """Structured mesh. Radial spacing geometric (constant ratio) or uniform.

    Full screen: the solution does not depend on z, a few layers are enough.
    Partial screen: vertical spacing of 1 cm at the screen tip, growing away from it.
    """
    n = int(n_r * refine)
    if radial == "geometric":
        r = np.geomspace(R_W, R_OUT, n + 1)
    else:
        r = np.linspace(R_W, R_OUT, n + 1)
    if full_screen:
        z = np.linspace(0.0, D, (n_z or int(25 * refine)) + 1)
    else:
        z = graded(0.0, D, Z_TIP, 0.01 / refine, 1.15 ** (1 / refine))
    return MeshTri.init_tensor(r, z)


def solve_well(mesh, full_screen, q_target=Q_PROD):
    """Solve with a unit drawdown, then scale to the target rate (the problem is linear)."""
    basis = Basis(mesh, ElementTriP1())
    A = asm(darcy_axi, basis)
    r, z = mesh.p
    z_bottom = 0.0 if full_screen else Z_TIP
    screen = np.nonzero(np.isclose(r, R_W) & (z >= z_bottom - 1e-9))[0]
    outer = np.nonzero(np.isclose(r, R_OUT))[0]

    h = np.zeros(basis.N)
    h[screen] = -1.0                 # unit drawdown at the screen, h = 0 far away
    h = solve(*condense(A, np.zeros(basis.N), x=h, D=np.concatenate([screen, outer])))

    # (A h)_i is the flow entering the domain at node i, per radian (factor 2*pi out)
    react = A @ h
    q_screen = -2 * np.pi * react[screen].sum()      # leaves through the screen
    q_outer = 2 * np.pi * react[outer].sum()         # enters at r = R
    scale = q_target / q_screen
    return dict(mesh=mesh, basis=basis, h=h * scale, react=react * scale,
                screen=screen, z_bottom=z_bottom, drawdown=scale,
                imbalance=abs(q_screen - q_outer) / q_screen)


def sandface_flux(sol):
    """Darcy flux entering the screen, node by node, from the nodal reactions.

    Each screen node collects the inflow over half of the screen segments
    around it, so flux = inflow / (r_w * tributary length).
    """
    z = sol["mesh"].p[1]
    order = sol["screen"][np.argsort(z[sol["screen"]])]
    zs = z[order]
    lo = np.r_[zs[0], 0.5 * (zs[:-1] + zs[1:])]
    hi = np.r_[0.5 * (zs[:-1] + zs[1:]), zs[-1]]
    inflow = -sol["react"][order]                    # per radian
    return zs, inflow / (R_W * (hi - lo)), inflow, lo, hi


def flux_at_tip(sol, band=BAND):
    """Inflow over the lowest `band` metres of the screen, divided by that area.

    The point flux at the screen tip is singular (it grows without limit when the
    mesh is refined), but the inflow over a finite length converges.
    """
    zs, _, inflow, lo, hi = sandface_flux(sol)
    z0 = sol["z_bottom"]
    overlap = np.clip(np.minimum(hi, z0 + band) - np.maximum(lo, z0), 0.0, None)
    return np.sum(inflow * overlap / (hi - lo)) / (R_W * band)


def erodible_zone(sol, q_c=Q_C):
    """Elements where the Darcy flux exceeds q_c: volume (m3) and radial extent (m)."""
    mesh, basis = sol["mesh"], sol["basis"]
    g = basis.interpolate(sol["h"]).grad                     # constant per element (P1)
    q = K * np.sqrt(g[0] ** 2 + g[1] ** 2).mean(axis=1)
    pts = mesh.p[:, mesh.t]
    rc = pts[0].mean(axis=0)
    area = 0.5 * np.abs((pts[0, 1] - pts[0, 0]) * (pts[1, 2] - pts[1, 0])
                        - (pts[0, 2] - pts[0, 0]) * (pts[1, 1] - pts[1, 0]))
    hot = q > q_c
    volume = np.sum(2 * np.pi * rc[hot] * area[hot])
    extent = pts[0][:, hot].max() if hot.any() else R_W
    return volume, extent, q


def thiem(r, drawdown):
    """Exact head for full penetration: logarithmic in r."""
    return -drawdown + drawdown * np.log(r / R_W) / np.log(R_OUT / R_W)


if __name__ == "__main__":
    plotstyle.apply()

    # ================================================ A. full screen vs Thiem
    full = solve_well(build_mesh(True), True)
    r_nodes = full["mesh"].p[0]
    s_exact = Q_PROD * np.log(R_OUT / R_W) / (2 * np.pi * K * D)
    # heads compared with Thiem for the same drawdown at the screen
    err_h = np.max(np.abs(full["h"] - thiem(r_nodes, full["drawdown"])))
    print("=== A. Full screen, Q = 150 m3/h ===")
    print(f"Mesh                      : {full['mesh'].p.shape[1]} nodes, 120 radial elements")
    print(f"Max head error at nodes   : {err_h:.1e} m (same drawdown as Thiem)")
    print(f"Drawdown for Q = 150 m3/h : FEM {full['drawdown']:.3f} m, Thiem {s_exact:.3f} m "
          f"({abs(full['drawdown'] / s_exact - 1):.1e} relative)")
    print(f"Mass balance              : {full['imbalance']:.1e} (relative)")
    q_mean_full = Q_PROD / (2 * np.pi * R_W * D)
    print(f"Flux at the sandface      : {q_mean_full * 1000:.2f} mm/s (Q / screen area)")

    # Convergence of the discharge for a given drawdown, geometric vs uniform spacing
    n_list = [20, 40, 80, 160, 320, 640, 1280]
    conv = {"geometric": [], "uniform": []}
    for kind in conv:
        for n in n_list:
            m = build_mesh(True, n_r=n, radial=kind, n_z=2)
            sol = solve_well(m, True, q_target=1.0)
            # drawdown needed for Q = 1, compared with Thiem
            conv[kind].append(abs(sol["drawdown"] / (np.log(R_OUT / R_W) / (2 * np.pi * K * D)) - 1))
    print("\nDrawdown error for a given rate, by number of radial elements")
    print("   n     geometric    uniform")
    for i, n in enumerate(n_list):
        print(f"{n:5d}    {conv['geometric'][i]:.1e}     {conv['uniform'][i]:.1e}")
    order = np.log(conv["geometric"][-2] / conv["geometric"][-1]) / np.log(2)
    print(f"Observed order, geometric spacing: {order:.2f}")

    vol_full, ext_full, _ = erodible_zone(full)
    r_c = Q_PROD / (2 * np.pi * D * Q_C)
    vol_exact = np.pi * (r_c ** 2 - R_W ** 2) * D
    print(f"\nFlux above q_c = {Q_C * 1000:.0f} mm/s out to r = {ext_full:.3f} m "
          f"(exact {r_c:.3f} m), volume {vol_full:.2f} m3 (exact {vol_exact:.2f} m3)")

    # ======================================= B. partial screen (upper half)
    part = solve_well(build_mesh(False), False)
    print("\n=== B. Screen over the upper half, same Q ===")
    print(f"Mesh                      : {part['mesh'].p.shape[1]} nodes")
    print(f"Drawdown at the screen    : {part['drawdown']:.2f} m "
          f"(full screen: {full['drawdown']:.2f} m)")
    print(f"Mass balance              : {part['imbalance']:.1e} (relative)")
    q_mean_part = Q_PROD / (2 * np.pi * R_W * SCREEN_PARTIAL)
    q_tip = flux_at_tip(part)
    print(f"Mean flux at the sandface : {q_mean_part * 1000:.2f} mm/s")
    print(f"Flux over the lowest {BAND} m : {q_tip * 1000:.2f} mm/s "
          f"({q_tip / q_mean_full:.1f} x the full-screen value)")
    vol_part, ext_part, q_elem = erodible_zone(part)
    print(f"Flux above q_c out to r = {ext_part:.3f} m, "
          f"volume {vol_part:.2f} m3 (full screen: {vol_full:.2f} m3)")

    # Production rate at which the flux at the screen reaches q_c
    q_max_full = Q_PROD * Q_C / q_mean_full * 3600
    q_max_part = Q_PROD * Q_C / q_tip * 3600
    print(f"\nRate at which the sandface flux reaches q_c: full screen {q_max_full:.0f} m3/h, "
          f"half screen {q_max_part:.0f} m3/h")

    # Mesh refinement: what converges at the tip, and what does not
    print("\n=== Mesh refinement, case B ===")
    tip_node = {}
    for ref in (1, 2):
        s_r = part if ref == 1 else solve_well(build_mesh(False, refine=2), False)
        zs, qn, *_ = sandface_flux(s_r)
        v_r, e_r, _ = erodible_zone(s_r)
        tip_node[ref] = qn[0]
        print(f"  refine x{ref}: {s_r['mesh'].p.shape[1]:6d} nodes, drawdown {s_r['drawdown']:.3f} m, "
              f"flux over {BAND} m {flux_at_tip(s_r) * 1000:.3f} mm/s, "
              f"erodible volume {v_r:.2f} m3, point flux at the tip {qn[0] * 1000:.1f} mm/s")
    print("The point flux at the tip keeps growing (singularity); the averaged flux, "
          "the drawdown and the erodible volume converge.")

    # ============================================================== figures
    # Figure 1: verification, head profile and convergence
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    rr = np.geomspace(R_W, R_OUT, 400)
    ax1.plot(rr, -thiem(rr, s_exact), color=plotstyle.C2, label="Thiem (exact)")
    for kind, color, mk in (("geometric", plotstyle.C1, "o"), ("uniform", plotstyle.C3, "s")):
        m = build_mesh(True, n_r=20, radial=kind, n_z=2)
        sol = solve_well(m, True)
        rn, hn = m.p[0], sol["h"]
        sel = np.isclose(m.p[1], 0.0)
        ax1.plot(rn[sel], -hn[sel], mk, color=color, mfc="none", mew=1.4, ms=5,
                 label=f"FEM, 20 elements, {kind} spacing")
    ax1.set_xscale("log")
    ax1.invert_yaxis()
    ax1.set_xlabel("Distance from the well axis r (m)")
    ax1.set_ylabel("Drawdown (m)")
    ax1.set_title("Head around the well, Q = 150 m3/h")
    ax1.legend(loc="lower right")

    ax2.loglog(n_list, conv["geometric"], "o-", color=plotstyle.C1, label="geometric spacing")
    ax2.loglog(n_list, conv["uniform"], "s-", color=plotstyle.C3, label="uniform spacing")
    ref_line = conv["geometric"][2] * (np.array(n_list) / n_list[2]) ** -2.0
    ax2.loglog(n_list, ref_line, ":", color=plotstyle.INK_2, lw=1.2, label="slope -2")
    ax2.set_xlabel("Radial elements")
    ax2.set_ylabel("Relative error on the drawdown")
    ax2.set_title("The mesh must follow the flow")
    ax2.legend(loc="lower left")
    fig.savefig("figures/04_thiem_verification.png")

    # Figure 2: case B, drawdown field, flow paths and erodible zone (log r axis)
    mesh = part["mesh"]
    r, z = mesh.p
    lr = np.log10(r)
    tri = mtri.Triangulation(lr, z, mesh.t.T)
    fig, ax = plt.subplots(figsize=(10, 4.2))
    s = -part["h"]
    levels = np.linspace(0, s.max(), 15)
    cf = ax.tricontourf(tri, s, levels=levels, cmap="Blues")
    ax.tricontour(tri, s, levels=levels[::2], colors="white", linewidths=0.6)
    # flow paths in (log10 r, z): d(log10 r)/dt = v_r / (r ln 10)
    interp = mtri.LinearTriInterpolator(tri, part["h"])
    xg, zg = np.meshgrid(np.linspace(lr.min(), lr.max(), 400), np.linspace(0.05, D - 0.05, 120))
    g_lr, g_z = interp.gradient(xg, zg)
    rg = 10 ** xg
    v_r = -K * np.asarray(g_lr) / (rg * np.log(10))       # dh/dr = dh/dlog10r / (r ln10)
    v_z = -K * np.asarray(g_z)
    u = v_r / (rg * np.log(10))
    starts = np.column_stack([np.full(9, lr.max() - 0.02), np.linspace(2, D - 2, 9)])
    ax.streamplot(xg, zg, u, v_z, start_points=starts, color=plotstyle.INK_2,
                  linewidth=0.9, arrowsize=0.9, integration_direction="forward",
                  broken_streamlines=False)
    # erodible zone: element flux above q_c
    hot = q_elem > Q_C
    for e in np.nonzero(hot)[0]:
        pts = mesh.t[:, e]
        ax.fill(lr[pts], z[pts], color=plotstyle.CRIT, lw=0, alpha=0.85)
    ax.plot([np.log10(R_W)] * 2, [Z_TIP, D], color=plotstyle.INK, lw=5, solid_capstyle="butt")
    ax.text(np.log10(R_W) + 0.03, D - 1.5, "screen", color="white", fontsize=9,
            va="top", fontweight="bold")
    ax.text(np.log10(R_W) + 0.03, Z_TIP - 1.0, "screen tip", color=plotstyle.INK,
            fontsize=9, va="top")
    ax.annotate(f"Darcy flux above q_c\n(out to r = {ext_part:.2f} m at the tip)",
                xy=(np.log10(0.2), Z_TIP + 4), xytext=(np.log10(0.13), 6),
                color=plotstyle.CRIT, fontsize=9,
                arrowprops=dict(arrowstyle="-", color=plotstyle.CRIT, lw=0.8))
    ticks = [0.1, 1, 10, 100, 500]
    ax.set_xticks(np.log10(ticks))
    ax.set_xticklabels([f"{t:g}" for t in ticks])
    ax.set_xlim(lr.min(), lr.max())
    ax.set_ylim(0, D)
    ax.set_xlabel("Distance from the well axis r (m, log scale)")
    ax.set_ylabel("Elevation z (m)")
    ax.set_title("Screen over the upper half: flow converges on the screen tip")
    ax.grid(False)
    cb = fig.colorbar(cf, ax=ax, pad=0.01)
    cb.set_label("Drawdown (m)")
    fig.savefig("figures/04_partial_screen_field.png")

    # Figure 3: flux along the screen, full vs partial, same production rate
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    zs_f, qn_f, *_ = sandface_flux(full)
    zs_p, qn_p, *_ = sandface_flux(part)
    # end nodes left out: the tip node is singular, the others carry a small
    # end effect of the nodal reactions (about 1 %)
    ax.plot(qn_f[1:-1] * 1000, zs_f[1:-1], color=plotstyle.C2, label="full screen")
    ax.plot(qn_p[1:-1] * 1000, zs_p[1:-1], color=plotstyle.C1, label="screen over the upper half")
    ax.plot(q_tip * 1000, Z_TIP + BAND / 2, "o", color=plotstyle.C1, mfc="none", mew=1.6)
    ax.annotate(f"mean over the lowest {BAND} m\n{q_tip * 1000:.1f} mm/s",
                xy=(q_tip * 1000, Z_TIP + BAND / 2), xytext=(q_tip * 1000 * 0.9, Z_TIP - 10),
                fontsize=9, color=plotstyle.C1,
                arrowprops=dict(arrowstyle="-", color=plotstyle.C1, lw=0.8))
    ax.axvline(Q_C * 1000, color=plotstyle.CRIT, lw=1.2, ls="--")
    ax.text(Q_C * 1000 * 1.04, 2, "q_c (illustrative)", color=plotstyle.CRIT, fontsize=9)
    ax.set_xscale("log")
    ax.set_xticks([1, 2, 5, 10])
    ax.set_xticklabels(["1", "2", "5", "10"])
    ax.set_ylim(0, D)
    ax.set_xlabel("Darcy flux entering the screen (mm/s, log scale)")
    ax.set_ylabel("Elevation z (m)")
    ax.set_title("Flux along the screen, Q = 150 m3/h")
    ax.legend(loc="upper right")
    fig.savefig("figures/04_sandface_flux.png")
    print("\nFigures saved in figures/")
