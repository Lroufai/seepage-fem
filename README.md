# seepage-fem

**Finite element modelling of groundwater flow under river dikes and around well screens, from a verified 1D column to 2D sections, as a first step towards backward erosion piping and sand production.**

Backward erosion piping starts where seepage under a dike converges at the landward toe: the excess head in the sand aquifer first lifts or cracks the clay blanket (uplift), the exit gradient then mobilises sand (heave), and a pipe grows backwards under the dike. Every one of these stages depends on the head field under the dike, and on how laboratory-scale soil properties carry over to the field.

Sand production in geothermal wells follows the same logic around a well screen: flow converges on the screen, and where the Darcy flux exceeds a critical value the sand is carried into the well.

This repository builds these flow fields step by step with the finite element method, in Python with [scikit-fem](https://github.com/kinnala/scikit-fem), and checks each step against an independent reference before moving on.

| Script | What it does | Verified against |
|---|---|---|
| `01_column_homogeneous.py` | 1D steady flow, homogeneous column. Solved twice: a hand-written assembly of the 2 × 2 element matrices, and scikit-fem | Exact solution; the two implementations against each other |
| `02_column_blanket_uplift.py` | Sand aquifer under a clay blanket at the landward toe. Uplift and heave factors of safety, blanket-thickness study, mesh convergence | Exact two-layer solution; flux continuity in every element |
| `03_dike_section_2d.py` | 2D vertical section: river bed, impermeable dike footprint, blanket over the polder. Head field, flow paths, head under the blanket, uplift along the polder | Leakage-length (blanket) theory; global mass balance; mesh refinement |
| `04_well_screen_radial.py` | Axisymmetric flow to a production well (150 m³/h). Full screen and screen over the upper half: flux along the screen, zone where the flux exceeds a critical value | Thiem solution; geometric vs uniform radial mesh; mass balance; mesh refinement |

## Governing equation

Steady, saturated, isotropic Darcy flow with no source term:

$$\nabla \cdot \left( K \, \nabla h \right) = 0$$

Weak form solved with piecewise-linear Lagrange elements (P1, Galerkin):

$$\int_\Omega K \, \nabla h \cdot \nabla v \, d\Omega = 0 \qquad \forall v$$

For the well (script 04) the same weak form is written in cylindrical coordinates, with an extra factor r in the integral: a ring of soil at radius r has a volume proportional to r.

Heads are prescribed on the river bed and the polder surface (on the well screen and the outer radius in script 04). Every other boundary (aquifer base, lateral boundaries, dike footprint) is no-flow, which the weak form imposes naturally. Each nodal equation of the assembled system is a discrete mass balance: what enters a node leaves it.

## Results

### 1. Homogeneous column: the method is exact at the nodes

Both implementations reproduce the linear solution to machine precision (max error 2.7 × 10⁻¹⁵ m), and differ from each other by 4 × 10⁻¹⁶ m. The 3-element example prints the global matrix, `[[1,-1,0,0],[-1,2,-1,0],[0,-1,2,-1],[0,0,-1,1]]`, that can be checked by hand.

![Homogeneous column](figures/01_column_homogeneous.png)

### 2. Blanket over an aquifer: almost all the head is lost in the clay

4 m of sand (K = 10⁻⁴ m/s) under 1 m of clay (K = 10⁻⁷ m/s), with an excess head of 3 m in the aquifer.

- Matches the exact two-layer solution to 10⁻¹³ m; the Darcy flux is identical in all 50 elements (relative spread 5 × 10⁻¹¹).
- **99.6 % of the head is lost in the blanket**: the vertical gradient across it is 3.0, against a critical gradient of 0.78.
- Pore pressure under the blanket 39.1 kPa against a blanket weight of 17.5 kPa: **FS uplift = 0.45**, FS heave = 0.26. The blanket would lift: this is the first stage of piping.
- A blanket of **3.8 m** would be needed to reach FS = 1 under a 3 m excess head.

![Head profile](figures/02_column_blanket_profile.png)
![Blanket thickness study](figures/02_blanket_thickness_study.png)

**Mesh must follow the layers.** With the sand-clay interface placed in the middle of an element, the error converges at first order (observed order 1.00). With a node on the interface, the error drops back to machine precision. In 2D and 3D, this is why the mesh is aligned with, and refined at, the blanket-aquifer interface.

![Mesh convergence](figures/02_mesh_convergence.png)

### 3. 2D cross-section under a dike

Aquifer 10 m thick, blanket 1.5 m over the polder, seepage length 40 m from the river bed to the landward toe, flood level 4 m above the polder. 7 287 nodes, 13 940 triangles, solved in about one second.

![Head field](figures/03_head_field.png)

- **Mass balance**: inflow from the river bed and outflow through the blanket agree to 4 × 10⁻¹⁰ (relative), computed from the nodal reactions.
- **Comparison with the leakage-length theory** used in Dutch dike assessment (λ = √(K_a D d / K_b) = 122 m): the head under the blanket at the toe is 14.44 m with the FEM against 14.52 m with the 1D theory, and the two curves differ by at most 1.9 % of the head difference along the whole polder. The theory assumes horizontal flow in the aquifer and vertical flow in the blanket; the 2D model does not, which accounts for the small difference near the dike.
- **Mesh refinement**: doubling the resolution in both directions changes the head at the toe by 3 mm.
- **Uplift**: FS = 0.60 at the toe, below 1 over the first 112 m of the polder. This is where sand boils would be expected.

![Head under the blanket](figures/03_head_under_blanket.png)
![Uplift safety](figures/03_uplift_safety.png)

### 4. Production well: flow converges on the screen tip

Confined reservoir 50 m thick (K = 5 × 10⁻⁵ m/s), screen radius 0.1 m, production 150 m³/h, typical of a Dutch geothermal doublet.

![Thiem verification](figures/04_thiem_verification.png)

- **Full screen, against the Thiem solution**: with geometric radial spacing (constant ratio between successive nodes), the heads are exact at the nodes (error 5 × 10⁻¹³ m) and the drawdown for a given rate converges at second order (observed order 2.00). With uniform spacing, 1 280 elements still leave a 3 % error. **The mesh must follow the flow**: radial flow is logarithmic in r.
- **Screen over the upper half, same rate**: the drawdown rises from 22.6 to 35.2 m, the mean flux at the sandface doubles (2.65 mm/s), and the flux over the lowest 0.5 m of the screen reaches 4.95 mm/s, **3.7 times the full-screen value**. Mass balance closes to 3 × 10⁻¹¹.
- **Where erosion would start**: with an illustrative critical Darcy flux q_c = 1 mm/s, the zone above q_c extends to 0.13 m from the axis for the full screen (exact value 0.133 m) and to 0.36 m at the tip of the half screen. The rate at which the sandface flux reaches q_c drops from 113 m³/h (full screen) to 30 m³/h (half screen).

![Head field around a partially screened well](figures/04_partial_screen_field.png)
![Flux along the screen](figures/04_sandface_flux.png)

- **Mesh refinement**: doubling the resolution changes the drawdown by 0.05 %, the flux over the lowest 0.5 m by 0.3 % and the volume above q_c by 1 %. The point flux at the tip node does not converge (40.6 then 56.2 mm/s): the tip is a singularity, like the landward toe of the dike. Any erosion criterion applied there needs a length over which the flux is averaged, which is where a material length scale (grain size, screen slot) has to enter the model.

## Run it

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python 01_column_homogeneous.py
python 02_column_blanket_uplift.py
python 03_dike_section_2d.py
python 04_well_screen_radial.py
```

Each script prints its checks to the terminal and writes its figures to `figures/`. Total run time is about ten seconds.

## Assumptions and limits

- Steady state, fully saturated, isotropic layers with uniform conductivity. A sandy aquifer responds to a flood in minutes to hours, so steady state is a reasonable and conservative first approximation for a flood lasting days; it is less so under thick or compressible blankets.
- No erosion yet: the model gives the hydraulic loading that drives piping (exit gradient, uplift), not pipe growth.
- The landward toe is a geometric singularity (the no-flow dike footprint meets the blanket). Heads converge there (3 mm change on refinement); point gradients at the corner do not, so gradients are evaluated over the blanket thickness rather than at the corner.
- The critical flux q_c in script 04 is illustrative. Measuring it, and the parameters of the erosion law, is the job of laboratory experiments.
- Parameter values are illustrative, not site-specific.

## Next steps

1. **Pipe progression.** Add a pipe as a high-conductivity line element at the top of the aquifer and grow it with a critical-gradient criterion, to compare against Sellmeijer's rule and laboratory-scale experiments.
2. **Heterogeneity at the interface.** Replace the uniform aquifer with layered and stochastic conductivity fields, including coarse-over-fine contrasts derived from fluvial architecture, and quantify how the exit gradient and the critical head change.
3. **From laboratory to field.** Use the same code at laboratory scale (small-scale piping set-ups) and field scale, to study how parameters calibrated on a sample transfer to a dike, and where pore-scale structure, imaged by X-ray micro-CT, enters that chain.
4. **Coupling.** Hydro-mechanical coupling (effective stress, deformation, porosity and permeability evolving with erosion), in the spirit of the coupled erosion model of [Scheperboer, Suiker, Bosco and Clemens (2022)](https://doi.org/10.1007/s11440-022-01479-8), for the dike blanket and for the sand around a well screen, with the erosion parameters calibrated on laboratory tests.

## Author

Intchi Amadou Roufai Abdoul Latif · MSc Biogeosciences, University of Neuchâtel
