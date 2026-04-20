"""
crm_geometry.py
===============
Stand-alone pyGeo parametrisation of the NASA Common Research Model (CRM) wing.

Reference geometry
------------------
Vassberg et al. (2008), "Development of a Common Research Model for Applied
CFD Validation Studies", AIAA 2008-6919.

Full-scale metric dimensions used throughout:
  Semi-span   b/2  = 29.38 m
  Ref area    Sref = 383.69 m²  (half-span: 191.845 m²)
  MAC              = 7.005 m
  Aspect ratio AR  = 9.0
  Taper ratio  λ   = 0.275  (C_tip / C_root)
  LE sweep         ≈ 25° inboard (root→Yehudi) / 35° outboard (Yehudi→tip)
  Dihedral         = 5°
  Twist            ≈ +2° (root) to −6° (tip) washout

Airfoil sections
----------------
True CRM65 sections require the official NASA geometry files.  This script
generates NACA 4-series proxy sections at three span stations; replace the
written .dat files with official CRM65 data for production analysis.

Design variables  (mirror of MACH-Aero opt_ffd tutorial)
---------------------------------------------------------
Global:
  twist    – per-section rotation about the spanwise axis (Z)  [nRefAxPts-1]
  dihedral – per-section Y-translation of the reference axis   [nRefAxPts-1]
  taper    – root and tip chord scale factor                    [2]
Local:
  shape    – Y-displacement of FFD control points              [N_local]

Usage
-----
  # Geometry + FFD only (no pyGeo required):
  python crm_geometry.py

  # Full DVGeometry setup + deformation (requires pyGeo):
  python crm_geometry.py

  # Also save a planform PNG:
  python crm_geometry.py --plot
"""

from __future__ import annotations

import argparse
import pathlib
import warnings

import numpy as np

# ---------------------------------------------------------------------------
# Optional pyGeo import
# ---------------------------------------------------------------------------
try:
    from pygeo import DVGeometry, pyGeo as _pyGeo  # type: ignore
    HAS_PYGEO = True
except ImportError:
    HAS_PYGEO = False
    warnings.warn(
        "pyGeo not found – .dat section files and FFD box will be written, "
        "but wing surface and DVGeometry are disabled.",
        ImportWarning,
        stacklevel=1,
    )

# ---------------------------------------------------------------------------
# 1.  CRM reference planform  (full-scale, metres)
# ---------------------------------------------------------------------------

SEMI_SPAN            = 29.38    # m   b/2
SREF                 = 383.69   # m²  full reference area
MAC                  = 7.005    # m
AR                   = 9.0
TAPER                = 0.275    # C_tip / C_root

DIHEDRAL_DEG          = 5.0
LE_SWEEP_INBOARD_DEG  = 25.0   # root → Yehudi break  (η = 0.00 – 0.37)
LE_SWEEP_OUTBOARD_DEG = 35.0   # Yehudi → tip         (η = 0.37 – 1.00)

# Normalised span stations
ETA_SOB    = 0.10   # side-of-body (exposed-root / fuselage junction)
ETA_YEHUDI = 0.37   # Yehudi break (inner / outer panel junction)
ETA_TIP    = 1.00

Z_SOB    = ETA_SOB    * SEMI_SPAN   # 2.938 m
Z_YEHUDI = ETA_YEHUDI * SEMI_SPAN   # 10.871 m
Z_TIP    = ETA_TIP    * SEMI_SPAN   # 29.38  m

# Trig helpers
_dih = np.tan(np.radians(DIHEDRAL_DEG))
_swi = np.tan(np.radians(LE_SWEEP_INBOARD_DEG))
_swo = np.tan(np.radians(LE_SWEEP_OUTBOARD_DEG))

# ---- Chord lengths -------------------------------------------------------
# C_root set so that the two-panel trapezoidal half-wing matches SREF/2.
# Panel 1 (root → Yehudi):  area = (C_root + C_yeh)/2 * Z_yeh
# Panel 2 (Yehudi → tip):   area = (C_yeh  + C_tip)/2 * (Z_tip - Z_yeh)
# Solving for C_yeh given C_root = 10.24 m (chosen to satisfy SREF):
CHORD_ROOT   = 10.24                  # m  (at η = 0, fuselage centreline)
CHORD_TIP    = TAPER * CHORD_ROOT     # m  (= 2.816 m)

# Yehudi chord from area constraint:
#   SREF/2 = C_root*Z_yeh/2 + C_yeh*Z_tip/2 + C_tip*(Z_tip-Z_yeh)/2
CHORD_YEHUDI = (
    2.0 * (SREF / 2.0 - CHORD_ROOT * Z_YEHUDI / 2.0
           - CHORD_TIP * (Z_TIP - Z_YEHUDI) / 2.0)
    / Z_TIP
)  # ≈ 7.495 m

# SOB chord by linear interpolation on the inboard panel
CHORD_SOB = CHORD_ROOT + (CHORD_YEHUDI - CHORD_ROOT) * (Z_SOB / Z_YEHUDI)

# ---- Wing-station arrays (root, SOB, Yehudi, tip) ------------------------
# Parallel lists used everywhere; np.interp gives piece-wise-linear planform.

_STA_Z      = [0.0, Z_SOB, Z_YEHUDI, Z_TIP]

_STA_X_LE   = [
    0.0,
    _swi * Z_SOB,
    _swi * Z_YEHUDI,
    _swi * Z_YEHUDI + _swo * (Z_TIP - Z_YEHUDI),
]

_STA_Y_LE   = [_dih * z for z in _STA_Z]   # vertical rise due to dihedral

_STA_CHORD  = [CHORD_ROOT, CHORD_SOB, CHORD_YEHUDI, CHORD_TIP]

# Geometric twist: +ve = nose-up; CRM has ~8° washout from SOB to tip.
_STA_TWIST  = [+2.0, +1.5, -0.5, -6.0]   # degrees

# Approximate half-thickness / chord at each station (for FFD sizing)
_STA_TC_HALF = [0.065, 0.062, 0.055, 0.045]   # ≈ 13%, 12.4%, 11%, 9% t/c


def planform_at_z(z: float) -> tuple[float, float, float]:
    """Return (x_LE, y_LE, chord) at spanwise position z via linear interp."""
    x_le  = float(np.interp(z, _STA_Z, _STA_X_LE))
    y_le  = float(np.interp(z, _STA_Z, _STA_Y_LE))
    chord = float(np.interp(z, _STA_Z, _STA_CHORD))
    return x_le, y_le, chord


# ---------------------------------------------------------------------------
# 2.  Airfoil section generation  (NACA 4-series proxies for CRM65)
# ---------------------------------------------------------------------------

def naca4_coords(m_pct: float, p_tenth: float, t_pct: float,
                 n_pts: int = 129) -> np.ndarray:
    """
    Generate NACA 4-digit airfoil coordinates in Selig format.

    Parameters
    ----------
    m_pct   : max camber in percent of chord (e.g. 2 → 2%)
    p_tenth : chordwise position of max camber in tenths (e.g. 4 → 40%)
    t_pct   : max thickness in percent of chord (e.g. 12 → 12%)
    n_pts   : total number of surface points (odd → symmetric about LE)

    Returns
    -------
    coords : ndarray shape (n_pts, 2) in Selig order
             upper-surface TE → LE, then lower-surface LE → TE
    """
    m = m_pct   / 100.0
    p = p_tenth / 10.0
    t = t_pct   / 100.0

    n_half = (n_pts + 1) // 2
    beta = np.linspace(0.0, np.pi, n_half)
    x    = 0.5 * (1.0 - np.cos(beta))          # cosine spacing

    # Thickness distribution (NACA 4-digit)
    yt = 5.0 * t * (
        0.2969 * np.sqrt(x)
        - 0.1260 * x
        - 0.3516 * x ** 2
        + 0.2843 * x ** 3
        - 0.1015 * x ** 4
    )

    # Camber line and gradient
    if m == 0.0 or p == 0.0:
        yc      = np.zeros_like(x)
        dyc_dx  = np.zeros_like(x)
    else:
        yc = np.where(
            x < p,
            m / p ** 2 * (2 * p * x - x ** 2),
            m / (1 - p) ** 2 * ((1 - 2 * p) + 2 * p * x - x ** 2),
        )
        dyc_dx = np.where(
            x < p,
            2 * m / p ** 2 * (p - x),
            2 * m / (1 - p) ** 2 * (p - x),
        )

    theta = np.arctan(dyc_dx)
    xu = x - yt * np.sin(theta);  yu = yc + yt * np.cos(theta)
    xl = x + yt * np.sin(theta);  yl = yc - yt * np.cos(theta)

    # Selig: upper TE→LE, then lower LE→TE (shared LE point dropped)
    x_full = np.concatenate([xu[::-1], xl[1:]])
    y_full = np.concatenate([yu[::-1], yl[1:]])
    return np.column_stack([x_full, y_full])


def write_dat(path: pathlib.Path, label: str, coords: np.ndarray) -> None:
    """Write airfoil in Selig .dat format."""
    with open(path, "w") as f:
        f.write(f"{label}\n")
        for xc, yc in coords:
            f.write(f" {xc:.8f}  {yc:.8f}\n")


# CRM65-proxy sections: NACA 4-series with representative thickness/camber.
# Substitute official CRM65 .dat files (from commonresearchmodel.larc.nasa.gov)
# for production analyses.
_CRM_SECTIONS = {
    #  filename          m%   p   t%    span location
    "crm_root.dat":    (2.0, 4, 12.0),   # η = 0.00, root
    "crm_yehudi.dat":  (2.0, 4, 11.0),   # η = 0.37, Yehudi
    "crm_tip.dat":     (1.5, 4,  9.0),   # η = 1.00, tip
}


# ---------------------------------------------------------------------------
# 3.  FFD lattice  (Plot3D format, matching MACH-Aero opt_ffd convention)
# ---------------------------------------------------------------------------

def build_ffd(
    outdir: pathlib.Path,
    n_x: int   = 8,
    n_y: int   = 2,
    n_z: int   = 10,
    margin_frac: float = 0.05,
) -> pathlib.Path:
    """
    Build a swept, tapered FFD lattice enclosing the CRM half-span wing and
    write it in Plot3D (.xyz) format.

    Control-point layout: n_x (streamwise) × n_y (normal to planform) × n_z (spanwise).
    Each spanwise slice is an axis-aligned rectangle in (x, y) that tracks the
    local leading-edge position, chord, and dihedral height.

    Parameters
    ----------
    outdir       : directory in which to write crm_ffd.xyz
    n_x          : streamwise control points (default 8)
    n_y          : normal-to-planform control points, must be 2 for upper/lower (default 2)
    n_z          : spanwise control points (default 10)
    margin_frac  : fractional clearance added to local chord on each side (default 0.05)

    Returns
    -------
    Path to the written FFD file.
    """
    if n_y != 2:
        raise ValueError("n_y must be 2 (upper and lower faces of the FFD box).")

    # Spanwise stations for the FFD lattice – denser near root (exponent < 1)
    span_frac = np.linspace(0.0, 1.0, n_z) ** 0.8
    z_sects   = span_frac * Z_TIP

    X = np.zeros((n_y * n_z, n_x))
    Y = np.zeros((n_y * n_z, n_x))
    Z = np.zeros((n_y * n_z, n_x))

    tc_half_root = _STA_TC_HALF[0]
    tc_half_tip  = _STA_TC_HALF[-1]

    row = 0
    for k in range(n_z):
        z_k          = z_sects[k]
        x_le, y_le, chord = planform_at_z(z_k)
        tc_half      = float(np.interp(z_k, [0.0, Z_TIP], [tc_half_root, tc_half_tip]))
        half_thick   = tc_half * chord
        margin       = margin_frac * chord

        x_min = x_le  - margin
        x_max = x_le  + chord + margin
        y_lo  = y_le  - half_thick - margin * 0.5
        y_hi  = y_le  + half_thick + margin * 0.5

        for j in range(n_y):
            y_val = y_lo if j == 0 else y_hi
            X[row, :] = np.linspace(x_min, x_max, n_x)
            Y[row, :] = np.full(n_x, y_val)
            Z[row, :] = np.full(n_x, z_k)
            row += 1

    ffd_path = outdir / "crm_ffd.xyz"
    with open(ffd_path, "w") as f:
        f.write("\t\t1\n")
        f.write(f"\t\t{n_x}\t\t{n_y}\t\t{n_z}\n")
        for arr in (X, Y, Z):
            for row_data in arr:
                f.write("\t" + "\t".join(f"{v:.8f}" for v in row_data) + "\n")

    print(f"  [FFD] Wrote {n_x}×{n_y}×{n_z} lattice → {ffd_path.name}")
    return ffd_path


# ---------------------------------------------------------------------------
# 4.  DVGeometry setup
# ---------------------------------------------------------------------------

def setup_dvgeo(ffd_path: str):
    """
    Instantiate DVGeometry for the CRM wing and register all design variables.

    Global design variables
    -----------------------
    twist    : rotation of each span section about the spanwise (Z) axis.
               Reproduces the twist/incidence animation in the opt_ffd tutorial GIF.
               Shape: [nRefAxPts - 1]   bounds: [-15°, +15°]

    dihedral : Y-translation of reference-axis control points.
               Produces bending / dihedral changes along the span.
               Shape: [nRefAxPts - 1]   bounds: [-5 m, +5 m]

    taper    : chord scale factor at root (index 0) and tip (index 1),
               with linear interpolation in between.
               Shape: [2]               bounds: [0.5, 1.5]

    Local design variables
    ----------------------
    shape    : independent Y-displacement of every FFD control point.
               Controls local airfoil-section shape changes.
               Shape: [n_x * n_y * n_z] bounds: [-0.5, +0.5]

    Returns
    -------
    (DVGeo, nRefAxPts) : tuple
    """
    DVGeo = DVGeometry(ffd_path)

    # Reference axis at 25% chord, aligned with the spanwise (k / z) index
    nRefAxPts = DVGeo.addRefAxis("wing", xFraction=0.25, alignIndex="k")

    # ------------------------------------------------------------------
    # Callback definitions (same pattern as MACH-Aero opt_ffd tutorial)
    # ------------------------------------------------------------------

    def twist_cb(val, geo):
        """Rotate each span section about Z (changes local angle of incidence)."""
        for i in range(1, nRefAxPts):
            geo.rot_z["wing"].coef[i] = val[i - 1]

    def dihedral_cb(val, geo):
        """Translate reference-axis nodes in Y (bends the wing up/down)."""
        C = geo.extractCoef("wing")
        for i in range(1, nRefAxPts):
            C[i, 1] += val[i - 1]
        geo.restoreCoef(C, "wing")

    def taper_cb(val, geo):
        """
        Scale chord linearly from root to tip.
        val[0] = root chord scale factor
        val[1] = tip  chord scale factor
        """
        s     = geo.extractS("wing")
        slope = (val[1] - val[0]) / (s[-1] - s[0])
        for i in range(nRefAxPts):
            geo.scale_x["wing"].coef[i] = slope * (s[i] - s[0]) + val[0]

    n_twist = nRefAxPts - 1   # root twist is fixed; free variables at stations 1…N

    DVGeo.addGlobalDV(
        dvName="twist",
        value=[0.0] * n_twist,
        func=twist_cb,
        lower=-15.0,
        upper=+15.0,
        scale=1.0,
    )
    DVGeo.addGlobalDV(
        dvName="dihedral",
        value=[0.0] * n_twist,
        func=dihedral_cb,
        lower=-5.0,
        upper=+5.0,
        scale=1.0,
    )
    DVGeo.addGlobalDV(
        dvName="taper",
        value=[1.0, 1.0],
        func=taper_cb,
        lower=0.5,
        upper=1.5,
        scale=1.0,
    )

    # Local shape: Y-displacement of each FFD control point
    DVGeo.addLocalDV("shape", lower=-0.5, upper=+0.5, axis="y", scale=1.0)

    return DVGeo, nRefAxPts


# ---------------------------------------------------------------------------
# 5.  Test surface-point set  (stand-in for a CFD mesh surface)
# ---------------------------------------------------------------------------

def sample_wing_surface(n_chord: int = 40, n_span: int = 30) -> np.ndarray:
    """
    Generate a structured grid of surface points on the CRM half-span wing.

    Points lie on a parabolic approximation of the airfoil cross-section.
    In a real workflow, replace these with coordinates extracted from a CFD
    mesh (e.g. via USMesh.getSurfaceCoordinates()).

    Returns
    -------
    pts : ndarray shape (2 * n_chord * n_span, 3)  — (x, y, z) in metres
    """
    z_span = np.linspace(0.0, Z_TIP, n_span)
    pts    = []

    for z in z_span:
        x_le, y_le, chord = planform_at_z(z)
        tc_half = float(np.interp(z, _STA_Z, _STA_TC_HALF))

        xi      = np.linspace(0.0, 1.0, n_chord)   # normalised chordwise
        x_sec   = x_le + xi * chord

        # Parabolic thickness envelope (rough proxy for CRM cross-section)
        half_t  = tc_half * chord * 4.0 * xi * (1.0 - xi)

        for xu, yu in zip(x_sec, y_le + half_t):    # upper surface
            pts.append([xu, yu, z])
        for xl, yl in zip(x_sec, y_le - half_t):    # lower surface
            pts.append([xl, yl, z])

    return np.array(pts, dtype=float)


# ---------------------------------------------------------------------------
# 6.  Optional planform visualisation
# ---------------------------------------------------------------------------

def plot_planform(outdir: pathlib.Path) -> None:
    """Save and display a top-view planform sketch of the CRM wing."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("  [plot] matplotlib not available; skipping.")
        return

    z_dense = np.linspace(0.0, Z_TIP, 200)
    le_x    = np.array([planform_at_z(z)[0] for z in z_dense])
    te_x    = np.array([planform_at_z(z)[0] + planform_at_z(z)[2] for z in z_dense])
    qc_x    = np.array([planform_at_z(z)[0] + 0.25 * planform_at_z(z)[2] for z in z_dense])

    fig, ax = plt.subplots(figsize=(13, 5))
    ax.fill_betweenx(z_dense, le_x, te_x, alpha=0.12, color="steelblue")
    ax.plot(le_x, z_dense, "b-",  lw=1.5, label="Leading edge")
    ax.plot(te_x, z_dense, "r-",  lw=1.5, label="Trailing edge")
    ax.plot(qc_x, z_dense, "g--", lw=1.0, label="Quarter chord (ref axis)")

    for z_mark, label in [(Z_SOB, f"Side-of-body\nη={ETA_SOB}"),
                           (Z_YEHUDI, f"Yehudi break\nη={ETA_YEHUDI}")]:
        ax.axhline(z_mark, color="grey", ls=":", lw=0.8)
        ax.text(te_x[-1] + 0.3, z_mark, label, va="center", fontsize=7, color="grey")

    ax.set_xlabel("Streamwise x  (m)", fontsize=11)
    ax.set_ylabel("Spanwise z  (m)",   fontsize=11)
    ax.set_title(
        f"NASA CRM Wing Planform  —  AR={AR}, λ={TAPER}, "
        f"LE sweep {LE_SWEEP_INBOARD_DEG}°/{LE_SWEEP_OUTBOARD_DEG}°, "
        f"Γ={DIHEDRAL_DEG}°,  b/2={SEMI_SPAN} m",
        fontsize=10,
    )
    ax.legend(fontsize=9)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()

    out = outdir / "crm_planform.png"
    fig.savefig(out, dpi=150)
    print(f"  [plot] Saved {out.name}")
    plt.show()


# ---------------------------------------------------------------------------
# 7.  Main driver
# ---------------------------------------------------------------------------

def main(args: argparse.Namespace) -> None:
    here   = pathlib.Path(__file__).parent
    outdir = here

    _banner("NASA CRM Wing  —  pyGeo Parametrisation")

    if not HAS_PYGEO:
        print(
            "\n  [WARNING] pyGeo not installed.\n"
            "  Airfoil .dat files and FFD lattice will be written.\n"
            "  Wing-surface geometry and DVGeometry steps are skipped.\n"
        )

    # ------------------------------------------------------------------
    # Step 1: write CRM65-proxy airfoil .dat files
    # ------------------------------------------------------------------
    print("\n── Step 1 · Airfoil section files ──")
    for fname, (m_pct, p_tenth, t_pct) in _CRM_SECTIONS.items():
        coords = naca4_coords(m_pct, p_tenth, t_pct)
        path   = outdir / fname
        write_dat(path, fname.replace(".dat", "").upper(), coords)
        print(f"  {fname:22s}  m={m_pct}%  p={p_tenth*10}%  t/c={t_pct}%  "
              f"({coords.shape[0]} pts)")

    # ------------------------------------------------------------------
    # Step 2: build 3-D wing surface with pyGeo.pyGeo
    # ------------------------------------------------------------------
    print("\n── Step 2 · 3-D wing surface ──")
    if HAS_PYGEO:
        naf       = len(_STA_Z)
        # Airfoil files: root section reused at SOB; yehudi and tip sections distinct
        af_files  = [
            str(outdir / "crm_root.dat"),     # η = 0.00  root
            str(outdir / "crm_root.dat"),     # η = 0.10  SOB  (same section)
            str(outdir / "crm_yehudi.dat"),   # η = 0.37  Yehudi
            str(outdir / "crm_tip.dat"),      # η = 1.00  tip
        ]
        wing = _pyGeo(
            "liftingSurface",
            xsections = af_files,
            scale     = _STA_CHORD,
            offset    = np.zeros((naf, 2)),
            x         = _STA_X_LE,
            y         = _STA_Y_LE,
            z         = _STA_Z,
            rotX      = [0.0] * naf,
            rotY      = [0.0] * naf,
            rotZ      = _STA_TWIST,          # geometric twist  (degrees)
            tip       = "rounded",
            bluntTe   = True,
            squareTeTip   = True,
            teHeight  = 0.005 * CHORD_TIP,  # 0.5% of tip chord
        )
        wing.writeTecplot(str(outdir / "crm_wing.dat"))
        wing.writeIGES(   str(outdir / "crm_wing.igs"))
        print("  crm_wing.dat  (Tecplot surface)")
        print("  crm_wing.igs  (IGES B-spline surface)")
    else:
        print("  (skipped – pyGeo unavailable)")

    # ------------------------------------------------------------------
    # Step 3: generate FFD lattice
    # ------------------------------------------------------------------
    print("\n── Step 3 · FFD lattice ──")
    ffd_path = build_ffd(outdir, n_x=8, n_y=2, n_z=10)

    # ------------------------------------------------------------------
    # Step 4: setup DVGeometry and embed test point set
    # ------------------------------------------------------------------
    print("\n── Step 4 · DVGeometry setup ──")
    if HAS_PYGEO:
        DVGeo, nRefAxPts = setup_dvgeo(str(ffd_path))

        pts = sample_wing_surface(n_chord=40, n_span=30)
        DVGeo.addPointSet(pts, "wing_surface")
        print(f"  Reference axis: {nRefAxPts} control points  (alignIndex='k')")
        print(f"  Embedded {pts.shape[0]:,} surface points into FFD volume")
        dv_vals = DVGeo.getValues()
        for name, v in dv_vals.items():
            print(f"    DV '{name}':  {len(v)} variables")
    else:
        print("  (skipped – pyGeo unavailable)")

    # ------------------------------------------------------------------
    # Step 5: apply test design variables (reproduces tutorial-GIF motion)
    # ------------------------------------------------------------------
    print("\n── Step 5 · Test deformation ──")
    if HAS_PYGEO:
        dv_dict = DVGeo.getValues()

        # Twist: linear wash-out from 0° at root to 5° at tip
        dv_dict["twist"]    = np.linspace(0.0, 5.0, nRefAxPts - 1)
        # Dihedral: slight upward bending toward tip
        dv_dict["dihedral"] = np.linspace(0.0, 0.5, nRefAxPts - 1)
        # Taper: root unchanged, tip chord 20% shorter
        dv_dict["taper"]    = np.array([1.0, 0.8])
        # Shape: small bump on every 5th local DV
        dv_dict["shape"][::5] = 0.08

        DVGeo.setDesignVars(dv_dict)
        DVGeo.update("wing_surface")

        DVGeo.writePlot3d(   str(outdir / "crm_ffd_deformed.xyz"))
        DVGeo.writePointSet("wing_surface", str(outdir / "crm_surf_deformed"))
        print("  crm_ffd_deformed.xyz   (deformed FFD lattice)")
        print("  crm_surf_deformed.dat  (deformed surface points)")
    else:
        print("  (skipped – pyGeo unavailable)")

    # ------------------------------------------------------------------
    # Summary table
    # ------------------------------------------------------------------
    print("\n── Planform summary ──")
    print(f"  {'η':>5}  {'z (m)':>8}  {'x_LE (m)':>9}  "
          f"{'y_LE (m)':>9}  {'chord (m)':>10}  {'twist (°)':>10}")
    for z, x_le, y_le, chord, twist in zip(
        _STA_Z, _STA_X_LE, _STA_Y_LE, _STA_CHORD, _STA_TWIST
    ):
        eta = z / SEMI_SPAN
        print(f"  {eta:>5.2f}  {z:>8.3f}  {x_le:>9.3f}  "
              f"{y_le:>9.3f}  {chord:>10.3f}  {twist:>10.1f}")

    print(f"\n  CHORD_ROOT   = {CHORD_ROOT:.4f} m")
    print(f"  CHORD_YEHUDI = {CHORD_YEHUDI:.4f} m")
    print(f"  CHORD_TIP    = {CHORD_TIP:.4f} m")
    print(f"  CHORD_TIP/ROOT (taper check) = {CHORD_TIP/CHORD_ROOT:.4f}  "
          f"(target {TAPER})")
    # Area check
    half_area = (
        (CHORD_ROOT + CHORD_YEHUDI) / 2.0 * Z_YEHUDI
        + (CHORD_YEHUDI + CHORD_TIP) / 2.0 * (Z_TIP - Z_YEHUDI)
    )
    print(f"  Sref check   = {2*half_area:.2f} m²  (target {SREF} m²)")

    if args.plot:
        print("\n── Planform plot ──")
        plot_planform(outdir)

    print("\nDone.\n")


def _banner(msg: str) -> None:
    line = "─" * (len(msg) + 4)
    print(f"\n{line}")
    print(f"  {msg}")
    print(f"{line}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="NASA CRM wing pyGeo parametrisation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Save and display a planform PNG (requires matplotlib)",
    )
    main(parser.parse_args())
