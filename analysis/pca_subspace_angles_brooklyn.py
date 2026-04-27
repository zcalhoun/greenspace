#!/usr/bin/env python3
"""
PCA Subspace Angles — Brooklyn (pm traversal, 500m buffer)
==========================================================
Implements the diagnostic from notes/03_Methods/pca_subspace_angles.md.

Driving question
----------------
How much additional information does the 0.6m Greenspace mapping provide,
over and above what is already captured by NLCD land cover and
NDVI/Albedo (Sentinel-2)?

Approach
--------
For each of the 9 219 pm traversal points we compute:

  X  (baseline)  = [ CLR(NLCD class fractions) | mean NDVI | mean Albedo ]
                   15 CLR columns + 2 continuous columns, all standardised
                   to unit variance.

  Z  (probe)     = CLR(Greenspace class fractions)
                   7 CLR columns, column-centred.

We then measure how much of the Greenspace subspace lies *outside* the
NLCD+NDVI/Albedo subspace using the residual variance R⊥ and the set of
principal angles between the two subspaces.

Speed design
------------
The original loop-per-point approach was O(n · C · R²/res²).  This version
uses a 2-D integral image (prefix sum) per class/band: one O(H·W) build step,
then O(r_pix_y · n) horizontal-strip lookups give the exact circular-buffer
sum for every point simultaneously.  The GS raster stays at its native 0.6m
resolution throughout.  Runtime: ~30–60 s instead of ~10 min.

Usage
-----
    python analysis/pca_subspace_angles_brooklyn.py

Outputs  →  notes/05_Results/
    pca_subspace_angles_brooklyn.md
    pca_subspace_angles_brooklyn.json
    figures/principal_angles_500m.png
    figures/composition_variance.png
"""

import json
import time
import warnings
from pathlib import Path

import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from rasterio.transform import rowcol

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
REPO      = Path(__file__).resolve().parents[1]
DATA      = Path("/Users/zdc6/Desktop/Greenspace Analysis")
NLCD_PATH = DATA / "data/nlcd/brooklyn.tif"
GS_PATH   = DATA / "data/0.6m-Greenspace_mapping/Brooklyn__NY_resultv2.tif"
NDVI_PATH = DATA / "data/ndvi_albedo/Brooklyn.tif"
TRAV_PATH = DATA / "data/traversals/brooklyn/pm/trav.shp"
OUT_DIR   = REPO / "notes/05_Results"
FIG_DIR   = OUT_DIR / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ── Land-cover class definitions ──────────────────────────────────────────────
NLCD_CLASSES = [11, 21, 22, 23, 24, 31, 41, 42, 43, 52, 71, 81, 82, 90, 95]
NLCD_LABELS  = {
    11: "Open Water",    21: "Dev. Open",      22: "Dev. Low",
    23: "Dev. Med.",     24: "Dev. High",       31: "Barren",
    41: "Dec. Forest",   42: "Ev. Forest",      43: "Mixed Forest",
    52: "Shrub",         71: "Herbaceous",      81: "Hay/Pasture",
    82: "Cultivated",    90: "Woody Wetland",   95: "Herb. Wetland",
}
GS_CLASSES   = [0, 4, 6, 8, 12, 13, 14]
GS_LABELS    = {
    0:  "Not Vegetated",           4:  "Dec. Broadleaf Forest",
    6:  "Ev. Needleleaf Forest",   8:  "Dec. Needleleaf Forest",
    12: "Ev. Shrubland",           13: "Dec. Shrubland",
    14: "Grassland",
}
NDVI_BANDS   = ["NDVI", "Albedo"]

RADIUS_M  = 500
REF_LAT   = 40.68  # Brooklyn centroid latitude


# ── Geographic helpers ─────────────────────────────────────────────────────────
def meters_to_deg(radius_m, lat=REF_LAT):
    """(delta_lat_deg, delta_lon_deg) for a circle of radius_m metres."""
    lat_deg = radius_m / 111_320.0
    lon_deg = radius_m / (111_320.0 * np.cos(np.radians(lat)))
    return lat_deg, lon_deg


def pixel_radii(radius_m, transform):
    """(r_pix_y, r_pix_x) for a given raster transform."""
    r_lat, r_lon = meters_to_deg(radius_m)
    res_y = abs(transform.e)
    res_x = abs(transform.a)
    return int(np.ceil(r_lat / res_y)) + 1, int(np.ceil(r_lon / res_x)) + 1


def point_pixel_coords(points_lonlat, transform, H, W):
    """Clipped (rows, cols) integer arrays for all points."""
    lons, lats = points_lonlat[:, 0], points_lonlat[:, 1]
    rows, cols = rowcol(transform, lons, lats)
    rows = np.clip(np.asarray(rows), 0, H - 1)
    cols = np.clip(np.asarray(cols), 0, W - 1)
    return rows, cols


# ── Raster loading ─────────────────────────────────────────────────────────────
def load_window(raster_path, points_lonlat, radius_m, n_bands=1):
    """
    Load the minimal rectangular region of raster_path that covers all
    points plus radius_m metres on each side.

    Returns (data, transform)  where data is (H, W) for n_bands=1
    or (n_bands, H, W) for n_bands > 1.
    """
    r_lat, r_lon = meters_to_deg(radius_m)
    lons, lats = points_lonlat[:, 0], points_lonlat[:, 1]

    with rasterio.open(raster_path) as src:
        r0, c0 = rowcol(src.transform, lons.min() - r_lon, lats.max() + r_lat)
        r1, c1 = rowcol(src.transform, lons.max() + r_lon, lats.min() - r_lat)
        r0 = max(0, r0);  c0 = max(0, c0)
        r1 = min(src.height - 1, r1);  c1 = min(src.width  - 1, c1)
        win = rasterio.windows.Window(c0, r0, c1 - c0 + 1, r1 - r0 + 1)
        data = src.read(window=win)          # (bands, H, W)
        tf   = src.window_transform(win)

    if n_bands == 1:
        data = data[0]                       # (H, W)
    print(f"  {raster_path.name}: {data.shape[-2]}×{data.shape[-1]} px, "
          f"res ≈ {abs(tf.a)*111_320:.1f}m/px")
    return data, tf


# ── Integral-image disk sampler ────────────────────────────────────────────────
def build_integral_image(arr, out_dtype):
    """
    2-D prefix sum of arr.  Returns a zero-padded (H+1, W+1) array of
    out_dtype so that a rectangle [r0..r1, c0..c1] is queryable in O(1)
    via the standard four-corner formula.

    Uses in-place cumsum to minimise peak memory.
    """
    H, W  = arr.shape
    # Allocate padded output; copy arr (with possible type conversion) into it.
    padded = np.zeros((H + 1, W + 1), dtype=out_dtype)
    padded[1:, 1:] = arr                              # implicit cast
    view = padded[1:, 1:]                             # (H, W) view — no copy
    np.cumsum(view, axis=1, out=view)
    np.cumsum(view, axis=0, out=view)
    return padded


def sample_disk(padded, rows, cols, r_pix_y, r_pix_x, H, W):
    """
    Exact circular-buffer sum for every (row, col) point using horizontal-
    strip lookups on a pre-built integral image.

    Complexity: O(2·r_pix_y · n)  — independent of the raster pixel count.
    """
    sums = np.zeros(len(rows), dtype=padded.dtype)

    for dy in range(-r_pix_y, r_pix_y + 1):
        # Horizontal half-width at this row offset (ellipse formula)
        t = 1.0 - (dy / (r_pix_y + 0.5)) ** 2
        if t <= 0:
            continue
        dx = int(np.floor(r_pix_x * np.sqrt(t)))

        r  = rows + dy
        c1 = np.clip(cols - dx, 0, W - 1)
        c2 = np.clip(cols + dx, 0, W - 1)
        ok = (r >= 0) & (r < H)
        r  = np.where(ok, r, 0)          # safe index (masked out below)

        # Four-corner integral-image query for a single-row strip
        strip = (padded[r + 1, c2 + 1]
                 - padded[r,     c2 + 1]
                 - padded[r + 1, c1    ]
                 + padded[r,     c1    ])
        sums += np.where(ok, strip, 0)

    return sums


# ── Feature extraction ─────────────────────────────────────────────────────────
def categorical_fractions(raster, transform, class_ids, rows, cols, radius_m):
    """
    Class composition within a circular buffer of radius_m for every point.
    Returns (n, C) float64 array, rows sum to 1.
    """
    H, W = raster.shape
    r_pix_y, r_pix_x = pixel_radii(radius_m, transform)

    n, C   = len(rows), len(class_ids)
    counts = np.zeros((n, C), dtype=np.float64)

    for j, cls in enumerate(class_ids):
        binary = (raster == cls)                   # bool, H×W
        padded = build_integral_image(binary, np.int32)
        counts[:, j] = sample_disk(padded, rows, cols, r_pix_y, r_pix_x, H, W)
        del padded                                 # free ~4 GB for GS

    totals = counts.sum(axis=1, keepdims=True)
    totals[totals == 0] = 1.0
    return counts / totals


def continuous_band_means(bands, transform, rows, cols, radius_m):
    """
    Mean of each band within a circular buffer of radius_m for every point.
    bands : (B, H, W) float array
    Returns (n, B) float64.
    """
    H, W   = bands.shape[-2], bands.shape[-1]
    r_pix_y, r_pix_x = pixel_radii(radius_m, transform)
    B      = bands.shape[0]
    result = np.zeros((len(rows), B), dtype=np.float64)

    # Pixel count inside disk (varies near raster edges)
    ones   = np.ones((H, W), dtype=np.float32)
    pcnt_p = build_integral_image(ones, np.float32)
    pcount = sample_disk(pcnt_p, rows, cols, r_pix_y, r_pix_x, H, W).clip(min=1)
    del pcnt_p

    for b in range(B):
        band   = bands[b].astype(np.float64)
        padded = build_integral_image(band, np.float64)
        sums   = sample_disk(padded, rows, cols, r_pix_y, r_pix_x, H, W)
        result[:, b] = sums / pcount
        del padded

    return result


# ── CLR transformation ─────────────────────────────────────────────────────────
def clr_transform(X, delta_factor=1e-3):
    """
    Centered log-ratio (Aitchison 1982).
    Zeros replaced with delta_factor × per-column minimum nonzero value.
    """
    X = X.copy()
    for j in range(X.shape[1]):
        col = X[:, j]
        nz  = col[col > 0]
        delta = (delta_factor * nz.min()) if len(nz) > 0 else delta_factor
        col[col == 0] = delta
        X[:, j] = col
    log_X = np.log(X)
    return log_X - log_X.mean(axis=1, keepdims=True)


# ── Subspace angle analysis ────────────────────────────────────────────────────
def subspace_angles(X_raw, Z_raw, var_threshold=0.95):
    """
    Principal angles between the baseline subspace (X) and the probe
    subspace (Z), following pca_subspace_angles.md.

    X_raw  : (n, p)  — combined NLCD + NDVI/Albedo features (already CLR'd / scaled)
    Z_raw  : (n, q)  — Greenspace compositions

    Returns a dict of all diagnostic quantities.
    """
    # ── Prepare baseline X ──
    # CLR columns are already produced by the caller; here we just
    # column-centre and scale each feature to unit variance.
    X = X_raw.copy()
    X -= X.mean(axis=0)
    stds = X.std(axis=0)
    stds[stds == 0] = 1.0
    X /= stds

    # ── Prepare probe Z ──
    Z = clr_transform(Z_raw)
    Z -= Z.mean(axis=0)

    # Step 1 — SVD of X
    U_X, s_X, _ = np.linalg.svd(X, full_matrices=False)
    var_X  = s_X ** 2
    cv_X   = np.cumsum(var_X) / var_X.sum()
    k      = int(np.searchsorted(cv_X, var_threshold)) + 1
    k      = min(k, U_X.shape[1])
    U_Xk   = U_X[:, :k]

    # Step 2 — SVD of Z
    U_Z, s_Z, _ = np.linalg.svd(Z, full_matrices=False)
    var_Z  = s_Z ** 2
    cv_Z   = np.cumsum(var_Z) / var_Z.sum()
    m      = int(np.searchsorted(cv_Z, var_threshold)) + 1
    m      = min(m, U_Z.shape[1])
    U_Zm   = U_Z[:, :m]

    # Step 3 — Cross-Gram → principal angles
    M       = U_Xk.T @ U_Zm
    _, gammas, _ = np.linalg.svd(M, full_matrices=False)
    s_ang   = min(k, m)
    gammas  = np.clip(gammas[:s_ang], 0.0, 1.0)
    angles  = np.degrees(np.arccos(gammas))

    # Step 4 — Residual variance
    Z_proj  = U_Xk @ (U_Xk.T @ Z)
    Z_perp  = Z - Z_proj
    R_perp  = (np.linalg.norm(Z_perp, "fro") ** 2
               / np.linalg.norm(Z,     "fro") ** 2)

    return {
        "n_points":             int(X.shape[0]),
        "k_baseline":           int(k),
        "m_gs":                 int(m),
        "principal_angles_deg": angles.tolist(),
        "cosines":              gammas.tolist(),
        "mean_sq_cosine":       float(np.mean(gammas ** 2)),
        "residual_variance":    float(R_perp),
        "cumvar_baseline":      cv_X[:k].tolist(),
        "cumvar_gs":            cv_Z[:m].tolist(),
        "singular_vals_baseline": s_X.tolist(),
        "singular_vals_gs":     s_Z.tolist(),
    }


# ── Plots ──────────────────────────────────────────────────────────────────────
def plot_principal_angles(result, out_path):
    angles  = np.array(result["principal_angles_deg"])
    cosines = np.array(result["cosines"])
    s       = len(angles)
    R       = result["residual_variance"]
    k       = result["k_baseline"]
    m       = result["m_gs"]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    cmap = plt.cm.RdYlGn(np.linspace(0.15, 0.85, s))

    # Left: principal angles
    ax = axes[0]
    ax.bar(np.arange(1, s + 1), angles, color=cmap, edgecolor="grey", lw=0.5)
    ax.axhline(45, color="grey",  ls="--", lw=0.8, label="45°")
    ax.axhline(90, color="black", ls=":",  lw=0.8, label="90°")
    ax.set_xlabel("Principal angle index", fontsize=11)
    ax.set_ylabel("θᵢ  (degrees)", fontsize=11)
    ax.set_title(
        f"Principal angles: Greenspace vs NLCD + NDVI/Albedo\n"
        f"(r = {RADIUS_M}m,  {k} baseline PCs × {m} GS PCs)",
        fontsize=10,
    )
    ax.set_ylim(0, 95);  ax.legend(fontsize=9)

    # Right: squared cosines
    ax2 = axes[1]
    ax2.bar(np.arange(1, s + 1), cosines ** 2, color=cmap, edgecolor="grey", lw=0.5)
    ax2.axhline(0.5, color="grey", ls="--", lw=0.8, label="0.5")
    ax2.set_xlabel("Principal angle index", fontsize=11)
    ax2.set_ylabel("cos²(θᵢ) — per-dimension overlap", fontsize=11)
    ax2.set_title(
        f"Per-dimension overlap\n"
        f"mean cos² = {result['mean_sq_cosine']:.3f}   R⊥ = {R:.3f}",
        fontsize=10,
    )
    ax2.set_ylim(0, 1.05);  ax2.legend(fontsize=9)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path.name}")


def plot_variance_decomposition(result, out_path):
    """Scree plot of baseline and GS singular values."""
    sv_base = np.array(result["singular_vals_baseline"])
    sv_gs   = np.array(result["singular_vals_gs"])

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, sv, label, k in [
        (axes[0], sv_base, "NLCD + NDVI/Albedo (baseline)", result["k_baseline"]),
        (axes[1], sv_gs,   "Greenspace",                    result["m_gs"]),
    ]:
        var  = sv ** 2
        cv   = np.cumsum(var) / var.sum()
        ax.bar(np.arange(1, len(sv) + 1), var / var.sum(),
               color="#4393c3", edgecolor="grey", lw=0.4, label="% var per PC")
        ax.axvline(k + 0.5, color="#d6604d", ls="--", lw=1.2,
                   label=f"k={k} (≥95% var)")
        ax.set_xlabel("PC index", fontsize=10)
        ax.set_ylabel("Fraction of variance", fontsize=10)
        ax.set_title(label, fontsize=10)
        ax.legend(fontsize=8)

    fig.suptitle("Variance explained by leading PCs", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path.name}")


# ── Report writer ──────────────────────────────────────────────────────────────
def write_report(result):
    angles  = np.array(result["principal_angles_deg"])
    cosines = np.array(result["cosines"])
    k, m    = result["k_baseline"], result["m_gs"]
    R       = result["residual_variance"]
    msc     = result["mean_sq_cosine"]
    n       = result["n_points"]
    near_ortho    = int(np.sum(angles > 70))
    near_collinear = int(np.sum(angles < 20))
    novelty = "substantial" if R > 0.5 else "moderate" if R > 0.25 else "limited"

    angle_rows = "\n".join(
        f"| {i+1} | {a:.1f}° | {c:.3f} | {c**2:.3f} |"
        for i, (a, c) in enumerate(zip(angles, cosines))
    )

    md = f"""# PCA Subspace Angles: Brooklyn (pm, 500m buffer)
## Greenspace vs NLCD + NDVI/Albedo

### Summary

For each of {n} pm traversal points we extract class compositions within a 500m
circular buffer, apply a centered log-ratio transform, and measure how much of the
Greenspace feature subspace lies *outside* the combined NLCD + NDVI/Albedo subspace.

**Baseline X:** CLR(NLCD) + mean NDVI + mean Albedo, all standardised to unit variance.
**Probe Z:** CLR(Greenspace class fractions).

| Quantity | Value | Interpretation |
|---|---|---|
| Baseline PCs retained (≥95% var) | **{k}** | Effective dim. of NLCD + NDVI/Albedo |
| Greenspace PCs retained (≥95% var) | **{m}** | Effective dim. of Greenspace |
| **Residual variance R⊥** | **{R:.3f}** | Fraction of GS variance not spanned by baseline |
| Mean cos²(θ) | {msc:.3f} | Average per-dimension overlap ∈ [0, 1] |
| Near-orthogonal angles (>70°) | {near_ortho} of {len(angles)} | Genuinely new GS dimensions |
| Near-collinear angles (<20°) | {near_collinear} of {len(angles)} | Strongly redundant GS dimensions |

**Bottom line:** Greenspace has {novelty} geometric novelty relative to NLCD + NDVI/Albedo
at the 500m scale. **{int(round(R*100))}% of Greenspace variance** lies outside what the
baseline already captures.

---

### Principal Angles

The {len(angles)} principal angles between the {k}-dimensional baseline subspace and the
{m}-dimensional Greenspace subspace:

| # | θᵢ (°) | cos(θᵢ) | cos²(θᵢ) |
|---|--------|---------|----------|
{angle_rows}

- cos²(θᵢ) ≈ 1 → that GS dimension is nearly spanned by the baseline (redundant).
- cos²(θᵢ) ≈ 0 → genuinely orthogonal — information NLCD + NDVI/Albedo cannot express.

![Principal angles](figures/principal_angles_500m.png)

---

### Variance Decomposition

![Variance decomposition](figures/composition_variance.png)

---

### Interpretation

R⊥ = {R:.3f} is an upper bound on the fraction of temperature variance that could be
uniquely attributed to Greenspace above and beyond NLCD and NDVI/Albedo, assuming
linear predictability.  The {near_ortho} near-orthogonal dimension(s) (θ > 70°) represent
axes of land-cover variation that are genuinely new — likely fine-grained distinctions
between forest sub-types and vegetated/non-vegetated boundaries that NLCD's 30m
classification cannot resolve and that NDVI/Albedo do not discriminate.

---

### Methods

See `notes/03_Methods/pca_subspace_angles.md` for the full derivation.

**Feature construction (500m buffer):**
- NLCD class fractions → CLR transform → column-standardised
- Mean NDVI, mean Albedo within buffer → column-standardised
- Combined baseline X ∈ ℝⁿˣ⁽¹⁵⁺²⁾, then SVD to retain {k} PCs (≥95% var)
- Greenspace class fractions → CLR transform → column-centred → SVD to retain {m} PCs

**Extraction method:** 2-D integral image (prefix sum) per class/band; horizontal-strip
disk queries give exact circular-buffer counts for all {n} points simultaneously.
No resampling — GS raster used at native 0.6m resolution.

Zero replacement for CLR: δ = 0.001 × per-column minimum non-zero value.
Buffer radii converted from metres to degrees at Brooklyn centroid lat {REF_LAT}°N.

**Data:**
- NLCD:        `data/nlcd/brooklyn.tif`                            ({len(NLCD_CLASSES)} classes)
- Greenspace:  `data/0.6m-Greenspace_mapping/Brooklyn__NY_resultv2.tif` ({len(GS_CLASSES)} classes)
- NDVI/Albedo: `data/ndvi_albedo/Brooklyn.tif`                    (2 bands)
- Traversal:   `data/traversals/brooklyn/pm/trav.shp`             ({n} points)
"""
    out = OUT_DIR / "pca_subspace_angles_brooklyn.md"
    out.write_text(md)
    print(f"  Saved: {out.name}")


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    t0 = time.time()
    print("=" * 60)
    print("PCA Subspace Angles — Brooklyn pm / 500m buffer")
    print("=" * 60)

    # ── 1. Traversal points ──────────────────────────────────────────────────
    print("\n[1] Loading traversal points …")
    gdf    = gpd.read_file(TRAV_PATH).dropna(subset=["temp_f", "geometry"])
    coords = np.array([(g.x, g.y) for g in gdf.geometry])
    print(f"  {len(coords)} valid points.")

    # ── 2. Load raster windows ───────────────────────────────────────────────
    print(f"\n[2] Loading raster windows (radius = {RADIUS_M}m) …")
    nlcd_data, nlcd_tf   = load_window(NLCD_PATH, coords, RADIUS_M, n_bands=1)
    gs_data,   gs_tf     = load_window(GS_PATH,   coords, RADIUS_M, n_bands=1)
    ndvi_data, ndvi_tf   = load_window(NDVI_PATH,  coords, RADIUS_M, n_bands=2)

    # ── 3. Point pixel coordinates ───────────────────────────────────────────
    print("\n[3] Computing point pixel coordinates …")
    nlcd_rows, nlcd_cols = point_pixel_coords(coords, nlcd_tf, *nlcd_data.shape)
    gs_rows,   gs_cols   = point_pixel_coords(coords, gs_tf,   *gs_data.shape)
    ndvi_rows, ndvi_cols = point_pixel_coords(coords, ndvi_tf, ndvi_data.shape[1], ndvi_data.shape[2])

    # ── 4. Extract features ──────────────────────────────────────────────────
    print(f"\n[4] NLCD compositions ({len(NLCD_CLASSES)} classes) …")
    t1 = time.time()
    X_nlcd = categorical_fractions(nlcd_data, nlcd_tf, NLCD_CLASSES,
                                   nlcd_rows, nlcd_cols, RADIUS_M)
    print(f"  Done in {time.time()-t1:.1f}s  shape={X_nlcd.shape}")

    print(f"\n[5] Greenspace compositions ({len(GS_CLASSES)} classes) …")
    t1 = time.time()
    Z_gs = categorical_fractions(gs_data, gs_tf, GS_CLASSES,
                                  gs_rows, gs_cols, RADIUS_M)
    print(f"  Done in {time.time()-t1:.1f}s  shape={Z_gs.shape}")

    print("\n[6] NDVI / Albedo band means …")
    t1 = time.time()
    X_ndvi = continuous_band_means(ndvi_data, ndvi_tf,
                                   ndvi_rows, ndvi_cols, RADIUS_M)
    print(f"  Done in {time.time()-t1:.1f}s  shape={X_ndvi.shape}")

    # ── 5. Assemble baseline X ───────────────────────────────────────────────
    print("\n[7] Assembling baseline X = CLR(NLCD) | NDVI | Albedo …")
    X_clr  = clr_transform(X_nlcd)    # (n, 15)
    X_base = np.hstack([X_clr, X_ndvi])   # (n, 17)
    print(f"  Baseline shape: {X_base.shape}")
    print(f"  GS probe shape: {Z_gs.shape}")

    # ── 6. Subspace angles ───────────────────────────────────────────────────
    print("\n[8] Computing subspace angles …")
    result = subspace_angles(X_base, Z_gs)
    print(f"  Baseline PCs (k): {result['k_baseline']}")
    print(f"  GS PCs (m):       {result['m_gs']}")
    print(f"  Principal angles: {np.round(result['principal_angles_deg'], 1)}")
    print(f"  Mean cos²(θ):     {result['mean_sq_cosine']:.4f}")
    print(f"  Residual var R⊥:  {result['residual_variance']:.4f}")

    # ── 7. Plots ─────────────────────────────────────────────────────────────
    print("\n[9] Saving figures …")
    plot_principal_angles(result, FIG_DIR / "principal_angles_500m.png")
    plot_variance_decomposition(result, FIG_DIR / "composition_variance.png")

    # ── 8. Save outputs ──────────────────────────────────────────────────────
    print("\n[10] Writing results …")
    payload = {
        "city": "brooklyn", "traversal": "pm",
        "radius_m": RADIUS_M, "result": result,
    }
    (OUT_DIR / "pca_subspace_angles_brooklyn.json").write_text(
        json.dumps(payload, indent=2)
    )
    write_report(result)

    print(f"\nTotal time: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
