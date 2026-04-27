#!/usr/bin/env python3
"""
PCA Subspace Angles — all cities, pm traversal, 500m buffer
============================================================
Runs the diagnostic from notes/03_Methods/pca_subspace_angles.md for every
city that has a pm traversal.

For each city we measure how much of the Greenspace subspace lies *outside*
the combined NLCD + NDVI/Albedo subspace, using the residual variance R⊥
and the set of principal angles between the two subspaces.

Outputs (notes/05_Results/)
---------------------------
  cities/<city>_subspace_angles.md      per-city narrative + tables
  cities/<city>_subspace_angles.json    raw numeric results
  figures/cities/<city>_angles.png      per-city principal-angle chart
  pca_subspace_angles_all_cities.md     cross-city summary table + chart
  pca_subspace_angles_all_cities.json   all-city numeric results
  figures/all_cities_r_perp.png         ranked bar chart of R⊥ across cities

Usage
-----
    python analysis/pca_subspace_angles_all_cities.py
"""

import json
import time
import traceback
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
DATA     = Path("/Users/zdc6/Desktop/Greenspace Analysis")
NLCD_DIR = DATA / "data/nlcd"
GS_DIR   = DATA / "data/0.6m-Greenspace_mapping"
NDVI_DIR = DATA / "data/ndvi_albedo"
TRAV_DIR = DATA / "data/traversals"
OUT_DIR  = DATA / "notes/05_Results"
CITY_DIR = OUT_DIR / "cities"
FIG_DIR  = OUT_DIR / "figures"
CFIG_DIR = FIG_DIR / "cities"
for d in [CITY_DIR, CFIG_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── City catalogue ─────────────────────────────────────────────────────────────
# Maps traversal folder name → NDVI filename and GS filename.
CITIES = {
    "asheville":     {"ndvi": "Asheville.tif",       "gs": "Asheville__NC_resultv2.tif"},
    "atlanta":       {"ndvi": "Atlanta.tif",          "gs": "Atlanta__GA_resultv2.tif"},
    "brockton":      {"ndvi": "Brockton.tif",         "gs": "Brockton__MA_resultv2.tif"},
    "brooklyn":      {"ndvi": "Brooklyn.tif",         "gs": "Brooklyn__NY_resultv2.tif"},
    "cedar_rapids":  {"ndvi": "Cedar Rapids.tif",     "gs": "Cedar_Rapids__IA_resultv2.tif"},
    "charlotte":     {"ndvi": "Charlotte.tif",        "gs": "Charlotte__NC_resultv2.tif"},
    "columbus":      {"ndvi": "Columbus.tif",         "gs": "Columbus__OH_resultv2.tif"},
    "durham":        {"ndvi": "Durham.tif",           "gs": "Durham__NC_resultv2.tif"},
    "jacksonville":  {"ndvi": "Jacksonville.tif",     "gs": "Jacksonville__FL_resultv2.tif"},
    "knoxville":     {"ndvi": "Knoxville.tif",        "gs": "Knoxville__TN_resultv2.tif"},
    "little_rock":   {"ndvi": "Little Rock.tif",      "gs": "Little_Rock__AR_resultv2.tif"},
    "lynchburg":     {"ndvi": "Lynchburg.tif",        "gs": "Lynchburg__VA_resultv2.tif"},
    "manhattan":     {"ndvi": "Manhattan.tif",        "gs": "Manhattan__NY_resultv2.tif"},
    "nashville":     {"ndvi": "Nashville.tif",        "gs": "Nashville__TN_resultv2.tif"},
    "oklahoma_city": {"ndvi": "Oklahoma City.tif",    "gs": "Oklahoma_City__OK_resultv2.tif"},
    "omaha":         {"ndvi": "Omaha.tif",            "gs": "Omaha__NE_resultv2.tif"},
    "philadelphia":  {"ndvi": "Philadelphia.tif",     "gs": "Philadelphia__PA_resultv2.tif"},
    "richmond":      {"ndvi": "Richmond.tif",         "gs": "Richmond__VA_resultv2.tif"},
    "san_diego":     {"ndvi": "San Diego.tif",        "gs": "San_Diego__CA_resultv2.tif"},
    "san_francisco": {"ndvi": "San Francisco.tif",    "gs": "San_Francisco__CA_resultv2.tif"},
    "spokane":       {"ndvi": "Spokane.tif",          "gs": "Spokane__WA_resultv2.tif"},
    "toledo":        {"ndvi": "Toledo.tif",           "gs": "Toledo__OH_resultv2.tif"},
}

GS_LABELS = {
    0: "Not Vegetated",          2: "Ev. Broadleaf Forest",
    4: "Dec. Broadleaf Forest",  6: "Ev. Needleleaf Forest",
    8: "Dec. Needleleaf Forest", 10: "Mixed-leaf Forest",
    12: "Ev. Shrubland",         13: "Dec. Shrubland",
    14: "Grassland",             15: "Unknown",
}
NLCD_LABELS = {
    11: "Open Water",   21: "Dev. Open",    22: "Dev. Low",
    23: "Dev. Med.",    24: "Dev. High",    31: "Barren",
    41: "Dec. Forest",  42: "Ev. Forest",   43: "Mixed Forest",
    51: "Dwarf Shrub",  52: "Shrub",        71: "Herbaceous",
    72: "Sedge",        73: "Lichens",      74: "Moss",
    81: "Hay/Pasture",  82: "Cultivated",   90: "Woody Wetland",
    95: "Herb. Wetland",
}

RADIUS_M = 500
N_SAMPLE = 1000   # points sampled per city; more than enough for stable SVD


# ── Geographic helpers ─────────────────────────────────────────────────────────
def meters_to_deg(radius_m, lat):
    lat_deg = radius_m / 111_320.0
    lon_deg = radius_m / (111_320.0 * np.cos(np.radians(lat)))
    return lat_deg, lon_deg


# ── Feature extraction — per-point windowed reads ─────────────────────────────
# For each sampled point we open a small window of exactly
# (2*r_pix_y + 1) × (2*r_pix_x + 1) pixels — the 500m buffer footprint.
# At 0.6 m/px that is ~1667 × 1097 px ≈ 1.8 M px per point, per raster.
# With 1000 points and an open file handle, the OS tile-cache keeps this fast.

def _disk_mask(r_pix_y, r_pix_x, res_y_deg, res_x_deg, r_lat_deg, r_lon_deg):
    """Pre-built boolean circular-mask template (2*r_pix+1 grid)."""
    dy = np.arange(-r_pix_y, r_pix_y + 1)
    dx = np.arange(-r_pix_x, r_pix_x + 1)
    DX, DY = np.meshgrid(dx, dy)
    return ((DX * res_x_deg / r_lon_deg) ** 2
          + (DY * res_y_deg / r_lat_deg) ** 2) <= 1.0


def categorical_fractions(raster_path, points_lonlat, class_ids, radius_m, lat):
    """
    Compute class compositions within a circular buffer for each point.
    Loads the entire raster into memory once, then does direct numpy array
    indexing per point — avoids repeated strip-decompression on row-striped TIFFs.
    Returns (n, C) float array, rows sum to 1.
    """
    r_lat, r_lon = meters_to_deg(radius_m, lat)
    n, C = len(points_lonlat), len(class_ids)
    fracs = np.zeros((n, C), dtype=np.float64)

    with rasterio.open(raster_path) as src:
        res_y   = abs(src.transform.e)
        res_x   = abs(src.transform.a)
        r_pix_y = int(np.ceil(r_lat / res_y)) + 1
        r_pix_x = int(np.ceil(r_lon / res_x)) + 1
        circ    = _disk_mask(r_pix_y, r_pix_x, res_y, res_x, r_lat, r_lon)
        arr     = src.read(1)           # load full raster into memory (uint8)
        H, W    = arr.shape
        tf      = src.transform

    for i, (lon, lat_pt) in enumerate(points_lonlat):
        row0, col0 = rowcol(tf, lon, lat_pt)
        r_lo = max(0, row0 - r_pix_y);  r_hi = min(H - 1, row0 + r_pix_y)
        c_lo = max(0, col0 - r_pix_x);  c_hi = min(W - 1, col0 + r_pix_x)
        if r_hi < r_lo or c_hi < c_lo:
            fracs[i] = np.ones(C) / C   # point outside raster extent
            continue
        patch = arr[r_lo:r_hi+1, c_lo:c_hi+1]
        mr_lo = r_lo - (row0 - r_pix_y);  mr_hi = mr_lo + patch.shape[0]
        mc_lo = c_lo - (col0 - r_pix_x);  mc_hi = mc_lo + patch.shape[1]
        m      = circ[mr_lo:mr_hi, mc_lo:mc_hi]
        pixels = patch[m]
        counts = np.array([np.sum(pixels == c) for c in class_ids], dtype=np.float64)
        total  = counts.sum()
        fracs[i] = counts / total if total > 0 else np.ones(C) / C

    return fracs


def continuous_band_means(raster_path, points_lonlat, n_bands, radius_m, lat):
    """
    Mean of each band within a circular buffer, NaN-safe.
    Loads the entire raster into memory once for fast numpy indexing.
    Returns (n, n_bands) float array.
    """
    r_lat, r_lon = meters_to_deg(radius_m, lat)
    n      = len(points_lonlat)
    result = np.zeros((n, n_bands), dtype=np.float64)

    with rasterio.open(raster_path) as src:
        res_y   = abs(src.transform.e)
        res_x   = abs(src.transform.a)
        r_pix_y = int(np.ceil(r_lat / res_y)) + 1
        r_pix_x = int(np.ceil(r_lon / res_x)) + 1
        circ    = _disk_mask(r_pix_y, r_pix_x, res_y, res_x, r_lat, r_lon)
        arr     = src.read().astype(np.float64)   # (bands, H, W)
        _, H, W = arr.shape
        tf      = src.transform

    for i, (lon, lat_pt) in enumerate(points_lonlat):
        row0, col0 = rowcol(tf, lon, lat_pt)
        r_lo = max(0, row0 - r_pix_y);  r_hi = min(H - 1, row0 + r_pix_y)
        c_lo = max(0, col0 - r_pix_x);  c_hi = min(W - 1, col0 + r_pix_x)
        if r_hi < r_lo or c_hi < c_lo:
            continue                      # point outside raster; leave row as 0
        patch = arr[:, r_lo:r_hi+1, c_lo:c_hi+1]
        mr_lo = r_lo - (row0 - r_pix_y);  mr_hi = mr_lo + patch.shape[1]
        mc_lo = c_lo - (col0 - r_pix_x);  mc_hi = mc_lo + patch.shape[2]
        m      = circ[mr_lo:mr_hi, mc_lo:mc_hi]
        for b in range(n_bands):
            vals  = patch[b][m]
            valid = vals[np.isfinite(vals)]
            result[i, b] = valid.mean() if len(valid) > 0 else 0.0

    return result


# ── CLR transformation ─────────────────────────────────────────────────────────
def clr_transform(X, delta_factor=1e-3):
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
    # Guard against any residual NaN/Inf (e.g. from edge points with no raster coverage)
    X_raw = np.where(np.isfinite(X_raw), X_raw, 0.0)
    Z_raw = np.where(np.isfinite(Z_raw), Z_raw, 0.0)

    X = X_raw.copy()
    X -= X.mean(axis=0)
    stds = X.std(axis=0);  stds[stds == 0] = 1.0
    X /= stds

    Z = clr_transform(Z_raw)
    Z -= Z.mean(axis=0)

    U_X, s_X, _ = np.linalg.svd(X, full_matrices=False)
    cv_X = np.cumsum(s_X**2) / (s_X**2).sum()
    k    = min(int(np.searchsorted(cv_X, var_threshold)) + 1, U_X.shape[1])
    U_Xk = U_X[:, :k]

    U_Z, s_Z, _ = np.linalg.svd(Z, full_matrices=False)
    cv_Z = np.cumsum(s_Z**2) / (s_Z**2).sum()
    m    = min(int(np.searchsorted(cv_Z, var_threshold)) + 1, U_Z.shape[1])
    U_Zm = U_Z[:, :m]

    M      = U_Xk.T @ U_Zm
    _, gammas, _ = np.linalg.svd(M, full_matrices=False)
    gammas = np.clip(gammas[:min(k, m)], 0.0, 1.0)
    angles = np.degrees(np.arccos(gammas))

    Z_perp = Z - U_Xk @ (U_Xk.T @ Z)
    R_perp = (np.linalg.norm(Z_perp, "fro")**2
              / np.linalg.norm(Z,     "fro")**2)

    return {
        "n_points":               int(X.shape[0]),
        "k_baseline":             int(k),
        "m_gs":                   int(m),
        "principal_angles_deg":   angles.tolist(),
        "cosines":                gammas.tolist(),
        "mean_sq_cosine":         float(np.mean(gammas**2)),
        "residual_variance":      float(R_perp),
        "cumvar_baseline":        cv_X[:k].tolist(),
        "cumvar_gs":              cv_Z[:m].tolist(),
        "singular_vals_baseline": s_X.tolist(),
        "singular_vals_gs":       s_Z.tolist(),
    }


# ── Per-city plot ──────────────────────────────────────────────────────────────
def plot_city(result, city, out_path):
    angles  = np.array(result["principal_angles_deg"])
    cosines = np.array(result["cosines"])
    s       = len(angles)
    cmap    = plt.cm.RdYlGn(np.linspace(0.15, 0.85, s))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    ax = axes[0]
    ax.bar(np.arange(1, s+1), angles, color=cmap, edgecolor="grey", lw=0.4)
    ax.axhline(45, color="grey",  ls="--", lw=0.8)
    ax.axhline(90, color="black", ls=":",  lw=0.8)
    ax.set_ylim(0, 95);  ax.set_xlabel("Angle index");  ax.set_ylabel("θᵢ (°)")
    ax.set_title(f"{city.replace('_',' ').title()} — principal angles\n"
                 f"(k={result['k_baseline']}, m={result['m_gs']})", fontsize=10)

    ax2 = axes[1]
    ax2.bar(np.arange(1, s+1), cosines**2, color=cmap, edgecolor="grey", lw=0.4)
    ax2.axhline(0.5, color="grey", ls="--", lw=0.8)
    ax2.set_ylim(0, 1.05);  ax2.set_xlabel("Angle index")
    ax2.set_ylabel("cos²(θᵢ)");
    ax2.set_title(f"Overlap per dimension\n"
                  f"R⊥ = {result['residual_variance']:.3f}, "
                  f"mean cos² = {result['mean_sq_cosine']:.3f}", fontsize=10)

    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


# ── Per-city markdown ──────────────────────────────────────────────────────────
def write_city_report(city, result, nlcd_classes, gs_classes, elapsed):
    angles  = np.array(result["principal_angles_deg"])
    cosines = np.array(result["cosines"])
    k, m    = result["k_baseline"], result["m_gs"]
    R       = result["residual_variance"]
    n       = result["n_points"]
    novelty = "substantial" if R > 0.5 else "moderate" if R > 0.25 else "limited"
    near_ortho = int(np.sum(angles > 70))

    angle_rows = "\n".join(
        f"| {i+1} | {a:.1f}° | {c:.3f} | {c**2:.3f} |"
        for i, (a, c) in enumerate(zip(angles, cosines))
    )
    gs_class_list = ", ".join(
        f"{c} ({GS_LABELS.get(c, '?')})" for c in gs_classes
    )
    nlcd_class_list = ", ".join(
        NLCD_LABELS.get(c, str(c)) for c in nlcd_classes
    )

    md = f"""# PCA Subspace Angles: {city.replace('_', ' ').title()} (pm, 500m buffer)

## Summary

| Quantity | Value |
|---|---|
| Points (pm traversal) | {n} |
| Baseline PCs (≥95% var) | {k} |
| Greenspace PCs (≥95% var) | {m} |
| **Residual variance R⊥** | **{R:.3f}** |
| Mean cos²(θ) | {result['mean_sq_cosine']:.3f} |
| Near-orthogonal angles (>70°) | {near_ortho} of {len(angles)} |

**{int(round(R*100))}% of Greenspace variance** lies outside the NLCD + NDVI/Albedo subspace —
{novelty} geometric novelty at the 500m scale.

---

## Principal Angles

Baseline: CLR(NLCD) + mean NDVI + mean Albedo (standardised), retaining {k} PCs.
Greenspace probe: CLR(GS fractions), retaining {m} PCs.

| # | θᵢ (°) | cos(θᵢ) | cos²(θᵢ) |
|---|--------|---------|----------|
{angle_rows}

![Principal angles](../figures/cities/{city}_angles.png)

---

## Data

- **NLCD classes present ({len(nlcd_classes)}):** {nlcd_class_list}
- **GS classes present ({len(gs_classes)}):** {gs_class_list}
- **Analysis time:** {elapsed:.1f}s
"""
    out = CITY_DIR / f"{city}_subspace_angles.md"
    out.write_text(md)


# ── All-cities summary plot ────────────────────────────────────────────────────
def plot_summary(all_results):
    cities = [r["city"] for r in all_results]
    r_perps = [r["result"]["residual_variance"] for r in all_results]
    order   = np.argsort(r_perps)[::-1]

    fig, ax = plt.subplots(figsize=(14, 5))
    colors  = plt.cm.RdYlGn(np.array(r_perps)[order])
    bars    = ax.bar(np.arange(len(cities)),
                     np.array(r_perps)[order],
                     color=colors, edgecolor="grey", lw=0.4)
    ax.set_xticks(np.arange(len(cities)))
    ax.set_xticklabels(
        [cities[i].replace("_", " ").title() for i in order],
        rotation=40, ha="right", fontsize=9,
    )
    ax.axhline(0.5, color="grey", ls="--", lw=0.8, label="R⊥ = 0.5")
    ax.set_ylabel("Residual variance R⊥", fontsize=11)
    ax.set_title(
        "Fraction of Greenspace variance outside NLCD + NDVI/Albedo subspace\n"
        "(500m buffer, pm traversal)",
        fontsize=11,
    )
    ax.set_ylim(0, 1);  ax.legend(fontsize=9)
    fig.tight_layout()
    out = FIG_DIR / "all_cities_r_perp.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out.name}")


# ── All-cities summary report ──────────────────────────────────────────────────
def write_summary(all_results):
    rows = sorted(all_results,
                  key=lambda r: r["result"]["residual_variance"],
                  reverse=True)

    table = "\n".join(
        f"| {r['city'].replace('_',' ').title()} "
        f"| {r['result']['n_points']} "
        f"| {r['result']['k_baseline']} "
        f"| {r['result']['m_gs']} "
        f"| **{r['result']['residual_variance']:.3f}** "
        f"| {r['result']['mean_sq_cosine']:.3f} "
        f"| {int(np.sum(np.array(r['result']['principal_angles_deg']) > 70))} |"
        for r in rows
    )

    r_vals  = [r["result"]["residual_variance"] for r in all_results]
    mean_R  = float(np.mean(r_vals))
    med_R   = float(np.median(r_vals))
    top3    = [r["city"].replace("_", " ").title()
               for r in rows[:3]]
    bot3    = [r["city"].replace("_", " ").title()
               for r in rows[-3:]]

    md = f"""# PCA Subspace Angles — All Cities Summary
## Greenspace vs NLCD + NDVI/Albedo (pm traversal, 500m buffer)

### Overview

Across {len(all_results)} cities, Greenspace retains substantial geometric novelty
relative to NLCD + NDVI/Albedo at the 500m scale.

| Statistic | R⊥ |
|---|---|
| Mean | {mean_R:.3f} |
| Median | {med_R:.3f} |
| Highest novelty | {", ".join(top3)} |
| Lowest novelty | {", ".join(bot3)} |

![R⊥ by city](figures/all_cities_r_perp.png)

---

### Per-City Results (ranked by R⊥)

| City | Points | Baseline PCs (k) | GS PCs (m) | R⊥ | Mean cos² | θ>70° |
|------|--------|-----------------|-----------|-----|-----------|-------|
{table}

---

### Interpretation

R⊥ is the fraction of Greenspace feature variance that lies outside the subspace
already spanned by NLCD land cover + NDVI + Albedo within 500m buffers.
It is an upper bound on the fraction of temperature variance uniquely
attributable to Greenspace above those baselines, under a linear model.

A consistently high R⊥ across cities indicates that the 0.6m Greenspace mapping
introduces class distinctions (e.g. deciduous vs. needleleaved sub-types,
fine-grained urban vegetation boundaries) that neither NLCD's categorical
mapping nor NDVI/Albedo's spectral indices can fully capture.

See individual city reports in `notes/05_Results/cities/` for principal-angle
breakdowns, and `notes/03_Methods/pca_subspace_angles.md` for the full
mathematical derivation.
"""
    out = OUT_DIR / "pca_subspace_angles_all_cities.md"
    out.write_text(md)
    print(f"  Saved: {out.name}")


# ── Single-city runner ─────────────────────────────────────────────────────────
def run_city(city, cfg):
    t0 = time.time()
    print(f"\n{'─'*55}")
    print(f"  {city.replace('_',' ').title()}")
    print(f"{'─'*55}")

    # Traversal points
    trav_path = TRAV_DIR / city / "pm" / "trav.shp"
    gdf       = gpd.read_file(trav_path).dropna(subset=["geometry"])
    temp_col  = next((c for c in gdf.columns if "temp" in c.lower()), None)
    if temp_col:
        gdf = gdf.dropna(subset=[temp_col])
    coords  = np.array([(g.x, g.y) for g in gdf.geometry])
    ref_lat = float(coords[:, 1].mean())
    print(f"  {len(coords)} traversal points,  lat ≈ {ref_lat:.2f}°")

    # Subsample evenly along the traversal route
    idx      = np.round(np.linspace(0, len(coords) - 1,
                                    min(N_SAMPLE, len(coords)))).astype(int)
    coords_s = coords[idx]
    print(f"  Subsampled to {len(coords_s)} points")

    nlcd_path = NLCD_DIR / f"{city}.tif"
    gs_path   = GS_DIR   / cfg["gs"]
    ndvi_path = NDVI_DIR / cfg["ndvi"]

    # Auto-detect present classes from a downsampled tile (avoids reading the
    # full city-wide raster into memory)
    print("  Detecting classes …", end=" ", flush=True)
    with rasterio.open(nlcd_path) as src:
        probe = src.read(1,
                         out_shape=(min(src.height, 2000), min(src.width, 2000)),
                         resampling=rasterio.enums.Resampling.nearest)
        nodata_nlcd  = src.nodata if src.nodata is not None else 0
        nlcd_classes = [int(v) for v in sorted(np.unique(probe))
                        if v != 0 and v != nodata_nlcd]
    with rasterio.open(gs_path) as src:
        probe = src.read(1,
                         out_shape=(min(src.height, 2000), min(src.width, 2000)),
                         resampling=rasterio.enums.Resampling.nearest)
        # GS class 0 ("Not Vegetated") is a valid land-cover class; include it.
        # Negative nodata values don't occur in uint8 rasters so no filter needed.
        gs_classes = [int(v) for v in sorted(np.unique(probe))]
    print(f"NLCD={nlcd_classes}  GS={gs_classes}")

    # ── Feature extraction (per-point windowed reads) ──────────────────────────
    print("  NLCD fractions …", end=" ", flush=True)
    t1     = time.time()
    X_nlcd = categorical_fractions(nlcd_path, coords_s, nlcd_classes, RADIUS_M, ref_lat)
    print(f"{time.time()-t1:.1f}s")

    print("  GS fractions …", end=" ", flush=True)
    t1   = time.time()
    Z_gs = categorical_fractions(gs_path, coords_s, gs_classes, RADIUS_M, ref_lat)
    print(f"{time.time()-t1:.1f}s")

    print("  NDVI/Albedo means …", end=" ", flush=True)
    t1     = time.time()
    X_ndvi = continuous_band_means(ndvi_path, coords_s, 2, RADIUS_M, ref_lat)
    print(f"{time.time()-t1:.1f}s")

    # Baseline X = CLR(NLCD) | NDVI | Albedo  (continuous bands z-scored inside
    # subspace_angles via the shared standardisation step)
    X_base = np.hstack([clr_transform(X_nlcd), X_ndvi])

    # Subspace angles
    result  = subspace_angles(X_base, Z_gs)
    elapsed = time.time() - t0

    print(f"  → k={result['k_baseline']}  m={result['m_gs']}  "
          f"R⊥={result['residual_variance']:.4f}  "
          f"({elapsed:.1f}s total)")

    # Save per-city outputs
    plot_city(result, city, CFIG_DIR / f"{city}_angles.png")
    write_city_report(city, result, nlcd_classes, gs_classes, elapsed)
    city_json = {
        "city": city, "traversal": "pm",
        "radius_m": RADIUS_M, "ref_lat": ref_lat,
        "nlcd_classes": nlcd_classes, "gs_classes": gs_classes,
        "elapsed_s": elapsed, "result": result,
    }
    (CITY_DIR / f"{city}_subspace_angles.json").write_text(
        json.dumps(city_json, indent=2)
    )

    return city_json


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    t_total = time.time()
    print("=" * 55)
    print("PCA Subspace Angles — all cities (pm, 500m)")
    print("=" * 55)

    all_results = []
    failed      = []

    for city, cfg in CITIES.items():
        try:
            res = run_city(city, cfg)
            all_results.append(res)
        except Exception as e:
            print(f"\n  !! FAILED: {city}: {e}")
            traceback.print_exc()
            failed.append(city)

    if not all_results:
        print("No cities completed — exiting.")
        return

    # Summary outputs
    print(f"\n{'='*55}")
    print("Writing summary …")
    plot_summary(all_results)
    write_summary(all_results)

    # Save all-city JSON
    (OUT_DIR / "pca_subspace_angles_all_cities.json").write_text(
        json.dumps(all_results, indent=2)
    )

    total = time.time() - t_total
    print(f"\nDone — {len(all_results)} cities in {total/60:.1f} min"
          + (f"  ({len(failed)} failed: {failed})" if failed else ""))


if __name__ == "__main__":
    main()
