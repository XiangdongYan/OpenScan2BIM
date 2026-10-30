"""
OpenScan2BIM — Step 01b: Deterministic Structural Primitive Extraction
=======================================================================
v02 (2026-07-09) — M2 data-quality layer added on top of the v01 geometry
pipeline (v01 kept unchanged as OpenScan2BIM_Step01b_Primitives.py):

  * Mixed-pixel / edge-artifact filter (Tang & Akinci-style): points are
    re-projected onto a spherical range grid around the scanner origin; a
    point is a MIXED-PIXEL candidate when its 3×3 angular neighbourhood
    contains surfaces BOTH clearly nearer and clearly farther than the
    point itself (phantom returns interpolated across a depth edge lie
    between foreground and background — legitimate edge points see only
    one side).  The margin grows with range so grazing surfaces with a
    legitimate in-plane range gradient are not flagged.
  * Isolated-outlier filter (deterministic SOR stand-in): voxels with few
    points AND no occupied neighbour voxel are flagged.
  * Per-point quality weight w ∈ [0,1] — standard TLS error model:
        w = w_range(r) × w_incidence(cos φ)
    with w_range = 1/(1+(r/RANGE_HALF)²) and w_incidence = |n·ray|
    (voxel normal vs ray direction).  Flagged points get w = 0.  E57
    intensity is NOT used in v02 (the Step-01 PLY does not carry it);
    the weight model is extendable when intensity is exported.
  * Glass ghost candidates: using the per-point semantic class from the
    Step-01 labeled PLY (window = 4), the spherical grid cells covered by
    window points define the glass surface range; points in those cells
    lying > GLASS_BEHIND beyond the glass surface are flagged as ghost
    CANDIDATES only — single-station evidence cannot separate through-
    glass geometry from reflection ghosts; final adjudication is the
    free-space counter-evidence test in Step 04 (M3).  Ghost candidates
    stay in the geometry stage but are accounted per patch (a patch with
    a high ghost fraction is itself a ghost-surface candidate).
  * Per-patch coverage accounting: coverage_ratio = occupied in-plane
    cell area / convex-hull extent area, plus per-patch mean weight and
    ghost fraction — "incompleteness becomes a quantified reliability
    attribute" for the fusion (M3) and decision (M5) layers.
  * Scanner origin: read from the E57 header (translation of scan 0,
    vendor-registered world frame — matches Step-01's transform=True),
    with --sensor_pos x,y,z as manual override.
  * Filtered points are EXCLUDED from plane peeling (cleaner normals and
    plane bands); --no_filter reproduces v01 geometry behaviour for the
    EXP-C ablation.  patch_ids.npy stays aligned with the input PLY
    (excluded points get patch id 0).

New outputs (in addition to the v01 set):
  <base>_point_quality.npz   per input point: w_uint8 (0–255 = 0–1),
                             flags uint8 (1=mixed, 2=isolated, 4=ghost_cand),
                             plus sensor_pos + params for provenance
  patches.csv gains: coverage_ratio, hull_area_m2, mean_w, ghost_frac
  summary gains: quality section (flag counts, weight stats, per-class
                 coverage) for the EXP-C table

v03 (2026-09-14): atomic-patch mode for the Step01c local-surface experiment.
The legacy <=0.35 m blind coplanar merge is disabled: Step01b emits only
3D-connected planar fragments, and every cross-gap association is deferred to
Step01c's delta_perp / g_parallel / ray-evidence procedure.  The patch CSV
adds factorized geometric attributes; legacy_geom_state is retained only for
backward traceability and existing GT diagnostics.

v04 (2026-09-14): preserves v03 geometry and output schema; configures
UTF-8 stdout so the summary is reproducible in Windows shells using GBK.

v01 (2026-07-08): deterministic dominant-plane peeling (Hough-style normal
binning → offset peak → global-plane inliers → 3D connected components;
leak-free by construction), station-level coplanar merge, local z-context
classification (multi-level ceilings expected), cylinder detection on the
residue (columns with measured tilt / duct runs).  Same input ⇒ identical
output (SHA1 logged).

Usage
-----
  python OpenScan2BIM_Step01b_Primitives02.py \\
      --ply ./outputs25/2ndLab-7_pano_labeled.ply \\
      --e57 ./assets/2ndLab-7.e57 \\
      --output_dir ./outputs26 \\
      --gt_ply <GT.ply>
"""

# ── Standard library ──────────────────────────────────────────────────────────
import os
import sys
import time
import hashlib
import argparse
import io
from typing import Dict, List, Optional, Tuple

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                                  errors="replace", line_buffering=True)

# ── Third-party ───────────────────────────────────────────────────────────────
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.ndimage import minimum_filter, maximum_filter
from plyfile import PlyData
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ══════════════════════════════════════════════════════════════════════════════
# Parameters
# ══════════════════════════════════════════════════════════════════════════════

VOXEL             = 0.02          # m — voxel size (surface resolution)
# Dominant-plane peeling
NBIN              = 0.15          # normal-space quantization step (~8.6°)
OFF_BIN           = 0.025         # m — plane-offset histogram bin
BAND_PLANE        = 0.035         # m — candidate band around offset peak
BAND_INLIER       = 0.030         # m — final inlier band after refit
COS_ALIGN         = np.cos(np.deg2rad(20.0))  # voxel-normal vs plane-normal
MIN_PLANE_SUPPORT = 800           # voxels (≈0.32 m²) — min peak support
MAX_PLANES        = 80            # peeling iterations upper bound
THETA_MERGE_DEG   = 5.0           # ° — coplanar patch-merge normal tolerance
BAND_MERGE        = 0.025         # m — mutual centroid-to-plane offset for merge
GAP_MERGE         = 0.35          # m — max bbox gap for coplanar merge (bridges
                                  #      soffit fragments split by ducts)
MIN_VOX_COMP      = 25            # voxels — drop micro-components before merge
MIN_AREA_WALL     = 0.30          # m² — min area for wall / slanted patches
MIN_AREA_HORIZ    = 0.10          # m² — min area for floor/ceiling fragments
WALL_MIN_ZSPAN    = 1.20          # m — vertical extent required for a wall
VERT_MAX_NZ       = np.sin(np.deg2rad(25.0))   # |n_z| ≤ → vertical candidate
HORIZ_MIN_NZ      = np.cos(np.deg2rad(15.0))   # |n_z| ≥ → horizontal candidate
COLUMN_DZ         = 0.15          # m — dead band for the above/below column test
COLUMN_DOMINANCE  = 2.0           #× — required dominance of one side
CEIL_LEVEL_TOL    = 0.06          # m — cluster tolerance for ceiling levels

# Cylinder detection on the peeling residue (M1b: columns / ducts)
MIN_VOX_CYL       = 200           # voxels — min cluster size to attempt a fit
CYL_R_MIN         = 0.04          # m — radius acceptance range
CYL_R_MAX         = 0.60
CYL_RMS_MAX       = 0.030         # m — radial RMS acceptance
CYL_COV_MIN_DEG   = 90.0          # ° — min angular coverage around the axis
CYL_LEN_MIN       = 0.50          # m — min extent along the axis
CYL_AXIS_EVR_MAX  = 0.05          # max λ0/Σλ of the normal covariance
COLUMN_MAX_TILT   = 30.0          # ° from vertical → column
PIPE_MIN_TILT     = 60.0          # ° from vertical → pipe/duct run

# ── M2 data-quality layer (v02) ───────────────────────────────────────────────
GRID_H            = 2048          # spherical range-grid height (W = 2H);
                                  #   0.088°/px — artifact detection resolution
MP_DELTA_ABS      = 0.15          # m — base near/far existence margin
MP_DELTA_REL      = 0.02          # ×r — margin growth with range (grazing
                                  #   surfaces have legitimate in-pixel range
                                  #   gradients; scaling prevents false flags)
ISO_MAX_PTS       = 5             # voxel points ≤ AND …
ISO_MAX_NEIGH     = 1             # … occupied 26-neighbours ≤ → isolated
RANGE_HALF        = 20.0          # m — range at which w_range = 0.5
GLASS_BEHIND      = 0.30          # m — beyond glass surface → ghost candidate
FLAG_MIXED        = 1             # bit flags in <base>_point_quality.npz
FLAG_ISOLATED     = 2
FLAG_GHOST_CAND   = 4

SEM_ID_MAP = {"ceiling": 1, "wall": 2, "door": 3, "window": 4, "floor": 5,
              "other": 0}
SEM_WINDOW = 4

GT_FIELDS = ["semantic_class", "scalar_semantic_class", "semantic",
             "label", "class_id"]

_BITS = 21                        # bits per packed voxel index (extent ≤ 2^21)


def log(msg: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


# ══════════════════════════════════════════════════════════════════════════════
# I/O
# ══════════════════════════════════════════════════════════════════════════════

def load_ply_xyz_sem(path: str) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Points (N,3) + per-point semantic class if the PLY carries one."""
    ply  = PlyData.read(path)
    data = ply["vertex"].data
    pts  = np.stack([np.asarray(data["x"]),
                     np.asarray(data["y"]),
                     np.asarray(data["z"])], axis=-1).astype(np.float64)
    sem = None
    for field in GT_FIELDS:
        if field in data.dtype.names:
            sem = np.asarray(data[field]).astype(np.int32)
            break
    return pts, sem


def load_gt_ply(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """GT points + labels.  NOTE: the GT file's point ORDER generally does
    NOT match the current Step 01 output (it was annotated on an older run),
    so labels must be transferred by coordinates, never by index."""
    ply  = PlyData.read(path)
    data = ply["vertex"].data
    pts  = np.stack([np.asarray(data["x"]),
                     np.asarray(data["y"]),
                     np.asarray(data["z"])], axis=-1).astype(np.float64)
    for field in GT_FIELDS:
        if field in data.dtype.names:
            return pts, np.asarray(data[field]).astype(np.int32)
    raise RuntimeError(f"No semantic field in {path}; expected one of {GT_FIELDS}")


def read_sensor_position(e57_path: str) -> np.ndarray:
    """Scanner origin in the vendor-registered world frame (E57 header
    translation of scan 0 — same frame Step 01 uses via transform=True)."""
    import pye57                                    # lazy: only needed here
    e57 = pye57.E57(e57_path)
    hdr = e57.get_header(0)
    return np.asarray(hdr.translation, dtype=np.float64)


# ══════════════════════════════════════════════════════════════════════════════
# M2 quality layer — spherical grid, artifact flags, weights, glass candidates
# ══════════════════════════════════════════════════════════════════════════════

def spherical_pixels(rel: np.ndarray, rng: np.ndarray,
                     H: int) -> np.ndarray:
    """Pixel index (int32, v*W+u) of each point on an H×2H spherical grid
    centred on the scanner (self-consistent mapping; independent of the
    Step-01 panorama convention)."""
    W = 2 * H
    az = np.arctan2(rel[:, 1], rel[:, 0])                    # [-π, π)
    el = np.arcsin(np.clip(rel[:, 2] / rng, -1.0, 1.0))      # [-π/2, π/2]
    u = np.minimum((W * (az + np.pi) / (2 * np.pi)).astype(np.int64), W - 1)
    v = np.minimum((H * (0.5 - el / np.pi)).astype(np.int64), H - 1)
    return (v * W + u).astype(np.int64)


def _grid_reduce(pix: np.ndarray, val: np.ndarray, npix: int,
                 mode: str) -> np.ndarray:
    """Per-pixel min or max of val (sort + reduceat; fast for 60M points)."""
    order = np.argsort(pix, kind="stable")
    ps, vs = pix[order], val[order]
    starts = np.flatnonzero(np.r_[True, np.diff(ps) > 0])
    red = np.minimum.reduceat(vs, starts) if mode == "min" \
        else np.maximum.reduceat(vs, starts)
    grid = np.full(npix, np.inf if mode == "min" else -np.inf, np.float32)
    grid[ps[starts]] = red
    return grid


def _nbr_filter(grid: np.ndarray, H: int, mode: str) -> np.ndarray:
    """3×3 neighbourhood min/max with azimuth wrap (pad columns, crop)."""
    W = 2 * H
    img = grid.reshape(H, W)
    img = np.concatenate([img[:, -1:], img, img[:, :1]], axis=1)
    f = minimum_filter if mode == "min" else maximum_filter
    out = f(img, size=3, mode="nearest")
    return out[:, 1:-1].reshape(-1)


def quality_layer(points: np.ndarray, sem: Optional[np.ndarray],
                  sensor: np.ndarray, log_fn) -> Tuple[np.ndarray, np.ndarray,
                                                       np.ndarray]:
    """
    Returns (flags uint8, rng float32, pix int64).
    Flags: FLAG_MIXED — depth-edge phantom candidates (nearer AND farther
    surfaces both present in the 3×3 angular neighbourhood);
    FLAG_GHOST_CAND — beyond the glass surface in window-covered cells.
    (FLAG_ISOLATED is added later from voxel occupancy.)
    """
    N = len(points)
    H, W = GRID_H, 2 * GRID_H
    npix = H * W
    rel = points - sensor
    rng = np.linalg.norm(rel, axis=1).astype(np.float32)
    rng = np.maximum(rng, 1e-6)
    pix = spherical_pixels(rel, rng, H)
    flags = np.zeros(N, np.uint8)

    # mixed-pixel candidates
    dmin = _grid_reduce(pix, rng, npix, "min")
    dmax = _grid_reduce(pix, rng, npix, "max")
    dmin_n = _nbr_filter(dmin, H, "min")
    dmax_n = _nbr_filter(dmax, H, "max")
    delta = MP_DELTA_ABS + MP_DELTA_REL * rng
    mixed = (dmin_n[pix] < rng - delta) & (dmax_n[pix] > rng + delta)
    flags[mixed] |= FLAG_MIXED
    log_fn(f"  quality: mixed-pixel candidates {int(mixed.sum()):,} "
           f"({mixed.mean() * 100:.2f}%)")

    # glass ghost candidates (needs the Step-01 per-point semantics)
    if sem is not None and np.any(sem == SEM_WINDOW):
        wsel = sem == SEM_WINDOW
        gmin = _grid_reduce(pix[wsel], rng[wsel], npix, "min")
        gmin_n = _nbr_filter(gmin, H, "min")           # dilate glass cells by 1
        ghost = np.isfinite(gmin_n[pix]) & (rng > gmin_n[pix] + GLASS_BEHIND)
        flags[ghost] |= FLAG_GHOST_CAND
        log_fn(f"  quality: window points {int(wsel.sum()):,} → glass ghost "
               f"candidates {int(ghost.sum()):,} ({ghost.mean() * 100:.2f}%)")
    else:
        log_fn("  quality: no window semantics in PLY → glass candidates skipped")

    return flags, rng, pix


# ══════════════════════════════════════════════════════════════════════════════
# Voxel hashing + moments
# ══════════════════════════════════════════════════════════════════════════════

def voxelize(points: np.ndarray, voxel: float):
    """
    Hash points into voxels. Returns (uniq_keys sorted, inverse, ijk_origin).
    Packed key = ix<<42 | iy<<21 | iz; origin shifted so all indices ≥ 1
    (±1 neighbour arithmetic can never borrow across bit fields).
    """
    origin = points.min(axis=0) - 1.5 * voxel
    ijk    = np.floor((points - origin) / voxel).astype(np.int64)
    span   = ijk.max(axis=0)
    if np.any(span >= (1 << _BITS) - 2):
        raise ValueError(f"Scene too large for {_BITS}-bit voxel packing: span={span}")
    keys = (ijk[:, 0] << (2 * _BITS)) | (ijk[:, 1] << _BITS) | ijk[:, 2]
    uniq, inverse = np.unique(keys, return_inverse=True)
    return uniq, inverse, origin


def per_voxel_moments(points: np.ndarray, inverse: np.ndarray, V: int):
    """Per-voxel count, coordinate sums and raw second moments (6 unique)."""
    n = np.bincount(inverse, minlength=V).astype(np.float64)
    S = np.empty((V, 3), np.float64)
    for a in range(3):
        S[:, a] = np.bincount(inverse, weights=points[:, a], minlength=V)
    # xx, yy, zz, xy, xz, yz
    M = np.empty((V, 6), np.float64)
    pairs = [(0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2)]
    for m, (a, b) in enumerate(pairs):
        M[:, m] = np.bincount(inverse, weights=points[:, a] * points[:, b],
                              minlength=V)
    return n, S, M


def neighbour_offsets(include_zero: bool):
    offs = []
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dz in (-1, 0, 1):
                if dx == dy == dz == 0 and not include_zero:
                    continue
                offs.append((dx, dy, dz))
    return offs


def _off_key(off) -> int:
    # Arithmetic (NOT bitwise) delta: packed fields never borrow for ±1
    # offsets because voxelize() shifts all indices to ≥ 1.  A bitwise OR
    # would corrupt keys for negative components (e.g. (0,1,-1)).
    dx, dy, dz = off
    return dx * (1 << (2 * _BITS)) + dy * (1 << _BITS) + dz


def neighbour_lookup(uniq: np.ndarray, off) -> Tuple[np.ndarray, np.ndarray]:
    """Indices (src, nbr) of voxel pairs where voxel src has neighbour at off."""
    nk    = uniq + _off_key(off)
    pos   = np.searchsorted(uniq, nk)
    pos_c = np.minimum(pos, len(uniq) - 1)
    valid = uniq[pos_c] == nk
    src   = np.nonzero(valid)[0]
    return src, pos_c[valid]


def isolated_voxel_flags(points: np.ndarray, voxel: float) -> np.ndarray:
    """FLAG_ISOLATED per point: voxel has ≤ ISO_MAX_PTS points and ≤
    ISO_MAX_NEIGH occupied 26-neighbours (deterministic SOR stand-in)."""
    uniq, inverse, _ = voxelize(points, voxel)
    V = len(uniq)
    n = np.bincount(inverse, minlength=V)
    nbrs = np.zeros(V, np.int32)
    for off in neighbour_offsets(include_zero=False):
        src, _nb = neighbour_lookup(uniq, off)
        nbrs[src] += 1
    iso_vox = (n <= ISO_MAX_PTS) & (nbrs <= ISO_MAX_NEIGH)
    return iso_vox[inverse]


def aggregated_normals(uniq, n, S, M):
    """
    Covariance over each voxel's 3×3×3 neighbourhood (6 cm support) →
    batched eigh → unit normals (smallest eigenvector) + centroids.
    """
    an, aS, aM = n.copy(), S.copy(), M.copy()
    for off in neighbour_offsets(include_zero=False):
        src, nbr = neighbour_lookup(uniq, off)
        an[src] += n[nbr]
        aS[src] += S[nbr]
        aM[src] += M[nbr]

    mu  = aS / an[:, None]
    cov = np.empty((len(uniq), 3, 3), np.float64)
    cov[:, 0, 0] = aM[:, 0] / an - mu[:, 0] * mu[:, 0]
    cov[:, 1, 1] = aM[:, 1] / an - mu[:, 1] * mu[:, 1]
    cov[:, 2, 2] = aM[:, 2] / an - mu[:, 2] * mu[:, 2]
    cov[:, 0, 1] = cov[:, 1, 0] = aM[:, 3] / an - mu[:, 0] * mu[:, 1]
    cov[:, 0, 2] = cov[:, 2, 0] = aM[:, 4] / an - mu[:, 0] * mu[:, 2]
    cov[:, 1, 2] = cov[:, 2, 1] = aM[:, 5] / an - mu[:, 1] * mu[:, 2]

    w, v = np.linalg.eigh(cov)                    # ascending eigenvalues
    normals   = v[:, :, 0]                        # smallest-λ eigenvector
    centroids = S / n[:, None]                    # own-voxel centroid
    return normals, centroids


# ══════════════════════════════════════════════════════════════════════════════
# Region growing + coplanar merge
# ══════════════════════════════════════════════════════════════════════════════

def adjacency_pairs(uniq) -> Tuple[np.ndarray, np.ndarray]:
    """All undirected 26-neighbour voxel pairs (each pair once), precomputed."""
    rows, cols = [], []
    for off in neighbour_offsets(include_zero=False):
        if _off_key(off) <= 0:                    # positive half-space only
            continue
        src, nbr = neighbour_lookup(uniq, off)
        if len(src):
            rows.append(src)
            cols.append(nbr)
    r = np.concatenate(rows) if rows else np.empty(0, np.int64)
    c = np.concatenate(cols) if cols else np.empty(0, np.int64)
    return r, c


def _components_of_subset(inlier_idx: np.ndarray, pair_r, pair_c, V: int):
    """Connected components restricted to a voxel subset. Returns groups."""
    inmask = np.zeros(V, bool)
    inmask[inlier_idx] = True
    sel = inmask[pair_r] & inmask[pair_c]
    lut = np.full(V, -1, np.int64)
    lut[inlier_idx] = np.arange(len(inlier_idx))
    r, c = lut[pair_r[sel]], lut[pair_c[sel]]
    graph = coo_matrix((np.ones(len(r), np.int8), (r, c)),
                       shape=(len(inlier_idx), len(inlier_idx)))
    n_comp, comp = connected_components(graph, directed=False)
    order = np.argsort(comp, kind="stable")
    comp_sorted = comp[order]
    boundaries = np.nonzero(np.diff(comp_sorted))[0] + 1
    chunks = np.split(order, boundaries)
    return [inlier_idx[ch] for ch in chunks]


def _canonical(normals: np.ndarray) -> np.ndarray:
    """Antipodal-invariant normal representative (sign-canonicalized copy)."""
    can = normals.copy()
    flip = (can[:, 2] < 0) \
        | ((np.abs(can[:, 2]) < 1e-6) & (can[:, 1] < 0)) \
        | ((np.abs(can[:, 2]) < 1e-6) & (np.abs(can[:, 1]) < 1e-6) & (can[:, 0] < 0))
    can[flip] *= -1.0
    return can


def peel_planes(uniq, normals, centroids, log_fn) -> Tuple[List[np.ndarray], np.ndarray]:
    """
    Deterministic Hough-style dominant-plane peeling.

    Repeatedly: pick the most populated normal-direction bin → find the
    strongest plane-offset peak along that direction → take GLOBAL plane
    inliers (normal-aligned, ≤ BAND_INLIER from the refit plane) → split
    them into 3D-connected patches → remove from the pool.  Membership is
    anchored to a global plane, so smoothness leaks across rounded corners
    cannot happen; curved surfaces never peak and remain in the residue
    (input for the M1b cylinder stage).  Fully vectorized, no sampling —
    same input ⇒ same patches.
    """
    V = len(uniq)
    can = _canonical(normals)
    pair_r, pair_c = adjacency_pairs(uniq)
    remaining = np.ones(V, bool)
    groups: List[np.ndarray] = []
    n_planes = 0

    for it in range(MAX_PLANES):
        idx = np.nonzero(remaining)[0]
        if len(idx) < MIN_PLANE_SUPPORT:
            break
        # 1 — dominant normal direction (quantized canonical normals)
        q  = np.round(can[idx] / NBIN).astype(np.int64) + 16
        qk = (q[:, 0] * 33 + q[:, 1]) * 33 + q[:, 2]
        vals, cnts = np.unique(qk, return_counts=True)
        seed_key = vals[int(np.argmax(cnts))]
        seed = idx[qk == seed_key]
        n0 = can[seed].mean(axis=0)
        n0 /= np.linalg.norm(n0)

        # 2 — offset histogram along n0 among normal-aligned voxels
        aligned = idx[np.abs(can[idx] @ n0) >= COS_ALIGN]
        if len(aligned) < MIN_PLANE_SUPPORT:
            remaining[seed] = False               # exhausted direction bin
            continue
        s  = centroids[aligned] @ n0
        hb = np.floor(s / OFF_BIN).astype(np.int64)
        hv, hc = np.unique(hb, return_counts=True)
        peak = hv[int(np.argmax(hc))]
        in_peak = (hb >= peak - 1) & (hb <= peak + 1)
        if in_peak.sum() < MIN_PLANE_SUPPORT:
            remaining[seed] = False
            continue
        s0   = float(np.median(s[in_peak]))
        cand = aligned[np.abs(s - s0) <= BAND_PLANE]

        # 3 — refit plane on candidates, final global-plane inliers
        mu  = centroids[cand].mean(axis=0)
        dc  = centroids[cand] - mu
        cov = dc.T @ dc / len(cand)
        w, v = np.linalg.eigh(cov)
        n_ref = v[:, 0]
        d_ref = -float(n_ref @ mu)
        inl = idx[(np.abs(can[idx] @ n_ref) >= COS_ALIGN)
                  & (np.abs(centroids[idx] @ n_ref + d_ref) <= BAND_INLIER)]
        if len(inl) < MIN_PLANE_SUPPORT:
            remaining[seed] = False
            continue

        # 4 — 3D-connected patches on the inlier set; consume all inliers
        for g in _components_of_subset(inl, pair_r, pair_c, V):
            if len(g) >= MIN_VOX_COMP:
                groups.append(g)
        remaining[inl] = False
        n_planes += 1

    log_fn(f"  plane peeling: {n_planes} planes, {len(groups)} connected patches, "
           f"residue {int(remaining.sum()):,} voxels (→ cylinders)")
    return groups, remaining


class UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))

    def find(self, a: int) -> int:
        while self.parent[a] != a:
            self.parent[a] = self.parent[self.parent[a]]
            a = self.parent[a]
        return a

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)


def fit_planes(idx_groups: List[np.ndarray], n, S, M):
    """Plane fit per group from aggregated voxel moments. Returns dict arrays."""
    G = len(idx_groups)
    normal  = np.zeros((G, 3))
    center  = np.zeros((G, 3))
    rms     = np.zeros(G)
    npts    = np.zeros(G)
    for g, vox in enumerate(idx_groups):
        gn = n[vox].sum()
        gS = S[vox].sum(axis=0)
        gM = M[vox].sum(axis=0)
        mu = gS / gn
        cov = np.array([
            [gM[0] / gn - mu[0] * mu[0], gM[3] / gn - mu[0] * mu[1], gM[4] / gn - mu[0] * mu[2]],
            [gM[3] / gn - mu[0] * mu[1], gM[1] / gn - mu[1] * mu[1], gM[5] / gn - mu[1] * mu[2]],
            [gM[4] / gn - mu[0] * mu[2], gM[5] / gn - mu[1] * mu[2], gM[2] / gn - mu[2] * mu[2]],
        ])
        w, v = np.linalg.eigh(cov)
        normal[g] = v[:, 0]
        center[g] = mu
        rms[g]    = float(np.sqrt(max(w[0], 0.0)))
        npts[g]   = gn
    d = -np.einsum("ij,ij->i", normal, center)
    return normal, d, center, rms, npts


def coplanar_merge(groups: List[np.ndarray], normal, d, center, bbox_min, bbox_max):
    """
    Merge patch fragments lying on the same physical plane (bridges soffit
    pieces separated by ducts).  Same criteria family as the cross-station
    association planned for Step 04.
    """
    G = len(groups)
    cos_merge = np.cos(np.deg2rad(THETA_MERGE_DEG))
    uf = UnionFind(G)
    for i in range(G):
        ndot = np.abs(normal[i + 1:] @ normal[i])
        cand = np.nonzero(ndot >= cos_merge)[0] + i + 1
        if len(cand) == 0:
            continue
        # mutual centroid-to-plane offsets
        off_i = np.abs(center[cand] @ normal[i] + d[i])
        off_j = np.abs(center[i] @ (normal[cand].T) + d[cand] * 1.0)
        ok = (off_i <= BAND_MERGE) & (off_j <= BAND_MERGE)
        # bbox proximity (per-axis interval gap)
        gap = np.maximum(0.0,
              np.maximum(bbox_min[cand] - bbox_max[i],
                         bbox_min[i] - bbox_max[cand])).max(axis=1)
        ok &= gap <= GAP_MERGE
        for j in cand[ok]:
            uf.union(i, int(j))

    root = np.array([uf.find(g) for g in range(G)])
    merged: Dict[int, List[int]] = {}
    for g, r in enumerate(root):
        merged.setdefault(int(r), []).append(g)
    out = [np.concatenate([groups[g] for g in members])
           for r, members in sorted(merged.items())]
    counts = [len(members) for r, members in sorted(merged.items())]
    return out, counts


# ══════════════════════════════════════════════════════════════════════════════
# Cylinder detection on the peeling residue (M1b — columns / ducts)
# ══════════════════════════════════════════════════════════════════════════════

def detect_cylinders(uniq, normals, centroids, residue_mask) -> List[Dict]:
    """
    Deterministic cylinder detection on the plane-peeling residue.

    A cylinder's surface normals all lie in the plane perpendicular to its
    axis, so the axis is the smallest-eigenvalue eigenvector of the normal
    covariance (antipodal-safe: outer products cancel the sign ambiguity).
    Radius/center come from an algebraic Kåsa circle fit in the axis-normal
    plane.  No sampling — same input ⇒ same cylinders.
    """
    V = len(uniq)
    idx_res = np.nonzero(residue_mask)[0]
    if len(idx_res) < MIN_VOX_CYL:
        return []
    pair_r, pair_c = adjacency_pairs(uniq)
    clusters = [c for c in _components_of_subset(idx_res, pair_r, pair_c, V)
                if len(c) >= MIN_VOX_CYL]

    out: List[Dict] = []
    for cl in sorted(clusters, key=len, reverse=True):
        nn = normals[cl]
        C  = nn.T @ nn / len(cl)
        w, v = np.linalg.eigh(C)
        if w[0] / max(w.sum(), 1e-12) > CYL_AXIS_EVR_MAX:
            continue                              # normals not coplanar
        axis = v[:, 0]
        if axis[2] < 0:
            axis = -axis
        c0 = centroids[cl].mean(axis=0)
        d  = centroids[cl] - c0
        ref = np.array([0.0, 0.0, 1.0]) if abs(axis[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
        e1 = np.cross(axis, ref); e1 /= np.linalg.norm(e1)
        e2 = np.cross(axis, e1)
        u, t = d @ e1, d @ e2
        A = np.column_stack([2.0 * u, 2.0 * t, np.ones(len(u))])
        b = u * u + t * t
        try:
            sol, *_ = np.linalg.lstsq(A, b, rcond=None)
        except np.linalg.LinAlgError:
            continue
        cu, ct, cc = sol
        r2 = cc + cu * cu + ct * ct
        if r2 <= 0:
            continue
        radius = float(np.sqrt(r2))
        if not (CYL_R_MIN <= radius <= CYL_R_MAX):
            continue
        rad_d  = np.sqrt((u - cu) ** 2 + (t - ct) ** 2)
        rms    = float(np.sqrt(np.mean((rad_d - radius) ** 2)))
        if rms > CYL_RMS_MAX:
            continue
        ang    = np.arctan2(t - ct, u - cu)
        cov_deg = float(len(np.unique(np.floor(np.rad2deg(ang) / 15.0))) * 15.0)
        if cov_deg < CYL_COV_MIN_DEG:
            continue
        along  = d @ axis
        length = float(np.percentile(along, 98) - np.percentile(along, 2))
        if length < CYL_LEN_MIN:
            continue
        tilt = float(np.rad2deg(np.arccos(np.clip(abs(axis[2]), 0.0, 1.0))))
        if tilt <= COLUMN_MAX_TILT:
            ctype = "column"
        elif tilt >= PIPE_MIN_TILT:
            ctype = "pipe_duct"
        else:
            ctype = "slanted_cylinder"
        center = c0 + (cu * e1 + ct * e2)
        out.append({
            "cyl_id":    len(out) + 1,
            "type":      ctype,
            "n_voxels":  int(len(cl)),
            "radius_m":  round(radius, 4),
            "length_m":  round(length, 3),
            "tilt_deg":  round(tilt, 2),
            "axis_x": round(float(axis[0]), 5),
            "axis_y": round(float(axis[1]), 5),
            "axis_z": round(float(axis[2]), 5),
            "center_x": round(float(center[0]), 4),
            "center_y": round(float(center[1]), 4),
            "center_z": round(float(center[2]), 4),
            "rms_m":     round(rms, 4),
            "coverage_deg": round(cov_deg, 0),
        })
    return out


# ══════════════════════════════════════════════════════════════════════════════
# Patch classification (local context, no global height constants)
# ══════════════════════════════════════════════════════════════════════════════

def in_plane_axes(nrm: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    ref = np.array([0.0, 0.0, 1.0]) if abs(nrm[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    e1  = np.cross(nrm, ref); e1 /= np.linalg.norm(e1)
    e2  = np.cross(nrm, e1)
    return e1, e2


def in_plane_area(vox_centers: np.ndarray, nrm: np.ndarray, voxel: float) -> float:
    """Occupancy area: project voxel centers onto the plane, count 2D cells."""
    e1, e2 = in_plane_axes(nrm)
    a = np.floor(vox_centers @ e1 / voxel).astype(np.int64)
    b = np.floor(vox_centers @ e2 / voxel).astype(np.int64)
    return float(len(np.unique(a * (1 << 32) + b))) * voxel * voxel


def hull_extent_area(vox_centers: np.ndarray, nrm: np.ndarray) -> float:
    """Convex-hull area of the in-plane 2D projection (patch plane extent);
    coverage_ratio = occupancy area / this.  Falls back to the 2D bbox."""
    e1, e2 = in_plane_axes(nrm)
    uv = np.column_stack([vox_centers @ e1, vox_centers @ e2])
    try:
        from scipy.spatial import ConvexHull
        if len(uv) >= 8:
            return float(ConvexHull(uv).volume)     # 2-D hull: volume == area
    except Exception:
        pass
    span = uv.max(axis=0) - uv.min(axis=0)
    return float(span[0] * span[1])


def column_fractions(centroids_all: np.ndarray, cell_sorted_idx, cell_indptr,
                     cell_ids_sorted_uniq, patch_vox_centers: np.ndarray,
                     z_ref: float, cell2d: float) -> Tuple[float, float]:
    """
    For a horizontal patch: among ALL voxels whose xy-cell overlaps the patch
    footprint, the fraction lying clearly above / below the patch plane.
      floor-like    → occupied space above, (almost) nothing below
      ceiling-like  → occupied space below, (almost) nothing above
      mid furniture → both sides occupied
    """
    ca = np.floor(patch_vox_centers[:, 0] / cell2d).astype(np.int64)
    cb = np.floor(patch_vox_centers[:, 1] / cell2d).astype(np.int64)
    cells = np.unique(ca * (1 << 32) + cb)
    pos   = np.searchsorted(cell_ids_sorted_uniq, cells)
    pos   = np.minimum(pos, len(cell_ids_sorted_uniq) - 1)
    cells = cells[cell_ids_sorted_uniq[pos] == cells]
    pos   = np.searchsorted(cell_ids_sorted_uniq, cells)
    if len(cells) == 0:
        return 0.0, 0.0
    idx = np.concatenate([cell_sorted_idx[cell_indptr[p]:cell_indptr[p + 1]]
                          for p in pos])
    dz  = centroids_all[idx, 2] - z_ref
    above = float(np.mean(dz >  COLUMN_DZ))
    below = float(np.mean(dz < -COLUMN_DZ))
    return above, below


def classify_patch(nrm, z_vals, area, above, below) -> Tuple[str, float, float]:
    """Return (guess, wall_tilt_deg, slab_tilt_deg)."""
    nz = abs(float(nrm[2]))
    wall_tilt = float(np.rad2deg(np.arcsin(min(nz, 1.0))))          # 0° = plumb
    slab_tilt = float(np.rad2deg(np.arccos(min(nz, 1.0))))          # 0° = level
    zspan = float(np.percentile(z_vals, 98) - np.percentile(z_vals, 2))

    if nz >= HORIZ_MIN_NZ:
        if area < MIN_AREA_HORIZ:
            return "clutter", wall_tilt, slab_tilt
        if below <= 0.02 and above >= 0.05:
            return "floor", wall_tilt, slab_tilt
        if above <= 0.02 and below >= 0.05:
            return "ceiling_level", wall_tilt, slab_tilt
        if above >= COLUMN_DOMINANCE * max(below, 1e-6):
            return "floor", wall_tilt, slab_tilt
        if below >= COLUMN_DOMINANCE * max(above, 1e-6):
            return "ceiling_level", wall_tilt, slab_tilt
        return "horizontal_mid", wall_tilt, slab_tilt               # tables etc.

    if nz <= VERT_MAX_NZ:
        if zspan >= WALL_MIN_ZSPAN and area >= MIN_AREA_WALL:
            return "wall", wall_tilt, slab_tilt
        return "vertical_clutter", wall_tilt, slab_tilt

    if area >= MIN_AREA_WALL:
        return "slanted", wall_tilt, slab_tilt                      # tilted facade?
    return "clutter", wall_tilt, slab_tilt


def factorized_planar_attributes(nrm, legacy_geom_state):
    """Describe directly measured geometry without assigning a BIM class."""
    nz = abs(float(nrm[2]))
    if nz >= HORIZ_MIN_NZ:
        orientation = "horizontal"
        context = {
            "floor": "lower",
            "ceiling_level": "upper",
            "horizontal_mid": "mid",
        }.get(legacy_geom_state, "unknown")
    elif nz <= VERT_MAX_NZ:
        orientation, context = "vertical", "NA"
    else:
        orientation, context = "inclined", "NA"
    return {
        "primitive_family": "planar",
        "observation_extent": "fragment",
        "planar_orientation": orientation,
        "horizontal_context": context,
        "legacy_geom_state": legacy_geom_state,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="OpenScan2BIM Step 01b v04 — atomic geometry fragments "
                    "primitives + M2 data-quality layer")
    p.add_argument("--ply", default="./outputs25/2ndLab-7_pano_labeled.ply",
                   help="Step 01 labeled PLY (xyz + semantic_class)")
    p.add_argument("--e57", default=None,
                   help="Source E57 (scanner origin from header); default "
                        "./assets/<base>.e57")
    p.add_argument("--sensor_pos", default=None,
                   help="Manual scanner origin 'x,y,z' (overrides --e57)")
    p.add_argument("--output_dir", default="./outputs26")
    p.add_argument("--base", default=None,
                   help="Output base name; default derived from --ply filename")
    p.add_argument("--voxel", type=float, default=VOXEL)
    p.add_argument("--no_filter", action="store_true",
                   help="EXP-C ablation: compute flags/weights but do NOT "
                        "exclude flagged points from the geometry stage "
                        "(reproduces v01 geometry behaviour)")
    p.add_argument("--gt_ply", default=None,
                   help="Optional manual GT PLY for the EXP-B point-level "
                        "wall/floor/ceiling check (coordinate transfer)")
    p.add_argument("--save_patch_ply", action="store_true",
                   help="Also save voxel-center PLY colored by patch id")
    return p.parse_args()


def _save_npz_retry(path: str, **arrays) -> None:
    try:
        np.savez_compressed(path, **arrays)
    except OSError:
        if os.path.exists(path):                 # transient AV/indexer lock
            os.remove(path)
        time.sleep(1.0)
        np.savez_compressed(path, **arrays)


def main() -> None:
    args  = parse_args()
    voxel = args.voxel
    base  = args.base
    if base is None:
        base = os.path.basename(args.ply)
        for suffix in ("_pano_labeled.ply", "_labeled.ply", ".ply"):
            if base.endswith(suffix):
                base = base[: -len(suffix)]
                break
    os.makedirs(args.output_dir, exist_ok=True)
    out = lambda name: os.path.join(args.output_dir, f"{base}{name}")
    t0 = time.time()

    log(f"Step 01b v04 — atomic primitives + quality layer  (voxel={voxel} m)")
    log(f"Loading {args.ply} …")
    points, sem = load_ply_xyz_sem(args.ply)
    N = len(points)
    log(f"  {N:,} points   semantics: {'yes' if sem is not None else 'no'}")

    # ── Scanner origin ────────────────────────────────────────────────────────
    sensor = None
    if args.sensor_pos:
        sensor = np.array([float(x) for x in args.sensor_pos.split(",")],
                          dtype=np.float64)
    else:
        e57_path = args.e57 or os.path.join("./assets/Level02Lab", f"{base}.e57")
        if os.path.exists(e57_path):
            try:
                sensor = read_sensor_position(e57_path)
            except Exception as exc:
                log(f"[WARN] E57 header read failed ({exc}) — "
                    f"range/incidence/mixed-pixel/glass disabled")
        else:
            log(f"[WARN] no E57 at {e57_path} and no --sensor_pos — "
                f"range/incidence/mixed-pixel/glass disabled")
    if sensor is not None:
        log(f"  scanner origin: [{sensor[0]:.3f}, {sensor[1]:.3f}, {sensor[2]:.3f}]")

    # ── M2 quality flags (all input points) ───────────────────────────────────
    flags = np.zeros(N, np.uint8)
    rng = pix = None
    if sensor is not None:
        flags, rng, pix = quality_layer(points, sem, sensor, log)
    iso = isolated_voxel_flags(points, voxel)
    flags[iso] |= FLAG_ISOLATED
    log(f"  quality: isolated-voxel points {int(iso.sum()):,} "
        f"({iso.mean() * 100:.2f}%)")

    artifact = (flags & (FLAG_MIXED | FLAG_ISOLATED)) > 0
    if args.no_filter:
        clean_idx = np.arange(N)
        log(f"  filter DISABLED (--no_filter): geometry uses all {N:,} points")
    else:
        clean_idx = np.nonzero(~artifact)[0]
        log(f"  filter: {int(artifact.sum()):,} artifact points excluded "
            f"({artifact.mean() * 100:.2f}%) → geometry on {len(clean_idx):,}")
    cpts = points[clean_idx]

    # ── Voxelization + moments (clean subset) ─────────────────────────────────
    uniq, inverse, origin = voxelize(cpts, voxel)
    V = len(uniq)
    log(f"  {V:,} occupied voxels")
    n, S, M = per_voxel_moments(cpts, inverse, V)

    # ── Normals from 3×3×3 aggregated covariance ──────────────────────────────
    normals, centroids = aggregated_normals(uniq, n, S, M)
    log(f"  normals estimated (6 cm support, batched eigh)")

    # ── Per-point quality weight w = w_range × w_incidence ────────────────────
    w_point = np.zeros(N, np.float32)
    if sensor is not None:
        w_range = 1.0 / (1.0 + (rng[clean_idx] / RANGE_HALF) ** 2)
        ray = points[clean_idx] - sensor
        ray /= np.maximum(np.linalg.norm(ray, axis=1, keepdims=True), 1e-9)
        cosphi = np.abs(np.einsum("ij,ij->i", normals[inverse], ray))
        w_point[clean_idx] = (w_range * cosphi).astype(np.float32)
    else:
        w_point[clean_idx] = 1.0                  # no error model available
    if not args.no_filter:
        w_point[artifact] = 0.0

    # ── Deterministic dominant-plane peeling ──────────────────────────────────
    groups, residue_mask = peel_planes(uniq, normals, centroids, log)

    # ── Plane fit: emit atomic 3D-connected fragments ─────────────────────────
    nrm, d, ctr, rms, npts = fit_planes(groups, n, S, M)
    merged_counts = np.ones(len(groups), dtype=np.int32)
    log(f"  atomic mode → {len(groups):,} connected planar fragments "
        "(no blind cross-gap merge; Step01c owns all reassembly)")

    # ── 2D column index for the floor/ceiling context test ────────────────────
    CELL2D = 0.50
    ca = np.floor(centroids[:, 0] / CELL2D).astype(np.int64)
    cb = np.floor(centroids[:, 1] / CELL2D).astype(np.int64)
    cell_ids = ca * (1 << 32) + cb
    cell_order = np.argsort(cell_ids, kind="stable")
    cell_sorted = cell_ids[cell_order]
    cell_uniq, cell_start = np.unique(cell_sorted, return_index=True)
    cell_indptr = np.append(cell_start, len(cell_sorted))

    # ── Classify patches ──────────────────────────────────────────────────────
    rows: List[Dict] = []
    patch_of_group = np.zeros(len(groups), np.int32)
    areas = []
    for g, vox in enumerate(groups):
        vc   = centroids[vox]
        area = in_plane_area(vc, nrm[g], voxel)
        areas.append(area)
    order_by_area = np.argsort(-np.asarray(areas), kind="stable")

    pid = 0
    accepted_rows_group: List[int] = []
    for rank, g in enumerate(order_by_area):
        vox  = groups[g]
        vc   = centroids[vox]
        area = areas[g]
        above, below = (0.0, 0.0)
        nz = abs(float(nrm[g][2]))
        if nz >= HORIZ_MIN_NZ:
            above, below = column_fractions(
                centroids, cell_order, cell_indptr, cell_uniq,
                vc, float(np.median(vc[:, 2])), CELL2D)
        guess, wall_tilt, slab_tilt = classify_patch(nrm[g], vc[:, 2], area,
                                                     above, below)
        if guess in ("clutter",) and area < MIN_AREA_HORIZ:
            continue                                   # unassigned (patch id 0)
        pid += 1
        patch_of_group[g] = pid
        accepted_rows_group.append(g)
        hull = hull_extent_area(vc, nrm[g])
        rows.append({
            "patch_id":   pid,
            **factorized_planar_attributes(nrm[g], guess),
            "guess":      guess,  # legacy alias; not a final semantic class
            "n_points":   int(npts[g]),
            "n_voxels":   int(len(vox)),
            "area_m2":    round(area, 4),
            "hull_area_m2":   round(hull, 4),
            "coverage_ratio": round(min(area / max(hull, 1e-9), 1.0), 4),
            "nx": round(float(nrm[g][0]), 6),
            "ny": round(float(nrm[g][1]), 6),
            "nz": round(float(nrm[g][2]), 6),
            "d":  round(float(d[g]), 6),
            "cx": round(float(ctr[g][0]), 4),
            "cy": round(float(ctr[g][1]), 4),
            "cz": round(float(ctr[g][2]), 4),
            "z_mean":  round(float(vc[:, 2].mean()), 4),
            "z_p02":   round(float(np.percentile(vc[:, 2], 2)), 4),
            "z_p98":   round(float(np.percentile(vc[:, 2], 98)), 4),
            "rms_m":   round(float(rms[g]), 5),
            "wall_tilt_deg": round(wall_tilt, 2),
            "slab_tilt_deg": round(slab_tilt, 2),
            "frac_above": round(above, 3),
            "frac_below": round(below, 3),
            "merged_fragments": int(merged_counts[g]),
            "mean_w": 0.0,                       # filled below from points
            "ghost_frac": 0.0,
        })

    log(f"  {pid} patches accepted")

    # ── Per-point patch ids (aligned to the FULL input PLY) ───────────────────
    vox_patch = np.zeros(V, np.int32)
    for g, vox in enumerate(groups):
        if patch_of_group[g] > 0:
            vox_patch[vox] = patch_of_group[g]
    point_patch = np.zeros(N, np.int32)
    point_patch[clean_idx] = vox_patch[inverse]
    ids_path = out("_patch_ids.npy")
    try:
        np.save(ids_path, point_patch)
    except OSError:
        # Transient Windows lock on the previous run's file (AV/indexer scan
        # of the 240 MB npy) — remove and retry once.
        if os.path.exists(ids_path):
            os.remove(ids_path)
        time.sleep(1.0)
        np.save(ids_path, point_patch)

    # ── Per-patch quality stats from point-level w / ghost flags ──────────────
    ghost = (flags & FLAG_GHOST_CAND) > 0
    if pid > 0:
        w_sum  = np.bincount(point_patch, weights=w_point, minlength=pid + 1)
        p_cnt  = np.bincount(point_patch, minlength=pid + 1).astype(np.float64)
        g_cnt  = np.bincount(point_patch, weights=ghost.astype(np.float64),
                             minlength=pid + 1)
        for r in rows:
            k = r["patch_id"]
            if p_cnt[k] > 0:
                r["mean_w"]     = round(float(w_sum[k] / p_cnt[k]), 4)
                r["ghost_frac"] = round(float(g_cnt[k] / p_cnt[k]), 4)

    # ── Quality NPZ (per input point; provenance included) ────────────────────
    _save_npz_retry(
        out("_point_quality.npz"),
        w_uint8=np.clip(np.round(w_point * 255.0), 0, 255).astype(np.uint8),
        flags=flags,
        sensor_pos=(sensor if sensor is not None else np.full(3, np.nan)),
        params=np.array([GRID_H, MP_DELTA_ABS, MP_DELTA_REL, ISO_MAX_PTS,
                         ISO_MAX_NEIGH, RANGE_HALF, GLASS_BEHIND],
                        dtype=np.float64),
        filtered=np.array([0 if args.no_filter else 1], dtype=np.int8),
    )
    log(f"Point quality → {out('_point_quality.npz')}")

    # ── Patch table CSV (+ determinism hash) ──────────────────────────────────
    import csv as _csv
    csv_path = out("_patches.csv")
    fieldnames = list(rows[0].keys()) if rows else ["patch_id"]
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        w = _csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    sha1 = hashlib.sha1(open(csv_path, "rb").read()).hexdigest()[:16]
    log(f"Patch table → {csv_path}  (SHA1 {sha1})")

    # ── QC top-down ───────────────────────────────────────────────────────────
    try:
        stride = max(1, V // 300_000)
        sel = np.arange(0, V, stride)
        cmap = plt.get_cmap("tab20")
        sel_patch = vox_patch[sel]
        fig, ax = plt.subplots(figsize=(10, 10))
        bg = sel_patch == 0
        ax.scatter(centroids[sel][bg, 0], centroids[sel][bg, 1], s=0.3,
                   c="#dddddd", linewidths=0)
        fg = ~bg
        ax.scatter(centroids[sel][fg, 0], centroids[sel][fg, 1], s=0.3,
                   c=cmap((sel_patch[fg] % 20).astype(int)), linewidths=0)
        ax.set_aspect("equal"); ax.set_title(f"{base} — patches (top-down)")
        png_path = out("_patch_topdown.png")
        try:
            fig.savefig(png_path, dpi=150, bbox_inches="tight")
        except OSError:
            if os.path.exists(png_path):     # transient AV/indexer lock
                os.remove(png_path)
            time.sleep(1.0)
            fig.savefig(png_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        log(f"QC top-down → {out('_patch_topdown.png')}")
    except Exception as exc:
        log(f"[WARN] QC render failed: {exc}")

    # ── Optional patch PLY (voxel centers) ────────────────────────────────────
    if args.save_patch_ply:
        from plyfile import PlyElement
        rgb = np.zeros((V, 3), np.uint8)
        pal = (plt.get_cmap("tab20")(np.arange(20))[:, :3] * 255).astype(np.uint8)
        m = vox_patch > 0
        rgb[m] = pal[vox_patch[m] % 20]
        verts = np.empty(V, dtype=[("x", "f4"), ("y", "f4"), ("z", "f4"),
                                   ("red", "u1"), ("green", "u1"), ("blue", "u1"),
                                   ("patch_id", "i4")])
        verts["x"], verts["y"], verts["z"] = centroids.T.astype(np.float32)
        verts["red"], verts["green"], verts["blue"] = rgb.T
        verts["patch_id"] = vox_patch
        PlyData([PlyElement.describe(verts, "vertex")], text=False).write(out("_patches.ply"))
        log(f"Patch PLY → {out('_patches.ply')}")

    # ── Cylinder detection on the residue (columns / ducts) ──────────────────
    cylinders = detect_cylinders(uniq, normals, centroids, residue_mask)
    if cylinders:
        for cylinder in cylinders:
            cylinder["primitive_family"] = "cylindrical"
            cylinder["observation_extent"] = "fragment"
            cylinder["planar_orientation"] = "NA"
            cylinder["horizontal_context"] = "NA"
            cylinder["legacy_geom_state"] = cylinder["type"]
        cyl_path = out("_cylinders.csv")
        with open(cyl_path, "w", newline="", encoding="utf-8") as fh:
            w = _csv.DictWriter(fh, fieldnames=list(cylinders[0].keys()))
            w.writeheader()
            w.writerows(cylinders)
        log(f"Cylinders → {cyl_path} ({len(cylinders)} accepted)")
    else:
        log("Cylinders: none accepted from residue")

    # ── Summary + optional GT check (EXP-B) ───────────────────────────────────
    lines = [
        f"Step 01b v04 summary — {base}   ({time.strftime('%Y-%m-%d %H:%M:%S')})",
        f"input: {args.ply}   points={N:,}  voxels={V:,}",
        f"params: voxel={voxel}  nbin={NBIN}  band_plane={BAND_PLANE}/"
        f"{BAND_INLIER}  atomic connected fragments (cross-gap merge=OFF)",
        f"quality: grid={GRID_H}×{2*GRID_H}  δ_mixed={MP_DELTA_ABS}+{MP_DELTA_REL}·r  "
        f"iso≤{ISO_MAX_PTS}pts/≤{ISO_MAX_NEIGH}nbr  range_half={RANGE_HALF} m  "
        f"glass_behind={GLASS_BEHIND} m  filter={'OFF' if args.no_filter else 'ON'}",
        f"patches: {pid}   residue voxels: {int(residue_mask.sum()):,} "
        f"(cylinder input)   patches.csv SHA1: {sha1}",
        "",
        "quality accounting (per input point):",
        f"  mixed-pixel   {int((flags & FLAG_MIXED  > 0).sum()):>12,} "
        f"({(flags & FLAG_MIXED  > 0).mean() * 100:5.2f}%)",
        f"  isolated      {int((flags & FLAG_ISOLATED > 0).sum()):>12,} "
        f"({(flags & FLAG_ISOLATED > 0).mean() * 100:5.2f}%)",
        f"  ghost cand.   {int((flags & FLAG_GHOST_CAND > 0).sum()):>12,} "
        f"({(flags & FLAG_GHOST_CAND > 0).mean() * 100:5.2f}%)",
        f"  weight w      mean={w_point[clean_idx].mean():.3f}  "
        f"p10={np.percentile(w_point[clean_idx], 10):.3f}  "
        f"p50={np.percentile(w_point[clean_idx], 50):.3f}  "
        f"p90={np.percentile(w_point[clean_idx], 90):.3f}"
        + ("" if sensor is not None else "   [no sensor: w=1 placeholder]"),
        "",
        "legacy geometry-state counts (not final BIM classes):",
    ]
    from collections import Counter
    cnt = Counter(r["guess"] for r in rows)
    for k in sorted(cnt):
        a = sum(r["area_m2"] for r in rows if r["guess"] == k)
        cov = np.mean([r["coverage_ratio"] for r in rows if r["guess"] == k])
        mw  = np.mean([r["mean_w"] for r in rows if r["guess"] == k])
        lines.append(f"  {k:18s} n={cnt[k]:4d}   area={a:9.2f} m²   "
                     f"coverage={cov:.2f}   mean_w={mw:.2f}")

    ceil_z = sorted(r["z_mean"] for r in rows if r["guess"] == "ceiling_level")
    levels: List[List[float]] = []
    for z in ceil_z:
        if levels and z - levels[-1][-1] <= CEIL_LEVEL_TOL:
            levels[-1].append(z)
        else:
            levels.append([z])
    if levels:
        lines.append("")
        lines.append("ceiling levels (multi-level expected in MEP corridors):")
        for lv in levels:
            lines.append(f"  z ≈ {np.mean(lv):7.3f} m   ({len(lv)} patch(es))")

    if cylinders:
        lines.append("")
        lines.append("cylinders (from peeling residue):")
        for c in cylinders:
            lines.append(f"  #{c['cyl_id']:2d} {c['type']:16s} r={c['radius_m']:.3f} m  "
                         f"len={c['length_m']:.2f} m  tilt={c['tilt_deg']:.1f}°  "
                         f"rms={c['rms_m']:.3f}  cover={c['coverage_deg']:.0f}°")
    else:
        lines.append("")
        lines.append("cylinders: none accepted (residue clusters failed the "
                     "cylinder criteria)")

    if args.gt_ply:
        log(f"GT check: loading {args.gt_ply} …")
        gt_pts, gt_lab = load_gt_ply(args.gt_ply)
        # Order-independent label transfer: hash GT points into the SAME
        # voxel grid, take the per-voxel majority GT label, then compare
        # voxel classes weighted by this run's per-voxel point count.
        # (Direct index comparison is invalid — the GT file's point order
        # differs from the current Step 01 output.)
        gt_lab = np.where((gt_lab >= 0) & (gt_lab <= 5), gt_lab, 0)
        ijk_gt = np.floor((gt_pts - origin) / voxel).astype(np.int64)
        ok_rng = np.all((ijk_gt >= 0) & (ijk_gt < (1 << _BITS) - 1), axis=1)
        keys_gt = (ijk_gt[ok_rng, 0] << (2 * _BITS)) \
                | (ijk_gt[ok_rng, 1] << _BITS) | ijk_gt[ok_rng, 2]
        pos   = np.searchsorted(uniq, keys_gt)
        pos_c = np.minimum(pos, V - 1)
        hit   = uniq[pos_c] == keys_gt
        vi    = pos_c[hit]
        li    = gt_lab[ok_rng][hit]
        counts  = np.bincount(vi * 6 + li, minlength=V * 6).reshape(V, 6)
        has_gt  = counts.sum(axis=1) > 0
        gt_vox  = counts.argmax(axis=1)

        sem_of_guess = {"wall": 2, "floor": 5, "ceiling_level": 1}
        lut = np.zeros(pid + 1, np.int32)
        for r in rows:
            lut[r["patch_id"]] = sem_of_guess.get(r["guess"], 0)
        pred_vox = lut[vox_patch]
        w = n                                        # per-voxel point count

        lines.append("")
        lines.append(f"[GT] structural check (EXP-B) — voxel-hash transfer, "
                     f"point-weighted; GT voxel coverage "
                     f"{has_gt.mean() * 100:.1f}%:")
        for name, sid in [("ceiling", 1), ("wall", 2), ("floor", 5)]:
            gt_m = (gt_vox == sid) & has_gt
            pr_m = (pred_vox == sid) & has_gt
            tp   = float(w[gt_m & pr_m].sum())
            prec = tp / max(w[pr_m].sum(), 1.0)
            rec  = tp / max(w[gt_m].sum(), 1.0)
            iou  = tp / max(w[gt_m | pr_m].sum(), 1.0)
            lines.append(f"  {name:8s} P={prec:.3f}  R={rec:.3f}  IoU={iou:.3f}"
                         f"   (GT pts ≈{int(w[gt_m].sum()):,})")

    lines.append("")
    lines.append(f"elapsed: {time.time() - t0:.1f} s")
    with open(out("_patch_summary.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    log(f"Summary → {out('_patch_summary.txt')}")
    for ln in lines:
        log("  | " + ln)


if __name__ == "__main__":
    main()
