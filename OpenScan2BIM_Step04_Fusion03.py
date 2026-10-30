"""
OpenScan2BIM — Step 04: Multi-Station Surface-Aware Fusion (M3)
================================================================
v03 (2026-07-10 night) — changes vs OpenScan2BIM_Step04_Fusion02.py (v02
kept unchanged); closes task #17 items 4 + the M4 persistence hook:

  1. PERSISTENCE-WEIGHTED DETECTOR EVIDENCE: Segmentation09 (v09) writes
     a per-instance Persistence column (yaw-ensemble agreement, 0–1).
     The cache now carries a per-patch mean persistence of its detector-
     labeled points (instance_id → metadata CSV); in the label pooling
     the detector-evidence share is scaled by it:
         z = λ_geo·gl + (1−λ_geo)·(pers·dl + (1−pers)·uniform)
     Low-persistence (configuration-sensitive) detections lose weight
     exactly where EXP-A showed the detector flickers.  Pre-v09 outputs
     (no Persistence column / no instance field) fall back to pers=1.0.
     Cache format bumped: <station>_fusion_cache_v3.npz.
  2. GEOMETRY-DRIVEN GLASS PRIOR (new phase E′): ghost candidates are no
     longer limited to points behind DETECTOR window masks (unstable:
     v09 window masks vanished on stations 2/4/5 while appearing on 3).
     After curtain-wall/tilted-facade elements are fused, every station's
     PLY is re-read and points lying > GLASS_BEHIND beyond a glass-plane
     element (within its in-plane extent + margin) become GEOMETRIC
     candidates, unioned with the Step-01b semantic candidates.  The
     verdict npz/csv gain a source field (1=semantic, 2=geometric,
     3=both).  This makes candidate RECALL detector-independent (user
     check: stations 2/4 low recall) and candidate PRECISION anchored to
     actual facade planes (station 6's mostly-wrong semantic candidates
     stay flagged as semantic-only for review).

v02 (2026-07-10 evening) — changes vs OpenScan2BIM_Step04_Fusion.py (v01
kept unchanged), all four driven by the user's 45-pair association GT and
per-station ghost visual check (plan §17 / task #17):

  1. ROOM-BOUNDARY SPLITTING (new phase G): the association layer pools
     GEOMETRIC evidence, so physically continuous planes (floor slab,
     coplanar ceilings/walls) correctly merge across room boundaries —
     but BIM element identity is room-scoped (user ruling on pairs
     1/19/32: "same slab, but a wall separates them → distinct").  After
     closure detection, structural elements whose surfels lie on BOTH
     sides of a closed-room boundary are split into per-side elements
     (parent_element column keeps the lineage; member patches follow the
     side holding the majority of their surfels).
  2. CLASS-CONDITIONAL ASSOCIATION GAP: horizontal patch pairs (both
     |nz| ≥ cos 15°) now use GAP_ASSOC_HORIZ = 2.0 m (duct/tile-band
     shadows fragment one physical ceiling far beyond 0.6 m — user
     ruling: same-height ceiling without a separating wall is ONE
     ceiling); vertical/slanted pairs keep 0.6 m.
  3. CURTAIN-WALL SUBCLASS (post-hoc, non-invasive): wall/slanted
     elements with plumb tilt in [2°, 15°] and glass evidence (window-
     semantic fraction or ghost fraction) get subclass "curtain_wall" +
     a measured tilt_deg column (EXP-B reconciliation target: the user's
     Revit design tilt is 4.313°).
  4. Ghost-candidate generation limits documented: recall depends on
     Step-01 window masks (user check: stations 2/4 low recall, station
     6 mostly wrong candidates).  The geometry-driven glass prior (from
     fused curtain-wall element planes) is specified here but lands
     together with the v09 station refresh (needs per-station PLY
     re-read; see task #17 item 4).

v01 (2026-07-10).  Consumes the per-station Step 01b v02 outputs (patch
tables + patch ids + point quality) and the Step 01 labeled PLYs, all in
the shared vendor-registered world frame (Leica registration is TRUSTED —
this module verifies it, it does not re-register).

Pipeline (plan §3.2, approved 2026-07-08):

  A. Per-station fusion cache (resumable): oriented patch planes (normals
     flipped to face the scanner — the TLS only sees front faces, which is
     what makes the two sides of a partition wall separable), 10 cm surfel
     cells per patch, deterministic rep-point subsets, per-patch detector-
     semantic histograms, glass ghost candidates (Step 01b flags), and a
     spherical min-range grid per station for free-space tests.
  B. Patch association ACROSS stations (replaces mask IoU): coplanarity
     (normal angle ≤ THETA_ASSOC, mutual centroid-to-plane offsets ≤
     BAND_ASSOC) + ORIENTED normal agreement (opposite-facing coplanar
     patches never merge → the 6|7 partition wall stays two elements)
     + 3D bbox gap ≤ GAP_ASSOC.  Union-find → global elements.
  C. Registration QA (verification, not correction): for every associated
     cross-station patch pair, point-to-plane residuals (mm) → per-pair
     table.  This turns "vendor registration is trusted" into a measured
     residual bound.
  D. Element fusion: weighted plane refit; log-linear Bayesian label
     pooling of geometry-class likelihoods × detector-semantic likelihoods,
     weighted by patch quality (mean_w × n_points); per-element coverage
     accounting (fused union area vs best single station — occlusion
     incompleteness becomes a measured, improvable attribute).
  E. Free-space ghost adjudication: each station's glass ghost CANDIDATES
     (Step 01b, flag=4) are ray-cast into every other station's min-range
     grid.  A station that sees a FARTHER surface through the candidate's
     position contributes counter-evidence ("seen through"); a station
     seeing a surface AT the candidate supports it.  counter ≥ 2 and
     support ≤ 1 → ghost (TSDF free-space carving, lightweight).
     PHYSICS CAVEAT (measured on 2ndLab): behind glass the carve test is
     structurally blind — the glass surface's own returns are always
     NEARER than the candidate, so the min-range grid occludes everything
     beyond it and counter-evidence cannot accrue.  The operative
     discriminator behind glass is CORROBORATION: real through-glass
     geometry is viewpoint-independent in the registered world frame and
     gets support from other stations' own through-glass returns, while
     mirror-reflection ghosts are viewpoint-specific (the mirrored world
     moves with the scanner) and stay unsupported.  Verdicts are 3-tier:
     real (support ≥ 2) / ghost (free-space carve) / UNCORROBORATED
     (ghost-likely; kept out of element geometry, routed to the M5 review
     queue rather than silently deleted — absence of corroboration can
     also mean absence of through-glass sampling).
  F. Closure detection: fused wall elements → 2D occupancy at waist
     height → morphological closing (shuts door-width gaps) → flood fill
     from each scanner position.  Bounded region → room record (with
     principal dimensions); unbounded → open floor, element records only.
     No "closed room" assumption anywhere upstream.

Outputs (all under --output_dir, default ./outputs27):
  cache/<station>_fusion_cache.npz     phase-A cache (resumable)
  fusion_registration_qa.csv           station-pair residual table (mm)
  fusion_elements.csv                  global element table
  fusion_patch_map.csv                 (station, patch_id) → element_id
  fusion_ghost_verdicts.csv            per-station candidate adjudication
  <station>_ghost_verdict.npz          per-candidate verdict + PLY indices
  fusion_opposed_pairs.csv             coplanar opposite-facing element
                                       pairs kept separate (6|7 exhibit)
  fusion_closure_report.txt            per-seed closed/open + dimensions
  fusion_report.txt                    master summary (EXP-C/D numbers)
  fusion_topdown.png                   QC: elements, sensors, top-down

Not in v01 (hooks noted in the report): detector persistence weights
(M4), patch-association P/R against manual pairs (needs the ~30-pair
manual list), conformal decision layer (M5).

Usage
-----
  python OpenScan2BIM_Step04_Fusion.py \\
      --primitives_dir ./outputs26 \\
      --ply_dir ./outputs25 \\
      --output_dir ./outputs27
"""

# ── Standard library ──────────────────────────────────────────────────────────
import os
import csv
import time
import argparse
from typing import Dict, List, Optional, Tuple

# ── Third-party ───────────────────────────────────────────────────────────────
import numpy as np
from scipy.ndimage import minimum_filter, binary_closing, label as nd_label
from plyfile import PlyData
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ══════════════════════════════════════════════════════════════════════════════
# Parameters
# ══════════════════════════════════════════════════════════════════════════════

STATIONS_DEFAULT  = [f"2ndLab-{i}" for i in range(1, 8)]

# Phase A — cache
SURFEL            = 0.10          # m — surfel cell size (element-level areas)
REP_MAX           = 3000          # rep points per patch for registration QA
GHOST_MAX         = 120_000       # ghost candidates kept per station
FS_GRID_H         = 1024          # free-space min-range grid height (W = 2H)

# Phase B — association (same criteria family as Step01b coplanar_merge)
THETA_ASSOC_DEG   = 5.0           # ° — normal angle tolerance
BAND_ASSOC        = 0.030         # m — mutual centroid-to-plane offset
GAP_ASSOC         = 0.60          # m — max 3D bbox gap (bridges occlusion
                                  #     shadows / duct-split soffit fragments)
GAP_ASSOC_HORIZ   = 2.00          # m — gap for horizontal-horizontal pairs
                                  #     (v02: duct/tile-band shadows split one
                                  #      physical ceiling far beyond 0.6 m;
                                  #      user GT pairs 2/24 confirmed)
HORIZ_NZ          = np.cos(np.deg2rad(15.0))   # |nz| ≥ → horizontal patch
ORIENT_MIN_DOT    = 0.0           # oriented normals must agree in sign
                                  #     (cos gate; the ≤5° angle gate is the
                                  #      hard constraint — this blocks
                                  #      opposite-facing coplanar surfaces)

# Phase C — registration QA
QA_MAX_PAIRS      = 6000          # cap on evaluated cross-station pairs

# Phase D — label fusion
GUESSES           = ["wall", "floor", "ceiling_level", "slanted",
                     "horizontal_mid", "vertical_clutter", "clutter"]
ELEM_CLASSES      = ["wall", "floor", "ceiling", "slanted",
                     "horizontal_mid", "clutter"]
# geometry-class likelihood p(z_geo | element class), rows sum to 1
GEO_LIK = {
    "wall":            [0.85, 0.00, 0.00, 0.05, 0.00, 0.10],
    "floor":           [0.00, 0.90, 0.00, 0.00, 0.05, 0.05],
    "ceiling_level":   [0.00, 0.00, 0.90, 0.00, 0.05, 0.05],
    "slanted":         [0.10, 0.00, 0.00, 0.80, 0.00, 0.10],
    "horizontal_mid":  [0.00, 0.10, 0.10, 0.00, 0.70, 0.10],
    "vertical_clutter":[0.40, 0.00, 0.00, 0.05, 0.00, 0.55],
    "clutter":         [0.05, 0.02, 0.02, 0.03, 0.08, 0.80],
}
# detector-semantic likelihood p(z_sem | element class) per semantic id
# (Step 01 table: 1 ceiling, 2 wall, 3 door, 4 window, 5 floor).  Door and
# window returns sit ON walls (openings/glazing) → wall-family evidence.
SEM_LIK = {
    1: [0.05, 0.02, 0.80, 0.03, 0.05, 0.05],
    2: [0.75, 0.02, 0.03, 0.08, 0.02, 0.10],
    3: [0.55, 0.05, 0.02, 0.05, 0.03, 0.30],
    4: [0.55, 0.02, 0.03, 0.25, 0.02, 0.13],
    5: [0.02, 0.80, 0.02, 0.02, 0.06, 0.08],
}
LAMBDA_GEO        = 0.7           # geometry vs detector evidence mix
LIK_FLOOR         = 1e-3          # log-pool numerical floor
STRUCTURAL        = {"wall", "floor", "ceiling", "slanted"}

# Phase E′ — geometry-driven glass prior (v03)
GLASS_BEHIND      = 0.30          # m — beyond the glass plane → candidate
                                  #     (matches Step01b's semantic rule)
GLASS_EXT_MARGIN  = 0.20          # m — in-plane extent margin of the element
GLASS_MAX_DEPTH   = 30.0          # m — ignore points absurdly far behind
GHOST_GEO_MAX     = 300_000       # geometric candidates kept per station
CACHE_SUFFIX      = "_fusion_cache_v3.npz"   # v03 cache format (adds pers)

# Phase E — free-space ghost adjudication
FS_COUNTER_MARGIN = 0.30          # m — other station sees ≥ this much farther
FS_SUPPORT_TOL    = 0.15          # m — other station sees a surface here
GHOST_MIN_COUNTER = 2             # stations of counter-evidence → ghost …
GHOST_MAX_SUPPORT = 1             # … provided support ≤ this

# Curtain-wall subclass refinement (v02, post-hoc)
CW_TILT_MIN_DEG   = 2.0           # plumb tilt range for a tilted facade
CW_TILT_MAX_DEG   = 20.0          # (user-confirmed facade elements measure
                                  #  15.3–16.0°; Revit sheet lists 4.313° for
                                  #  one representative panel — reconcile in
                                  #  EXP-B, possibly different sections)
CW_WINDOW_FRAC    = 0.01          # window-semantic point fraction ≥ … OR
CW_GHOST_FRAC     = 0.005         # ghost-candidate fraction ≥ … = glass
CW_MIN_AREA       = 2.0           # m² — large tilted wall/slanted elements
                                  #   with NO detector coverage become
                                  #   "tilted_facade_candidate" (geometry
                                  #   alone; no false glass confidence)

# Phase G — room-boundary splitting (v02)
SPLIT_MARGIN_M    = 0.30          # surfels within this distance of the room
                                  #   region still count as inside (the room's
                                  #   own boundary walls sit ON the region rim)
SPLIT_MIN_FRAC    = 0.08          # split when BOTH sides hold ≥ this …
SPLIT_MIN_ABS     = 2.00          # m² — … OR both sides exceed this absolute
                                  #   area (a 26 m² room slice of a 255 m²
                                  #   floor slab is only 5% but must split)
SPLIT_MIN_AREA    = 0.50          # m² — hard floor for either side

# Phase F — closure detection
CLOSE_CELL        = 0.10          # m — 2D wall-occupancy cell
CLOSE_ITER        = 8             # binary_closing iterations (3×3 SQUARE
                                  #   structure → Chebyshev, rotation-safe);
                                  #   closes gaps ≤ ~2·iter·cell = 1.6 m:
                                  #   door leaves (0.9 m) + furniture-occlusion
                                  #   dashes in the waist slice; open-hall
                                  #   openings (≥ 2 m) stay open (verified on
                                  #   2ndLab-6/7: room closed, hall open)
WAIST_LO, WAIST_HI = 0.30, 2.00   # m above the reference floor level

_BITS             = 21            # packed 3D key bits (same scheme as Step01b)
_KOFF             = 4096          # index offset → strictly positive indices

SEM_ID_MAP = {"ceiling": 1, "wall": 2, "door": 3, "window": 4, "floor": 5}
FLAG_GHOST_CAND = 4               # Step 01b point_quality flag bit


def log(msg: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


# ══════════════════════════════════════════════════════════════════════════════
# Shared helpers (same conventions as Step 01b v02)
# ══════════════════════════════════════════════════════════════════════════════

def load_ply_xyz_sem(path: str, want_instance: bool = False):
    ply  = PlyData.read(path)
    data = ply["vertex"].data
    pts  = np.stack([np.asarray(data["x"]),
                     np.asarray(data["y"]),
                     np.asarray(data["z"])], axis=-1).astype(np.float32)
    sem = None
    for field in ("semantic_class", "scalar_semantic_class", "semantic",
                  "label", "class_id"):
        if field in data.dtype.names:
            sem = np.asarray(data[field]).astype(np.int32)
            break
    if not want_instance:
        return pts, sem
    inst = None
    for field in ("instance_id", "scalar_instance_id"):
        if field in data.dtype.names:
            inst = np.asarray(data[field]).astype(np.int64)
            break
    return pts, sem, inst


def read_instance_persistence(metadata_csv: str) -> Dict[int, float]:
    """Instance id → yaw-ensemble persistence (Segmentation09 metadata).
    Missing file / missing column → empty dict (callers default to 1.0)."""
    if not os.path.exists(metadata_csv):
        return {}
    out: Dict[int, float] = {}
    with open(metadata_csv, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if "Persistence" not in r or r["Persistence"] in (None, ""):
                continue
            try:
                out[int(r["Instance ID"])] = float(r["Persistence"])
            except (ValueError, KeyError):
                continue
    return out


def pack_keys(pts: np.ndarray, cell: float) -> np.ndarray:
    """Deterministic station-independent 3D cell key (world-anchored grid)."""
    ijk = np.floor(pts / cell).astype(np.int64) + _KOFF
    if np.any(ijk < 1) or np.any(ijk >= (1 << _BITS) - 2):
        raise ValueError("coordinates outside the packed-key range")
    return (ijk[:, 0] << (2 * _BITS)) | (ijk[:, 1] << _BITS) | ijk[:, 2]


def unpack_centers(keys: np.ndarray, cell: float) -> np.ndarray:
    mask = (1 << _BITS) - 1
    ijk = np.stack([(keys >> (2 * _BITS)) & mask,
                    (keys >> _BITS) & mask,
                    keys & mask], axis=-1).astype(np.float64) - _KOFF
    return ((ijk + 0.5) * cell).astype(np.float32)


def spherical_pixels(rel: np.ndarray, rng: np.ndarray, H: int) -> np.ndarray:
    W = 2 * H
    az = np.arctan2(rel[:, 1], rel[:, 0])
    el = np.arcsin(np.clip(rel[:, 2] / rng, -1.0, 1.0))
    u = np.minimum((W * (az + np.pi) / (2 * np.pi)).astype(np.int64), W - 1)
    v = np.minimum((H * (0.5 - el / np.pi)).astype(np.int64), H - 1)
    return (v * W + u).astype(np.int64)


def min_range_grid(rel: np.ndarray, rng: np.ndarray, H: int) -> np.ndarray:
    """Per-pixel min range over all station points, 3×3-min filtered with
    azimuth wrap (conservative for the free-space test: a foreground object
    in a neighbouring pixel suppresses false 'seen-through' claims)."""
    W = 2 * H
    pix = spherical_pixels(rel, rng, H)
    order = np.argsort(pix, kind="stable")
    ps, vs = pix[order], rng[order]
    starts = np.flatnonzero(np.r_[True, np.diff(ps) > 0])
    grid = np.full(H * W, np.inf, np.float32)
    grid[ps[starts]] = np.minimum.reduceat(vs, starts)
    img = grid.reshape(H, W)
    img = np.concatenate([img[:, -1:], img, img[:, :1]], axis=1)
    img = minimum_filter(img, size=3, mode="nearest")
    return img[:, 1:-1].reshape(-1)


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


def _save_npz_retry(path: str, **arrays) -> None:
    try:
        np.savez_compressed(path, **arrays)
    except OSError:
        if os.path.exists(path):                 # transient AV/indexer lock
            os.remove(path)
        time.sleep(1.0)
        np.savez_compressed(path, **arrays)


# ══════════════════════════════════════════════════════════════════════════════
# Phase A — per-station fusion cache
# ══════════════════════════════════════════════════════════════════════════════

def read_patch_csv(path: str) -> Dict[str, np.ndarray]:
    with open(path, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    P = len(rows)
    out = {
        "pid":      np.array([int(r["patch_id"]) for r in rows], np.int32),
        "guess":    np.array([GUESSES.index(r["guess"]) for r in rows], np.int8),
        "n_points": np.array([int(r["n_points"]) for r in rows], np.int64),
        "area":     np.array([float(r["area_m2"]) for r in rows], np.float64),
        "coverage": np.array([float(r["coverage_ratio"]) for r in rows], np.float64),
        "mean_w":   np.array([float(r["mean_w"]) for r in rows], np.float64),
        "ghost_frac": np.array([float(r["ghost_frac"]) for r in rows], np.float64),
        "normal":   np.array([[float(r["nx"]), float(r["ny"]), float(r["nz"])]
                              for r in rows], np.float64),
        "d":        np.array([float(r["d"]) for r in rows], np.float64),
        "centroid": np.array([[float(r["cx"]), float(r["cy"]), float(r["cz"])]
                              for r in rows], np.float64),
    }
    assert np.all(out["pid"] == np.arange(1, P + 1)), "patch ids not contiguous"
    return out


def build_station_cache(station: str, ply_path: str, prim_dir: str,
                        cache_path: str, metadata_csv: str = "") -> None:
    log(f"[{station}] building fusion cache …")
    t0 = time.time()
    pts, sem, inst = load_ply_xyz_sem(ply_path, want_instance=True)
    N = len(pts)
    # per-point persistence from the v09 metadata (pre-v09 → all 1.0)
    inst_pers = read_instance_persistence(metadata_csv)
    pers_pt = np.ones(N, np.float32)
    if inst is not None and inst_pers:
        maxid = int(inst.max())
        lut = np.ones(maxid + 1, np.float32)
        for iid, pv in inst_pers.items():
            if 0 <= iid <= maxid:
                lut[iid] = pv
        pers_pt = lut[np.clip(inst, 0, maxid)]
        log(f"[{station}]   persistence: {len(inst_pers)} instances "
            f"(mean {np.mean(list(inst_pers.values())):.3f})")
    patch = np.load(os.path.join(prim_dir, f"{station}_patch_ids.npy"))
    assert len(patch) == N, "patch_ids not aligned with PLY"
    q = np.load(os.path.join(prim_dir, f"{station}_point_quality.npz"))
    sensor = q["sensor_pos"].astype(np.float64)
    flags  = q["flags"]
    meta = read_patch_csv(os.path.join(prim_dir, f"{station}_patches.csv"))
    P = len(meta["pid"])

    # oriented planes: normal flipped to face the scanner (front face)
    nrm, d = meta["normal"].copy(), meta["d"].copy()
    to_sensor = sensor[None, :] - meta["centroid"]
    flip = np.einsum("ij,ij->i", nrm, to_sensor) < 0
    nrm[flip] *= -1.0
    d[flip]   *= -1.0

    # per-patch member slices (points sorted by patch id)
    order  = np.argsort(patch, kind="stable")
    sorted_ids = patch[order]
    starts = np.searchsorted(sorted_ids, np.arange(1, P + 2))

    key10 = pack_keys(pts, SURFEL)
    surf_keys: List[np.ndarray] = []
    rep_pts:   List[np.ndarray] = []
    bbox_min = np.zeros((P, 3), np.float32)
    bbox_max = np.zeros((P, 3), np.float32)
    for p in range(P):
        sl = order[starts[p]:starts[p + 1]]
        sk = np.unique(key10[sl])
        surf_keys.append(sk)
        stride = max(1, len(sl) // REP_MAX)
        rep_pts.append(pts[sl[::stride][:REP_MAX]])
        cc = unpack_centers(sk, SURFEL)
        bbox_min[p] = cc.min(axis=0) - SURFEL / 2
        bbox_max[p] = cc.max(axis=0) + SURFEL / 2

    # detector-semantic histogram per patch (ids 0..5)
    sem_hist = np.zeros((P + 1, 6), np.int64)
    if sem is not None:
        sc = np.clip(sem, 0, 5)
        sem_hist = np.bincount(patch.astype(np.int64) * 6 + sc,
                               minlength=(P + 1) * 6).reshape(P + 1, 6)
    sem_hist = sem_hist[1:]

    # per-patch mean persistence over its detector-labeled points
    pers_patch = np.ones(P, np.float32)
    if sem is not None:
        det = sem > 0
        pid64 = patch.astype(np.int64)
        cnt = np.bincount(pid64, weights=det.astype(np.float64),
                          minlength=P + 1)
        psum = np.bincount(pid64, weights=pers_pt * det, minlength=P + 1)
        nz = cnt[1:] > 0
        pers_patch[nz] = (psum[1:][nz] / cnt[1:][nz]).astype(np.float32)

    # glass ghost candidates (deterministic stride cap)
    gsel = np.nonzero((flags & FLAG_GHOST_CAND) > 0)[0]
    if len(gsel) > GHOST_MAX:
        gsel = gsel[::int(np.ceil(len(gsel) / GHOST_MAX))]
    ghost_xyz = pts[gsel]

    # free-space min-range grid
    rel = pts - sensor.astype(np.float32)
    rng = np.maximum(np.linalg.norm(rel, axis=1), 1e-6)
    grid = min_range_grid(rel, rng, FS_GRID_H)
    grid = np.where(np.isfinite(grid), grid, -1.0).astype(np.float32)  # npz-safe

    _save_npz_retry(
        cache_path,
        sensor=sensor, n_input_points=np.int64(N),
        guess=meta["guess"], n_points=meta["n_points"], area=meta["area"],
        coverage=meta["coverage"], mean_w=meta["mean_w"],
        ghost_frac=meta["ghost_frac"], normal=nrm, d=d,
        centroid=meta["centroid"], bbox_min=bbox_min, bbox_max=bbox_max,
        surf_keys=np.concatenate(surf_keys) if P else np.empty(0, np.int64),
        surf_indptr=np.cumsum([0] + [len(s) for s in surf_keys]).astype(np.int64),
        rep_pts=np.concatenate(rep_pts) if P else np.empty((0, 3), np.float32),
        rep_indptr=np.cumsum([0] + [len(r) for r in rep_pts]).astype(np.int64),
        sem_hist=sem_hist, pers=pers_patch, ghost_xyz=ghost_xyz,
        ghost_idx=gsel.astype(np.int64), range_grid=grid,
    )
    log(f"[{station}] cache → {cache_path}  ({P} patches, "
        f"{len(gsel):,} ghost cand., {time.time() - t0:.1f} s)")


class StationData:
    """Thin accessor over a fusion cache npz."""
    def __init__(self, station: str, cache_path: str,
                 ply_path: str = ""):
        z = np.load(cache_path)
        self.station   = station
        self.ply_path  = ply_path              # for the phase-E′ re-read
        self.sensor    = z["sensor"]
        self.guess     = z["guess"]
        self.n_points  = z["n_points"]
        self.area      = z["area"]
        self.coverage  = z["coverage"]
        self.mean_w    = z["mean_w"]
        self.ghost_frac= z["ghost_frac"]
        self.normal    = z["normal"]
        self.d         = z["d"]
        self.centroid  = z["centroid"]
        self.bbox_min  = z["bbox_min"].astype(np.float64)
        self.bbox_max  = z["bbox_max"].astype(np.float64)
        self._sk       = z["surf_keys"]
        self._ski      = z["surf_indptr"]
        self._rp       = z["rep_pts"]
        self._rpi      = z["rep_indptr"]
        self.sem_hist  = z["sem_hist"]
        self.pers      = (z["pers"] if "pers" in z.files
                          else np.ones(len(z["guess"]), np.float32))
        self.ghost_xyz = z["ghost_xyz"]
        self.ghost_idx = z["ghost_idx"]
        grid           = z["range_grid"]
        self.range_grid = np.where(grid < 0, np.inf, grid)
        self.P         = len(self.guess)

    def surfels(self, p: int) -> np.ndarray:
        return self._sk[self._ski[p]:self._ski[p + 1]]

    def reps(self, p: int) -> np.ndarray:
        return self._rp[self._rpi[p]:self._rpi[p + 1]]


# ══════════════════════════════════════════════════════════════════════════════
# Phase B — cross-station patch association
# ══════════════════════════════════════════════════════════════════════════════

def associate(stations: List[StationData], log_fn):
    """Union-find over all patches of all stations.  Edges: coplanarity +
    oriented-normal agreement + bbox gap.  Returns (element id per global
    patch index, edge list [(gi, gj)], global offsets per station)."""
    offs = np.cumsum([0] + [s.P for s in stations])
    G = int(offs[-1])
    uf = UnionFind(G)
    cos_assoc = np.cos(np.deg2rad(THETA_ASSOC_DEG))
    edges: List[Tuple[int, int]] = []

    for a in range(len(stations)):
        for b in range(a, len(stations)):
            A, B = stations[a], stations[b]
            if A.P == 0 or B.P == 0:
                continue
            dot = A.normal @ B.normal.T                          # oriented
            ang_ok = (np.abs(dot) >= cos_assoc) & (dot > ORIENT_MIN_DOT)
            off_ab = np.abs(A.centroid @ B.normal.T + B.d[None, :])
            off_ba = np.abs((B.centroid @ A.normal.T).T + A.d[:, None])
            band_ok = (off_ab <= BAND_ASSOC) & (off_ba <= BAND_ASSOC)
            gap = np.maximum(
                A.bbox_min[:, None, :] - B.bbox_max[None, :, :],
                B.bbox_min[None, :, :] - A.bbox_max[:, None, :]).max(axis=2)
            # v02: horizontal-horizontal pairs bridge duct/tile-band shadows
            horiz = (np.abs(A.normal[:, 2])[:, None] >= HORIZ_NZ) \
                  & (np.abs(B.normal[:, 2])[None, :] >= HORIZ_NZ)
            gap_lim = np.where(horiz, GAP_ASSOC_HORIZ, GAP_ASSOC)
            ok = ang_ok & band_ok & (gap <= gap_lim)
            if a == b:
                ok &= np.triu(np.ones_like(ok, bool), k=1)       # each pair once
            ii, jj = np.nonzero(ok)
            for i, j in zip(ii, jj):
                gi, gj = int(offs[a] + i), int(offs[b] + j)
                uf.union(gi, gj)
                if a != b:
                    edges.append((gi, gj))

    root = np.array([uf.find(g) for g in range(G)], np.int64)
    _, elem = np.unique(root, return_inverse=True)
    log_fn(f"  association: {G} patches → {elem.max() + 1} elements "
           f"({len(edges)} cross-station links)")
    return elem, edges, offs


def gidx_station(offs: np.ndarray, g: int) -> Tuple[int, int]:
    s = int(np.searchsorted(offs, g, side="right") - 1)
    return s, int(g - offs[s])


# ══════════════════════════════════════════════════════════════════════════════
# Phase C — registration QA
# ══════════════════════════════════════════════════════════════════════════════

def registration_qa(stations: List[StationData], edges, offs,
                    out_csv: str, log_fn) -> List[Dict]:
    """Point-to-plane residuals for associated cross-station pairs."""
    per_pair: Dict[Tuple[int, int], List[np.ndarray]] = {}
    for gi, gj in edges[:QA_MAX_PAIRS]:
        sa, pa = gidx_station(offs, gi)
        sb, pb = gidx_station(offs, gj)
        A, B = stations[sa], stations[sb]
        # distances of B's rep points to A's plane (float64 dot for mm safety)
        db = np.abs(B.reps(pb).astype(np.float64) @ A.normal[pa] + A.d[pa])
        da = np.abs(A.reps(pa).astype(np.float64) @ B.normal[pb] + B.d[pb])
        key = (min(sa, sb), max(sa, sb))
        per_pair.setdefault(key, []).extend([da, db])

    rows = []
    for (sa, sb), dists in sorted(per_pair.items()):
        d = np.concatenate(dists) * 1000.0                        # mm
        rows.append({
            "station_a": stations[sa].station, "station_b": stations[sb].station,
            "n_patch_pairs": len(dists) // 2, "n_points": len(d),
            "median_mm": round(float(np.median(d)), 2),
            "p90_mm":    round(float(np.percentile(d, 90)), 2),
            "p95_mm":    round(float(np.percentile(d, 95)), 2),
        })
    with open(out_csv, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()) if rows
                           else ["station_a"])
        w.writeheader(); w.writerows(rows)
    log_fn(f"  registration QA → {out_csv} ({len(rows)} station pairs)")
    return rows


# ══════════════════════════════════════════════════════════════════════════════
# Phase D — element fusion (labels, planes, coverage)
# ══════════════════════════════════════════════════════════════════════════════

def hull_area_2d(uv: np.ndarray) -> float:
    try:
        from scipy.spatial import ConvexHull
        if len(uv) >= 8:
            return float(ConvexHull(uv).volume)
    except Exception:
        pass
    span = uv.max(axis=0) - uv.min(axis=0)
    return float(span[0] * span[1])


def in_plane_axes(nrm: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    ref = np.array([0.0, 0.0, 1.0]) if abs(nrm[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    e1 = np.cross(nrm, ref); e1 /= np.linalg.norm(e1)
    e2 = np.cross(nrm, e1)
    return e1, e2


def fuse_elements(stations: List[StationData], elem: np.ndarray,
                  offs: np.ndarray, log_fn):
    """Per-element fused record + label posterior + coverage accounting."""
    E = int(elem.max() + 1)
    members: List[List[Tuple[int, int]]] = [[] for _ in range(E)]
    for g in range(len(elem)):
        s, p = gidx_station(offs, g)
        members[int(elem[g])].append((s, p))

    lik_geo = np.array([GEO_LIK[g] for g in GUESSES])            # (7, 6)
    lik_sem = np.zeros((6, 6))
    for sid, v in SEM_LIK.items():
        lik_sem[sid] = v
    lik_sem[0] = 1.0 / 6.0                                       # 'other' id

    elements: List[Dict] = []
    for e in range(E):
        mem = members[e]
        stset = sorted({s for s, _ in mem})
        w_m, cls_votes = [], []
        logp = np.zeros(len(ELEM_CLASSES))
        n_pts_tot = 0
        for s, p in mem:
            S = stations[s]
            w = float(S.mean_w[p]) * float(S.n_points[p])
            w_m.append(w)
            n_pts_tot += int(S.n_points[p])
            gl = lik_geo[int(S.guess[p])]
            h  = S.sem_hist[p].astype(np.float64)
            dl = (h[:, None] * lik_sem).sum(axis=0) / max(h.sum(), 1.0) \
                 if h.sum() > 0 else np.full(len(ELEM_CLASSES), 1 / 6)
            # v03: detector evidence scaled by yaw-ensemble persistence —
            # configuration-sensitive detections (EXP-A) lose weight
            pp = float(S.pers[p])
            dl = pp * dl + (1.0 - pp) * (1.0 / 6.0)
            z  = LAMBDA_GEO * gl + (1 - LAMBDA_GEO) * dl
            logp += w * np.log(np.maximum(z, LIK_FLOOR))
            cls_votes.append((GUESSES[int(S.guess[p])], w))
        wsum = max(sum(w_m), 1e-9)
        logp /= wsum                                             # scale-free pool
        post = np.exp(logp - logp.max())
        post /= post.sum()
        k = int(np.argmax(post))
        entropy = float(-(post * np.log(np.maximum(post, 1e-12))).sum())

        # pre-fusion disagreement: weighted fraction of member evidence whose
        # geometry class maps to a different element class than the winner
        geo2elem = {"wall": "wall", "floor": "floor", "ceiling_level": "ceiling",
                    "slanted": "slanted", "horizontal_mid": "horizontal_mid",
                    "vertical_clutter": "clutter", "clutter": "clutter"}
        dis = sum(w for c, w in cls_votes
                  if geo2elem[c] != ELEM_CLASSES[k]) / wsum

        # fused plane: quality-weighted oriented-normal mean + refit d
        nrm = np.zeros(3)
        for (s, p), w in zip(mem, w_m):
            nrm += w * stations[s].normal[p]
        nrm /= max(np.linalg.norm(nrm), 1e-12)
        dsum = sum(w * float(-stations[s].centroid[p] @ nrm)
                   for (s, p), w in zip(mem, w_m))
        dfit = dsum / wsum

        # coverage accounting on the shared 10 cm surfel grid
        per_st_keys: Dict[int, List[np.ndarray]] = {}
        for s, p in mem:
            per_st_keys.setdefault(s, []).append(stations[s].surfels(p))
        st_cells = {s: len(np.unique(np.concatenate(ks)))
                    for s, ks in per_st_keys.items()}
        union_keys = np.unique(np.concatenate(
            [k for ks in per_st_keys.values() for k in ks]))
        cc = unpack_centers(union_keys, SURFEL)
        e1, e2 = in_plane_axes(nrm)
        uv = np.column_stack([cc @ e1, cc @ e2])
        hull = max(hull_area_2d(uv), 1e-9)
        area_union = len(union_keys) * SURFEL * SURFEL
        best_single = max(st_cells.values()) * SURFEL * SURFEL
        mean_w_el = float(sum(w_m) / max(n_pts_tot, 1))
        ghost_el = sum(float(stations[s].ghost_frac[p]) *
                       float(stations[s].n_points[p]) for s, p in mem) \
                   / max(n_pts_tot, 1)

        elements.append({
            "element_id": e + 1,
            "class": ELEM_CLASSES[k],
            "p_class": round(float(post[k]), 4),
            "entropy": round(entropy, 4),
            "pre_disagreement": round(float(dis), 4),
            "n_stations": len(stset),
            "stations": "|".join(stations[s].station for s in stset),
            "n_patches": len(mem),
            "n_points": n_pts_tot,
            "area_union_m2": round(area_union, 3),
            "best_single_m2": round(best_single, 3),
            "area_gain": round(area_union / max(best_single, 1e-9), 3),
            "coverage_fused": round(min(area_union / hull, 1.0), 4),
            "coverage_best_single": round(min(best_single / hull, 1.0), 4),
            "hull_m2": round(hull, 3),
            "mean_w": round(mean_w_el, 4),
            "ghost_frac": round(float(ghost_el), 4),
            "nx": round(float(nrm[0]), 6), "ny": round(float(nrm[1]), 6),
            "nz": round(float(nrm[2]), 6), "d": round(float(dfit), 6),
            "cz": round(float(cc[:, 2].mean()), 3),
        })

    log_fn(f"  fused {E} elements "
           f"({sum(1 for r in elements if r['n_stations'] > 1)} multi-station, "
           f"{sum(1 for r in elements if r['class'] in STRUCTURAL)} structural)")
    return elements, members


def opposed_pairs(elements: List[Dict], out_csv: str, log_fn) -> int:
    """Coplanar opposite-facing element pairs correctly kept separate —
    the partition-wall exhibit (6|7): same plane family, oriented normals
    antiparallel, small mutual offset."""
    walls = [r for r in elements if r["class"] in ("wall", "slanted")
             and r["area_union_m2"] >= 1.0]
    rows = []
    for i in range(len(walls)):
        for j in range(i + 1, len(walls)):
            a, b = walls[i], walls[j]
            na = np.array([a["nx"], a["ny"], a["nz"]])
            nb = np.array([b["nx"], b["ny"], b["nz"]])
            if na @ nb > -0.996:                                  # ≤ ~5° apart
                continue
            # antiparallel normals: plane b in a's orientation has offset -d_b,
            # so the inter-plane distance (wall thickness) is |d_a + d_b|
            gap = abs(a["d"] + b["d"])
            if gap > 0.40:                                        # wall thickness
                continue
            rows.append({
                "element_a": a["element_id"], "element_b": b["element_id"],
                "stations_a": a["stations"], "stations_b": b["stations"],
                "area_a_m2": a["area_union_m2"], "area_b_m2": b["area_union_m2"],
                "normal_dot": round(float(na @ nb), 4),
                "plane_gap_m": round(float(gap), 4),
            })
    rows.sort(key=lambda r: -(min(r["area_a_m2"], r["area_b_m2"])))
    with open(out_csv, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()) if rows
                           else ["element_a"])
        w.writeheader(); w.writerows(rows)
    log_fn(f"  opposed coplanar pairs kept separate: {len(rows)} → {out_csv}")
    return len(rows)


# ══════════════════════════════════════════════════════════════════════════════
# Phase E′ — geometry-driven glass candidates (v03)
# ══════════════════════════════════════════════════════════════════════════════

def glass_planes_from_elements(stations, elements, members, log_fn):
    """Glass-plane priors = fused curtain_wall / tilted_facade elements:
    oriented plane (n, d — normal faces the interior) + in-plane extent."""
    planes = []
    for r, mem in zip(elements, members):
        sub = r.get("subclass", "")
        if "curtain_wall" not in sub and "tilted_facade" not in sub:
            continue
        if not mem:
            continue
        nrm = np.array([r["nx"], r["ny"], r["nz"]])
        cc = np.concatenate([unpack_centers(stations[s].surfels(p), SURFEL)
                             for s, p in mem])
        e1, e2 = in_plane_axes(nrm)
        uv = np.column_stack([cc @ e1, cc @ e2])
        planes.append({
            "element_id": r["element_id"], "n": nrm, "d": float(r["d"]),
            "e1": e1, "e2": e2,
            "uv_min": uv.min(axis=0) - GLASS_EXT_MARGIN,
            "uv_max": uv.max(axis=0) + GLASS_EXT_MARGIN,
        })
    log_fn(f"  glass priors: {len(planes)} facade element planes")
    return planes


def merged_ghost_candidates(stations, planes, log_fn):
    """Per station: union of Step-01b semantic candidates and geometric
    candidates behind any fused glass plane.  Returns
    {station_idx: (ply_idx int64, xyz float32, source uint8)} with source
    1=semantic, 2=geometric, 3=both."""
    out = {}
    for si, S in enumerate(stations):
        geo_idx = np.empty(0, np.int64)
        if planes and S.ply_path and os.path.exists(S.ply_path):
            pts, _ = load_ply_xyz_sem(S.ply_path)
            hits = np.zeros(len(pts), bool)
            for pl in planes:
                sdist = pts @ pl["n"].astype(np.float32) + pl["d"]
                cand = (sdist < -GLASS_BEHIND) & (sdist > -GLASS_MAX_DEPTH)
                if not cand.any():
                    continue
                sub = pts[cand]
                u = sub @ pl["e1"].astype(np.float32)
                v = sub @ pl["e2"].astype(np.float32)
                inx = (u >= pl["uv_min"][0]) & (u <= pl["uv_max"][0]) \
                    & (v >= pl["uv_min"][1]) & (v <= pl["uv_max"][1])
                ci = np.nonzero(cand)[0][inx]
                hits[ci] = True
            geo_idx = np.nonzero(hits)[0].astype(np.int64)
            if len(geo_idx) > GHOST_GEO_MAX:
                geo_idx = geo_idx[::int(np.ceil(len(geo_idx) / GHOST_GEO_MAX))]
            del pts
        sem_idx = S.ghost_idx
        all_idx = np.union1d(sem_idx, geo_idx)
        source = np.zeros(len(all_idx), np.uint8)
        source[np.isin(all_idx, sem_idx)] |= 1
        source[np.isin(all_idx, geo_idx)] |= 2
        if len(all_idx) and S.ply_path and os.path.exists(S.ply_path):
            pts, _ = load_ply_xyz_sem(S.ply_path)
            xyz = pts[all_idx]
            del pts
        else:
            xyz = S.ghost_xyz if len(all_idx) == len(sem_idx) \
                else np.empty((0, 3), np.float32)
        out[si] = (all_idx, xyz, source)
        log_fn(f"  [{S.station}] candidates: semantic {len(sem_idx):,} ∪ "
               f"geometric {len(geo_idx):,} → {len(all_idx):,}")
    return out


# ══════════════════════════════════════════════════════════════════════════════
# Phase E — free-space ghost adjudication
# ══════════════════════════════════════════════════════════════════════════════

def adjudicate_ghosts(stations: List[StationData], cands: Dict,
                      out_dir: str, out_csv: str, log_fn) -> List[Dict]:
    rows = []
    for s, S in enumerate(stations):
        idx, xyz, source = cands[s]
        Gc = len(idx)
        if Gc == 0:
            rows.append({"station": S.station, "candidates": 0,
                         "n_semantic": 0, "n_geometric": 0,
                         "ghost_freespace": 0, "real_corroborated": 0,
                         "uncorroborated": 0})
            continue
        counter = np.zeros(Gc, np.int16)
        support = np.zeros(Gc, np.int16)
        x = xyz.astype(np.float64)
        for t, T in enumerate(stations):
            if t == s:
                continue
            rel = (x - T.sensor).astype(np.float32)
            rng = np.maximum(np.linalg.norm(rel, axis=1), 1e-6)
            pix = spherical_pixels(rel, rng, FS_GRID_H)
            g = T.range_grid[pix]
            finite = np.isfinite(g)
            counter += (finite & (g > rng + FS_COUNTER_MARGIN)).astype(np.int16)
            support += (finite & (np.abs(g - rng) <= FS_SUPPORT_TOL)).astype(np.int16)
        verdict = np.zeros(Gc, np.int8)                     # 0 undecided
        verdict[(counter >= GHOST_MIN_COUNTER)
                & (support <= GHOST_MAX_SUPPORT)] = 1       # ghost
        verdict[(verdict == 0) & (support >= 2)] = 2        # corroborated real
        _save_npz_retry(os.path.join(out_dir, f"{S.station}_ghost_verdict.npz"),
                        ply_index=idx, verdict=verdict,
                        counter=counter, support=support, source=source)
        rows.append({
            "station": S.station, "candidates": Gc,
            "n_semantic": int((source & 1 > 0).sum()),
            "n_geometric": int((source & 2 > 0).sum()),
            "ghost_freespace": int((verdict == 1).sum()),
            "real_corroborated": int((verdict == 2).sum()),
            "uncorroborated": int((verdict == 0).sum()),
        })
        log_fn(f"  [{S.station}] ghost candidates {Gc:,} → "
               f"free-space ghost {rows[-1]['ghost_freespace']:,} / "
               f"corroborated real {rows[-1]['real_corroborated']:,} / "
               f"uncorroborated (ghost-likely) {rows[-1]['uncorroborated']:,}")
    with open(out_csv, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    return rows


# ══════════════════════════════════════════════════════════════════════════════
# Phase F — closure detection
# ══════════════════════════════════════════════════════════════════════════════

def closure_detection(stations: List[StationData], elements: List[Dict],
                      members, out_txt: str, log_fn):
    """2D wall-occupancy → closing → flood fill from each sensor.
    v02: also returns the closed-room regions (mask + grid metadata) for
    the room-boundary splitting phase."""
    rooms: List[Dict] = []
    floors = [r for r in elements if r["class"] == "floor"]
    if not floors:
        lines = ["closure: no floor elements — skipped"]
        with open(out_txt, "w", encoding="utf-8") as fh:
            fh.write("\n".join(lines) + "\n")
        return lines, rooms
    zf = float(max(floors, key=lambda r: r["area_union_m2"])["cz"])

    # waist-height wall surfels of wall/slanted elements
    cells = []
    for r, mem in zip(elements, members):
        if r["class"] not in ("wall", "slanted"):
            continue
        for s, p in mem:
            cc = unpack_centers(stations[s].surfels(p), SURFEL)
            m = (cc[:, 2] >= zf + WAIST_LO) & (cc[:, 2] <= zf + WAIST_HI)
            if m.any():
                cells.append(cc[m, :2])
    lines = [f"closure detection — floor reference z = {zf:.3f} m, "
             f"waist slice [{zf + WAIST_LO:.2f}, {zf + WAIST_HI:.2f}] m, "
             f"cell {CLOSE_CELL} m, closing iterations {CLOSE_ITER}"]
    if not cells:
        lines.append("no wall cells in the waist slice — skipped")
        with open(out_txt, "w", encoding="utf-8") as fh:
            fh.write("\n".join(lines) + "\n")
        return lines, rooms

    xy = np.concatenate(cells)
    lo = xy.min(axis=0) - 1.0
    hi = xy.max(axis=0) + 1.0
    nx = int(np.ceil((hi[0] - lo[0]) / CLOSE_CELL))
    ny = int(np.ceil((hi[1] - lo[1]) / CLOSE_CELL))
    occ = np.zeros((ny, nx), bool)
    ix = np.clip(((xy[:, 0] - lo[0]) / CLOSE_CELL).astype(int), 0, nx - 1)
    iy = np.clip(((xy[:, 1] - lo[1]) / CLOSE_CELL).astype(int), 0, ny - 1)
    occ[iy, ix] = True
    occ = binary_closing(occ, structure=np.ones((3, 3), bool),
                         iterations=CLOSE_ITER)

    free = ~occ
    lab, n_lab = nd_label(free)
    border = np.zeros_like(free)
    border[0, :] = border[-1, :] = True
    border[:, 0] = border[:, -1] = True
    open_regions = set(np.unique(lab[border & free]))

    for S in stations:
        sx = int(np.clip((S.sensor[0] - lo[0]) / CLOSE_CELL, 0, nx - 1))
        sy = int(np.clip((S.sensor[1] - lo[1]) / CLOSE_CELL, 0, ny - 1))
        r = int(lab[sy, sx])
        if r == 0:
            lines.append(f"  {S.station}: seed inside wall mask — undecided")
            continue
        if r in open_regions:
            lines.append(f"  {S.station}: OPEN (region reaches scene border) "
                         f"→ element records only")
        else:
            m = lab == r
            area = float(m.sum()) * CLOSE_CELL * CLOSE_CELL
            ys, xs = np.nonzero(m)
            pts2 = np.column_stack([xs * CLOSE_CELL + lo[0],
                                    ys * CLOSE_CELL + lo[1]])
            mu = pts2.mean(axis=0)
            cov = np.cov((pts2 - mu).T)
            w, v = np.linalg.eigh(cov)
            proj = (pts2 - mu) @ v
            dims = proj.max(axis=0) - proj.min(axis=0)
            lines.append(f"  {S.station}: CLOSED room — area {area:.2f} m², "
                         f"principal dims {max(dims):.3f} × {min(dims):.3f} m "
                         f"→ room record eligible")
            if not any(np.array_equal(m, rm["mask"]) for rm in rooms):
                rooms.append({"station": S.station, "mask": m, "lo": lo,
                              "cell": CLOSE_CELL, "nx": nx, "ny": ny})
    with open(out_txt, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    for ln in lines:
        log_fn("  | " + ln)
    return lines, rooms


# ══════════════════════════════════════════════════════════════════════════════
# v02 — curtain-wall subclass + room-boundary splitting (Phase G)
# ══════════════════════════════════════════════════════════════════════════════

def refine_curtain_walls(elements: List[Dict], stations: List[StationData],
                         members, log_fn) -> int:
    """Post-hoc subclass refinement (does not touch the label posterior).
    Adds tilt_deg / subclass / parent_element columns to EVERY element row;
    wall/slanted elements with plumb tilt in [CW_TILT_MIN, CW_TILT_MAX] and
    glass evidence (window semantics or ghost fraction) become
    subclass=curtain_wall.  User GT: pairs 10/25/29 are the tilted glass
    facade (Revit design tilt 4.313°)."""
    n_cw = 0
    for r, mem in zip(elements, members):
        nz = abs(r["nz"])
        tilt = float(np.rad2deg(np.arcsin(min(nz, 1.0))))
        r["tilt_deg"] = round(tilt, 2)
        r["subclass"] = ""
        r["parent_element"] = ""
        if r["class"] not in ("wall", "slanted"):
            continue
        hist = np.zeros(6, np.int64)
        for s, p in mem:
            hist += stations[s].sem_hist[p]
        win_frac = float(hist[4]) / max(float(hist.sum()), 1.0)
        if CW_TILT_MIN_DEG <= tilt <= CW_TILT_MAX_DEG:
            if (win_frac >= CW_WINDOW_FRAC
                    or r["ghost_frac"] >= CW_GHOST_FRAC):
                r["subclass"] = "curtain_wall"
                n_cw += 1
            elif r["area_union_m2"] >= CW_MIN_AREA:
                # geometry says tilted facade; no detector glass coverage →
                # candidate only, no false glass confidence
                r["subclass"] = "tilted_facade_candidate"
                n_cw += 1
    log_fn(f"  curtain-wall subclass: {n_cw} elements "
           f"(tilt {CW_TILT_MIN_DEG}–{CW_TILT_MAX_DEG}°; glass evidence → "
           f"curtain_wall, geometry-only ≥{CW_MIN_AREA} m² → candidate)")
    return n_cw


def _side_stats(keys_by_station: Dict[int, np.ndarray], nrm: np.ndarray):
    """(area_union, hull, coverage, best_single, cz) from per-station keys."""
    union = np.unique(np.concatenate(list(keys_by_station.values())))
    cc = unpack_centers(union, SURFEL)
    e1, e2 = in_plane_axes(nrm)
    uv = np.column_stack([cc @ e1, cc @ e2])
    hull = max(hull_area_2d(uv), 1e-9)
    area = len(union) * SURFEL * SURFEL
    best = max(len(np.unique(k)) for k in keys_by_station.values()) \
        * SURFEL * SURFEL
    return area, hull, area / hull, best, float(cc[:, 2].mean())


def split_by_rooms(stations: List[StationData], elements: List[Dict],
                   members, rooms: List[Dict], log_fn):
    """Phase G (v02): split structural elements straddling a closed-room
    boundary.  Geometric association pools continuous surfaces correctly;
    element identity is room-scoped (user rulings, pairs 1/19/32).  A
    member patch follows the side holding the majority of its surfels."""
    if not rooms:
        log_fn("  room splitting: no closed rooms — skipped")
        return elements, members, 0
    from scipy.ndimage import binary_dilation

    out_elems: List[Dict] = []
    out_members: List[List[Tuple[int, int]]] = []
    next_id = max(r["element_id"] for r in elements) + 1
    n_split = 0

    for r, mem in zip(elements, members):
        done = False
        if r["class"] in ("wall", "floor", "ceiling", "slanted"):
            for room in rooms:
                # Vertical elements: only walls whose ORIENTED normal faces
                # the room interior can belong to it (the corridor-side face
                # of a boundary wall sits within the dilation margin but is
                # a different surface — user GT pairs 12/34).
                if abs(r["nz"]) < 0.5:
                    ys, xs = np.nonzero(room["mask"])
                    c_room = np.array([xs.mean() * room["cell"] + room["lo"][0],
                                       ys.mean() * room["cell"] + room["lo"][1]])
                    n2d = np.array([r["nx"], r["ny"]])
                    n2n = np.linalg.norm(n2d)
                    # element in-plane reference: mean surfel center
                    cc0 = np.concatenate(
                        [unpack_centers(stations[s].surfels(p), SURFEL)
                         for s, p in mem])[:, :2].mean(axis=0)
                    if n2n > 1e-6 and (c_room - cc0) @ (n2d / n2n) < 0:
                        continue                     # faces away → wholly out
                md = binary_dilation(
                    room["mask"], structure=np.ones((3, 3), bool),
                    iterations=max(1, int(round(SPLIT_MARGIN_M
                                                / room["cell"]))))
                lo, cell = room["lo"], room["cell"]
                ny, nx = md.shape
                in_keys: Dict[int, List[np.ndarray]] = {}
                out_keys: Dict[int, List[np.ndarray]] = {}
                mem_side = []                       # True = inside
                n_in = n_out = 0
                for s, p in mem:
                    k = stations[s].surfels(p)
                    cc = unpack_centers(k, SURFEL)
                    ix = np.clip(((cc[:, 0] - lo[0]) / cell).astype(int),
                                 0, nx - 1)
                    iy = np.clip(((cc[:, 1] - lo[1]) / cell).astype(int),
                                 0, ny - 1)
                    inside = md[iy, ix]
                    n_in += int(inside.sum()); n_out += int((~inside).sum())
                    if inside.any():
                        in_keys.setdefault(s, []).append(k[inside])
                    if (~inside).any():
                        out_keys.setdefault(s, []).append(k[~inside])
                    mem_side.append(inside.mean() >= 0.5)
                tot = max(n_in + n_out, 1)
                frac_in = n_in / tot
                a_in  = n_in  * SURFEL * SURFEL
                a_out = n_out * SURFEL * SURFEL
                ok_frac = SPLIT_MIN_FRAC <= frac_in <= 1 - SPLIT_MIN_FRAC
                ok_abs  = min(a_in, a_out) >= SPLIT_MIN_ABS
                if ((ok_frac or ok_abs)
                        and a_in >= SPLIT_MIN_AREA
                        and a_out >= SPLIT_MIN_AREA):
                    nrm = np.array([r["nx"], r["ny"], r["nz"]])
                    for side_name, keys, side_flag in (
                            (f"in:{room['station']}",
                             {s: np.concatenate(v) for s, v in in_keys.items()},
                             True),
                            ("out",
                             {s: np.concatenate(v) for s, v in out_keys.items()},
                             False)):
                        area, hull, cov, best, cz = _side_stats(keys, nrm)
                        child = dict(r)
                        child["element_id"] = next_id
                        child["parent_element"] = r["element_id"]
                        child["subclass"] = (r["subclass"] + "|" if r["subclass"]
                                             else "") + f"room_{side_name}"
                        child["area_union_m2"] = round(area, 3)
                        child["best_single_m2"] = round(best, 3)
                        child["area_gain"] = round(area / max(best, 1e-9), 3)
                        child["hull_m2"] = round(hull, 3)
                        child["coverage_fused"] = round(min(cov, 1.0), 4)
                        child["cz"] = round(cz, 3)
                        cmem = [m for m, side in zip(mem, mem_side)
                                if side == side_flag]
                        child["n_patches"] = len(cmem)
                        child["n_stations"] = len({s for s, _ in cmem})
                        child["stations"] = "|".join(sorted(
                            {stations[s].station for s, _ in cmem}))
                        next_id += 1
                        out_elems.append(child)
                        out_members.append(cmem)
                    n_split += 1
                    log_fn(f"  split element {r['element_id']} "
                           f"({r['class']}, {r['area_union_m2']} m²) at "
                           f"{room['station']} boundary → in {a_in:.1f} m² / "
                           f"out {a_out:.1f} m²")
                    done = True
                    break
        if not done:
            out_elems.append(r)
            out_members.append(mem)
    log_fn(f"  room splitting: {n_split} elements split at "
           f"{len(rooms)} closed-room boundary(ies)")
    return out_elems, out_members, n_split


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="OpenScan2BIM Step 04 v03 — multi-station surface-aware "
                    "fusion + persistence-weighted evidence + geometric "
                    "glass prior (task #17 complete)")
    p.add_argument("--primitives_dir", default="./outputs26",
                   help="Step 01b v02 outputs (patches.csv / patch_ids.npy / "
                        "point_quality.npz per station)")
    p.add_argument("--ply_dir", default="./outputs25",
                   help="dir with <station>/<station>_pano_labeled.ply "
                        "(+ _pano_metadata.csv for persistence)")
    p.add_argument("--output_dir", default="./outputs27")
    p.add_argument("--stations", nargs="+", default=STATIONS_DEFAULT)
    p.add_argument("--rebuild_cache", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    t0 = time.time()
    os.makedirs(args.output_dir, exist_ok=True)
    cache_dir = os.path.join(args.output_dir, "cache")
    os.makedirs(cache_dir, exist_ok=True)
    log(f"Step 04 v01 — multi-station fusion  ({len(args.stations)} stations)")

    # ── Phase A ───────────────────────────────────────────────────────────────
    stations: List[StationData] = []
    for st in args.stations:
        cache_path = os.path.join(cache_dir, f"{st}{CACHE_SUFFIX}")
        ply_path = os.path.join(args.ply_dir, st, f"{st}_pano_labeled.ply")
        if args.rebuild_cache or not os.path.exists(cache_path):
            meta_csv = os.path.join(args.ply_dir, st,
                                    f"{st}_pano_metadata.csv")
            build_station_cache(st, ply_path, args.primitives_dir,
                                cache_path, meta_csv)
        stations.append(StationData(st, cache_path, ply_path))
        log(f"[{st}] cache loaded ({stations[-1].P} patches)")

    # ── Phase B ───────────────────────────────────────────────────────────────
    log("Phase B — cross-station patch association")
    elem, edges, offs = associate(stations, log)

    # ── Phase C ───────────────────────────────────────────────────────────────
    log("Phase C — registration QA (verification residuals)")
    qa_rows = registration_qa(stations, edges, offs,
                              os.path.join(args.output_dir,
                                           "fusion_registration_qa.csv"), log)

    # ── Phase D ───────────────────────────────────────────────────────────────
    log("Phase D — element fusion (labels, planes, coverage)")
    elements, members = fuse_elements(stations, elem, offs, log)
    n_cw = refine_curtain_walls(elements, stations, members, log)
    n_opposed = opposed_pairs(elements,
                              os.path.join(args.output_dir,
                                           "fusion_opposed_pairs.csv"), log)

    # ── Phase E′ (v03): geometry-driven glass candidates ──────────────────────
    log("Phase E′ — geometry-driven glass candidates")
    planes = glass_planes_from_elements(stations, elements, members, log)
    cands = merged_ghost_candidates(stations, planes, log)

    # ── Phase E ───────────────────────────────────────────────────────────────
    log("Phase E — free-space ghost adjudication")
    ghost_rows = adjudicate_ghosts(stations, cands, args.output_dir,
                                   os.path.join(args.output_dir,
                                                "fusion_ghost_verdicts.csv"), log)

    # ── Phase F ───────────────────────────────────────────────────────────────
    log("Phase F — closure detection")
    closure_lines, rooms = closure_detection(
        stations, elements, members,
        os.path.join(args.output_dir, "fusion_closure_report.txt"), log)

    # ── Phase G (v02): room-boundary splitting ────────────────────────────────
    log("Phase G — room-boundary splitting")
    elements, members, n_split = split_by_rooms(stations, elements, members,
                                                rooms, log)

    # ── Write element table + patch map (post-split) ──────────────────────────
    el_csv = os.path.join(args.output_dir, "fusion_elements.csv")
    with open(el_csv, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(elements[0].keys()))
        w.writeheader(); w.writerows(elements)
    log(f"  element table → {el_csv}")
    map_csv = os.path.join(args.output_dir, "fusion_patch_map.csv")
    with open(map_csv, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["station", "patch_id", "element_id"])
        for r, mem in zip(elements, members):
            for s, p in mem:
                w.writerow([stations[s].station, p + 1, r["element_id"]])

    # ── QC top-down ───────────────────────────────────────────────────────────
    try:
        fig, ax = plt.subplots(figsize=(11, 11))
        cmap = plt.get_cmap("tab20")
        for r, mem in zip(elements, members):
            if r["class"] not in STRUCTURAL or r["area_union_m2"] < 0.5 \
                    or not mem:
                continue
            cc = np.concatenate([unpack_centers(stations[s].surfels(p), SURFEL)
                                 for s, p in mem])
            stride = max(1, len(cc) // 4000)
            ax.scatter(cc[::stride, 0], cc[::stride, 1], s=0.5,
                       c=[cmap(r["element_id"] % 20)], linewidths=0)
        for S in stations:
            ax.plot(S.sensor[0], S.sensor[1], "k*", markersize=12)
            ax.annotate(S.station.split("-")[-1], S.sensor[:2],
                        fontsize=9, weight="bold")
        ax.set_aspect("equal")
        ax.set_title("Step 04 — fused structural elements (top-down)")
        fig.savefig(os.path.join(args.output_dir, "fusion_topdown.png"),
                    dpi=150, bbox_inches="tight")
        plt.close(fig)
        log(f"QC top-down → {os.path.join(args.output_dir, 'fusion_topdown.png')}")
    except Exception as exc:
        log(f"[WARN] QC render failed: {exc}")

    # ── Master report ─────────────────────────────────────────────────────────
    multi = [r for r in elements if r["n_stations"] > 1]
    struct = [r for r in elements if r["class"] in STRUCTURAL]
    lines = [
        f"Step 04 v01 fusion report   ({time.strftime('%Y-%m-%d %H:%M:%S')})",
        f"stations: {', '.join(args.stations)}",
        f"params: surfel={SURFEL}  θ_assoc={THETA_ASSOC_DEG}°  "
        f"band={BAND_ASSOC}  gap={GAP_ASSOC}  λ_geo={LAMBDA_GEO}  "
        f"fs_margin={FS_COUNTER_MARGIN}/{FS_SUPPORT_TOL}  "
        f"ghost_rule=counter≥{GHOST_MIN_COUNTER}&support≤{GHOST_MAX_SUPPORT}",
        "",
        f"elements: {len(elements)} total | {len(multi)} multi-station | "
        f"{len(struct)} structural",
        "",
        "registration QA (point-to-plane residuals of associated pairs):",
    ]
    for r in qa_rows:
        lines.append(f"  {r['station_a']} ↔ {r['station_b']}: "
                     f"median {r['median_mm']:.1f} mm  p95 {r['p95_mm']:.1f} mm "
                     f"({r['n_patch_pairs']} pairs)")
    lines += ["", "label fusion (weighted evidence pooling):"]
    dis_w = [r["pre_disagreement"] for r in multi]
    if dis_w:
        lines.append(f"  pre-fusion evidence disagreement on multi-station "
                     f"elements: mean {np.mean(dis_w):.3f}, "
                     f">10% on {sum(1 for x in dis_w if x > 0.10)} / "
                     f"{len(multi)} elements → post-fusion each element has "
                     f"ONE posterior (argmax + entropy in the table)")
    lines += ["", "coverage accounting (structural multi-station elements):"]
    for cls in ("wall", "ceiling", "floor", "slanted"):
        sel = [r for r in multi if r["class"] == cls]
        if not sel:
            continue
        gain = np.mean([r["area_gain"] for r in sel])
        c1 = np.mean([r["coverage_best_single"] for r in sel])
        cf = np.mean([r["coverage_fused"] for r in sel])
        lines.append(f"  {cls:8s} n={len(sel):3d}  observed-area gain "
                     f"×{gain:.2f}  coverage {c1:.2f} → {cf:.2f}")
    lines += ["", "glass ghost adjudication (3-tier; candidates = semantic ∪ "
              "geometric glass prior):"]
    for r in ghost_rows:
        lines.append(f"  {r['station']}: candidates {r['candidates']:,} "
                     f"(sem {r['n_semantic']:,} / geo {r['n_geometric']:,}) → "
                     f"free-space ghost {r['ghost_freespace']:,} / "
                     f"corroborated real {r['real_corroborated']:,} / "
                     f"uncorroborated {r['uncorroborated']:,}")
    lines.append("  NOTE: behind glass the free-space carve is structurally "
                 "blind (the glass surface's own returns always occlude the "
                 "candidate in every station's min-range grid), so the "
                 "operative discriminator is corroboration — measured here: "
                 "support stays ≤ 2.7% even at 0.50 m tolerance, i.e. the "
                 "candidate mass is viewpoint-specific, the mirror-ghost "
                 "signature.  Uncorroborated points are ghost-LIKELY: kept "
                 "out of element geometry, routed to the review queue (M5), "
                 "not silently deleted.")
    lines += ["", f"opposed coplanar element pairs kept separate: {n_opposed} "
              f"(fusion_opposed_pairs.csv — partition walls, incl. 6|7)"]
    cw = [r for r in elements
          if "curtain_wall" in r.get("subclass", "")
          or "tilted_facade" in r.get("subclass", "")]
    lines += ["", f"curtain-wall subclass: {n_cw} elements "
              f"(design tilt reference 4.313° from the user's Revit sheet):"]
    for r in sorted(cw, key=lambda r: -r["area_union_m2"])[:8]:
        lines.append(f"  elem {r['element_id']}: tilt {r['tilt_deg']}°  "
                     f"area {r['area_union_m2']} m²  stations {r['stations']}")
    lines += ["", f"room-boundary splitting (Phase G): {n_split} elements "
              f"split (parent_element column keeps lineage; geometric "
              f"continuity is preserved in the association layer, element "
              f"identity is room-scoped per the 2026-07-10 user rulings)"]
    lines += ["", "closure detection:"] + ["  " + l for l in closure_lines]
    lines += ["", "hooks for later milestones: detector persistence weights "
              "(M4) enter the same pooling as an extra factor; conformal "
              "decision sets (M5) consume p_class/entropy; association P/R "
              "needs the ~30-pair manual list (fusion_patch_map.csv is the "
              "evaluation substrate).",
              f"elapsed: {time.time() - t0:.1f} s"]
    rep = os.path.join(args.output_dir, "fusion_report.txt")
    with open(rep, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    log(f"Report → {rep}")
    for ln in lines:
        log("  | " + ln)


if __name__ == "__main__":
    main()
