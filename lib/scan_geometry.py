"""Scan-grid, voxel-key, union-find and PLY helpers shared by s03 and s05a.

Verbatim from OpenScan2BIM_Step04_Fusion03.py (2026-10-30 consolidation): only
the definitions the current chain calls are kept. The rest of Fusion03 (the
v4 seven-station fusion) is not part of this pipeline.
"""
import numpy as np
from scipy.ndimage import minimum_filter
from plyfile import PlyData

# Phase A — cache
SURFEL            = 0.10          # m — surfel cell size (element-level areas)
FS_GRID_H         = 1024          # free-space min-range grid height (W = 2H)

_BITS             = 21            # packed 3D key bits (same scheme as Step01b)
_KOFF             = 4096          # index offset → strictly positive indices


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
