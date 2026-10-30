"""Panorama projection helpers shared by the view stages (s06a, s06b).

Verbatim from OpenScan2BIM_Step04c_PanoramaProjection01.py (2026-10-30
consolidation): only the three functions the current chain calls are kept.
"""
from math import pi
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np


def read_sensor(e57_path: Path) -> np.ndarray:
    import pye57
    e57 = pye57.E57(str(e57_path))
    header = e57.get_header(0)
    return np.asarray(header.translation, dtype=np.float64)


def pano_pixels(points: np.ndarray, sensor: np.ndarray, width: int, height: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    relative = points.astype(np.float64, copy=False) - sensor[None, :]
    ranges = np.linalg.norm(relative, axis=1)
    ranges = np.maximum(ranges, 1e-9)
    theta = np.arctan2(relative[:, 1], relative[:, 0])
    phi = np.arcsin(np.clip(relative[:, 2] / ranges, -1.0, 1.0))
    u = np.clip(((theta + pi) / (2 * pi) * width).astype(np.int32), 0, width - 1)
    v = np.clip(((pi / 2 - phi) / pi * height).astype(np.int32), 0, height - 1)
    return u, v, ranges.astype(np.float32)


def point_indices_for_patches(order: np.ndarray, starts: np.ndarray,
                              patch_ids: Iterable[int]) -> np.ndarray:
    chunks = [order[starts[patch_id]:starts[patch_id + 1]] for patch_id in patch_ids]
    if not chunks or any(len(chunk) == 0 for chunk in chunks):
        raise ValueError("One or more member atomic patches have no aligned source points")
    return np.concatenate(chunks)
