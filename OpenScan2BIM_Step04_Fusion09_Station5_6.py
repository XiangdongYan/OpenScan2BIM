"""OpenScan2BIM Step04 v09 -- selected-gate association for stations 5 and 6.

This first v04 pass deliberately owns only the new association substrate.
Step01b v04 atomic patches are provenance observations.  Step01c v12 already
reassembles them into station-local surfaces; this script associates those
surfaces across registered stations.  It does not yet pool material evidence,
author IFC, or revise a final BIM class.

The station-local 0.25 m proposal band is NOT re-used here. Cross-station
delta_perp is fixed at 0.15 m and normal-angle at 7 degrees. This run evaluates
only the historical GT pairs whose two stations are present.
"""
import argparse
import csv
import os
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from OpenScan2BIM_Step04_Fusion03 import (
    SURFEL,
    UnionFind,
    load_ply_xyz_sem,
    pack_keys,
    unpack_centers,
)


STATIONS = ["2ndLab-5", "2ndLab-6"]
PANO_DIR = "./outputs18"
ATOMIC_DIR = "./outputs49_step01b_atomic04_7station"
SURFACE_DIR = "./outputs55_step01c_local_surfaces12_7station"
LEGACY_CACHE_DIR = "./outputs20/cache"
ASSOC_GT_XLSX = "./assoc_eval/assoc_eval_pairs.xlsx"
OUT_DIR = "./outputs62_fusion09_station5_6"

THETA_SWEEP_DEG = (7.0,)
DELTA_ASSOC_M = 0.15
G_PARALLEL_VERTICAL_M = 0.60
G_PARALLEL_HORIZONTAL_M = 2.00
HORIZ_NZ = np.cos(np.deg2rad(15.0))
REP_MAX = 3000
GAP_REP_MAX = 400
GT_TRANSFER_MIN = 0.50


def log(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def parse_args():
    parser = argparse.ArgumentParser(
        description="OpenScan2BIM Step04 v04 surface-level association sweep"
    )
    parser.add_argument("--stations", nargs="+", default=STATIONS)
    parser.add_argument("--pano_dir", default=PANO_DIR)
    parser.add_argument("--atomic_dir", default=ATOMIC_DIR)
    parser.add_argument("--surface_dir", default=SURFACE_DIR)
    parser.add_argument("--legacy_cache_dir", default=LEGACY_CACHE_DIR)
    parser.add_argument("--assoc_gt_xlsx", default=ASSOC_GT_XLSX)
    parser.add_argument("--output_dir", default=OUT_DIR)
    return parser.parse_args()


def read_csv_rows(path: str) -> List[Dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def plane_from_points(points: np.ndarray, sensor: np.ndarray) -> Tuple[np.ndarray, float, np.ndarray]:
    """Fit an oriented local-surface plane from deterministic representative points."""
    centroid = points.mean(axis=0, dtype=np.float64)
    centered = points.astype(np.float64) - centroid
    _, vec = np.linalg.eigh(centered.T @ centered / max(len(points), 1))
    normal = vec[:, 0]
    if float(normal @ (sensor - centroid)) < 0:
        normal *= -1.0
    return normal, -float(normal @ centroid), centroid


def normal_angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    dot = float(np.clip(a @ b, -1.0, 1.0))
    return float(np.rad2deg(np.arccos(dot)))


def plane_axes(normal: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    anchor = np.array([0.0, 0.0, 1.0])
    if abs(float(normal @ anchor)) > 0.9:
        anchor = np.array([1.0, 0.0, 0.0])
    axis_u = np.cross(normal, anchor)
    axis_u /= np.linalg.norm(axis_u)
    axis_v = np.cross(normal, axis_u)
    return axis_u, axis_v


def projected_gap(points_a: np.ndarray, points_b: np.ndarray,
                  normal_a: np.ndarray, normal_b: np.ndarray,
                  origin: np.ndarray) -> float:
    """Shortest distance between representative footprints in a common plane."""
    normal = normal_a + normal_b
    normal /= np.linalg.norm(normal)
    axis_u, axis_v = plane_axes(normal)
    stride_a = max(1, len(points_a) // GAP_REP_MAX)
    stride_b = max(1, len(points_b) // GAP_REP_MAX)
    a = points_a[::stride_a][:GAP_REP_MAX].astype(np.float64) - origin
    b = points_b[::stride_b][:GAP_REP_MAX].astype(np.float64) - origin
    uv_a = np.column_stack([a @ axis_u, a @ axis_v])
    uv_b = np.column_stack([b @ axis_u, b @ axis_v])
    return float(cKDTree(uv_b).query(uv_a, k=1)[0].min())


def save_npz(path: str, **arrays) -> None:
    try:
        np.savez_compressed(path, **arrays)
    except OSError:
        if os.path.exists(path):
            os.remove(path)
        time.sleep(1.0)
        np.savez_compressed(path, **arrays)


@dataclass
class SurfaceStation:
    station: str
    sensor: np.ndarray
    surface_id: np.ndarray
    n_atomic_patches: np.ndarray
    n_points: np.ndarray
    area: np.ndarray
    mean_w: np.ndarray
    normal: np.ndarray
    d: np.ndarray
    centroid: np.ndarray
    orientation: np.ndarray
    elevation_band: np.ndarray
    _surf_keys: np.ndarray
    _surf_indptr: np.ndarray
    _reps: np.ndarray
    _rep_indptr: np.ndarray

    @property
    def S(self) -> int:
        return len(self.surface_id)

    def surfels(self, index: int) -> np.ndarray:
        return self._surf_keys[self._surf_indptr[index]:self._surf_indptr[index + 1]]

    def reps(self, index: int) -> np.ndarray:
        return self._reps[self._rep_indptr[index]:self._rep_indptr[index + 1]]


def build_surface_cache(station: str, pano_dir: str, atomic_dir: str,
                        surface_dir: str, cache_path: str) -> None:
    """Aggregate Step01b v04 point membership by the Step01c surface map."""
    if os.path.exists(cache_path):
        log(f"[{station}] cache exists; reuse {cache_path}")
        return
    log(f"[{station}] building surface cache")
    t0 = time.time()
    surface_rows = read_csv_rows(os.path.join(surface_dir, f"{station}_surfaces.csv"))
    map_rows = read_csv_rows(os.path.join(surface_dir, f"{station}_patch_to_surface.csv"))
    patch = np.load(os.path.join(atomic_dir, f"{station}_patch_ids.npy"), mmap_mode="r")
    quality = np.load(os.path.join(atomic_dir, f"{station}_point_quality.npz"))
    sensor = quality["sensor_pos"].astype(np.float64)
    point_weight = quality["w_uint8"].astype(np.float32) / 255.0
    points, semantic = load_ply_xyz_sem(
        os.path.join(pano_dir, station, f"{station}_pano_labeled.ply"),
        want_instance=False,
    )
    if len(points) != len(patch) or len(points) != len(point_weight):
        raise ValueError(f"[{station}] PLY, atomic patch, and quality arrays are not aligned")

    surface_ids = np.array([int(row["surface_id"]) for row in surface_rows], dtype=np.int32)
    if not np.array_equal(surface_ids, np.arange(1, len(surface_ids) + 1)):
        raise ValueError(f"[{station}] local surface ids must be contiguous from 1")
    atomic_max = int(np.max(patch))
    patch_to_surface = np.zeros(atomic_max + 1, dtype=np.int32)
    for row in map_rows:
        patch_to_surface[int(row["patch_id"])] = int(row["surface_id"])
    if np.any(patch_to_surface[1:] == 0):
        raise ValueError(f"[{station}] an atomic patch is missing a local-surface mapping")

    surface_of_point = patch_to_surface[np.asarray(patch, dtype=np.int32)]
    valid = surface_of_point > 0
    order = np.argsort(surface_of_point[valid], kind="stable")
    point_indices = np.flatnonzero(valid)[order]
    sorted_surface = surface_of_point[valid][order]
    starts = np.searchsorted(sorted_surface, np.arange(1, len(surface_ids) + 2))
    if np.any(np.diff(starts) == 0):
        raise ValueError(f"[{station}] a local surface has no source point")

    surface_key_sets: List[np.ndarray] = []
    representative_sets: List[np.ndarray] = []
    normals = np.zeros((len(surface_ids), 3), dtype=np.float64)
    planes_d = np.zeros(len(surface_ids), dtype=np.float64)
    centroids = np.zeros((len(surface_ids), 3), dtype=np.float64)
    n_points = np.diff(starts).astype(np.int64)
    mean_w = np.zeros(len(surface_ids), dtype=np.float32)
    keys_all = pack_keys(points, SURFEL)
    for index in range(len(surface_ids)):
        ids = point_indices[starts[index]:starts[index + 1]]
        stride = max(1, len(ids) // REP_MAX)
        representatives = points[ids[::stride][:REP_MAX]]
        normal, d, centroid = plane_from_points(representatives, sensor)
        normals[index], planes_d[index], centroids[index] = normal, d, centroid
        surface_key_sets.append(np.unique(keys_all[ids]))
        representative_sets.append(representatives)
        mean_w[index] = float(point_weight[ids].mean())

    surface_df = pd.DataFrame(surface_rows)
    member_count = surface_df["n_patches_merged"].astype(np.int32).to_numpy()
    area = surface_df["area_m2"].astype(np.float64).to_numpy()
    orientation_map = {"horizontal": 1, "vertical": 2, "oblique": 3}
    elevation_map = {"NA": 0, "unresolved": 1, "floor_reference": 2,
                     "near_upper_reference": 3, "intermediate": 4}
    orientation = surface_df["planar_orientation"].map(orientation_map).fillna(0).astype(np.uint8).to_numpy()
    elevation = surface_df["elevation_band"].map(elevation_map).fillna(0).astype(np.uint8).to_numpy()
    save_npz(
        cache_path,
        sensor=sensor,
        surface_id=surface_ids,
        n_atomic_patches=member_count,
        n_points=n_points,
        area=area,
        mean_w=mean_w,
        normal=normals,
        d=planes_d,
        centroid=centroids,
        orientation=orientation,
        elevation_band=elevation,
        surf_keys=np.concatenate(surface_key_sets),
        surf_indptr=np.cumsum([0] + [len(v) for v in surface_key_sets]).astype(np.int64),
        rep_pts=np.concatenate(representative_sets),
        rep_indptr=np.cumsum([0] + [len(v) for v in representative_sets]).astype(np.int64),
    )
    log(f"[{station}] cache: {len(surface_ids)} surfaces from {atomic_max} atomic patches "
        f"in {time.time() - t0:.1f}s")


def load_surface_station(station: str, cache_path: str) -> SurfaceStation:
    cache = np.load(cache_path)
    return SurfaceStation(
        station=station,
        sensor=cache["sensor"],
        surface_id=cache["surface_id"],
        n_atomic_patches=cache["n_atomic_patches"],
        n_points=cache["n_points"],
        area=cache["area"],
        mean_w=cache["mean_w"],
        normal=cache["normal"],
        d=cache["d"],
        centroid=cache["centroid"],
        orientation=cache["orientation"],
        elevation_band=cache["elevation_band"],
        _surf_keys=cache["surf_keys"],
        _surf_indptr=cache["surf_indptr"],
        _reps=cache["rep_pts"],
        _rep_indptr=cache["rep_indptr"],
    )


def surface_candidates(stations: List[SurfaceStation]) -> List[Dict]:
    """Return same-side candidates passing delta/gap gates at the widest angle."""
    max_theta = max(THETA_SWEEP_DEG)
    cos_limit = np.cos(np.deg2rad(max_theta))
    edges: List[Dict] = []
    for a, station_a in enumerate(stations):
        for b in range(a + 1, len(stations)):
            station_b = stations[b]
            dot = station_a.normal @ station_b.normal.T
            # Planes are oriented toward their own scanner. Opposite faces are
            # therefore excluded before the numerical angular sweep is applied.
            oriented_normal_ok = dot > 0.0
            angle_ok = dot >= cos_limit
            off_ab = np.abs(station_a.centroid @ station_b.normal.T + station_b.d[None, :])
            off_ba = np.abs((station_b.centroid @ station_a.normal.T).T + station_a.d[:, None])
            delta = np.maximum(off_ab, off_ba)
            ii, jj = np.nonzero(oriented_normal_ok & angle_ok & (delta <= DELTA_ASSOC_M))
            for i, j in zip(ii.tolist(), jj.tolist()):
                horizontal_pair = (abs(station_a.normal[i, 2]) >= HORIZ_NZ and
                                   abs(station_b.normal[j, 2]) >= HORIZ_NZ)
                gap_limit = G_PARALLEL_HORIZONTAL_M if horizontal_pair else G_PARALLEL_VERTICAL_M
                gap = projected_gap(
                    station_a.reps(i), station_b.reps(j),
                    station_a.normal[i], station_b.normal[j],
                    (station_a.centroid[i] + station_b.centroid[j]) / 2.0,
                )
                if gap > gap_limit:
                    continue
                angle = normal_angle_deg(station_a.normal[i], station_b.normal[j])
                edge = {
                    "station_a": station_a.station,
                    "surface_a": int(station_a.surface_id[i]),
                    "station_b": station_b.station,
                    "surface_b": int(station_b.surface_id[j]),
                    "normal_dot": round(float(dot[i, j]), 6),
                    "normal_angle_deg": round(angle, 6),
                    "delta_perp_m": round(float(delta[i, j]), 4),
                    "g_parallel_m": round(gap, 4),
                    "gap_limit_m": gap_limit,
                    "horizontal_pair": int(horizontal_pair),
                    "min_theta_sweep_deg": next(value for value in THETA_SWEEP_DEG if angle <= value),
                }
                edges.append(edge)
    return sorted(edges, key=lambda row: (row["normal_angle_deg"], row["delta_perp_m"], row["g_parallel_m"],
                                           row["station_a"], row["surface_a"]))


def global_offsets(stations: List[SurfaceStation]) -> np.ndarray:
    return np.cumsum([0] + [station.S for station in stations])


def union_for_theta(stations: List[SurfaceStation], candidates: List[Dict],
                    theta: float) -> Tuple[np.ndarray, np.ndarray, int]:
    offsets = global_offsets(stations)
    lookup = {(station.station, int(surface_id)): int(offsets[s] + i)
              for s, station in enumerate(stations)
              for i, surface_id in enumerate(station.surface_id)}
    union = UnionFind(int(offsets[-1]))
    n_edges = 0
    for edge in candidates:
        if edge["normal_angle_deg"] > theta:
            continue
        union.union(lookup[(edge["station_a"], edge["surface_a"])],
                    lookup[(edge["station_b"], edge["surface_b"])])
        n_edges += 1
    root = np.array([union.find(i) for i in range(int(offsets[-1]))], dtype=np.int64)
    _, element = np.unique(root, return_inverse=True)
    return element + 1, offsets, n_edges


def transfer_legacy_pairs(stations: List[SurfaceStation], legacy_cache_dir: str,
                          gt_xlsx: str) -> pd.DataFrame:
    """Map old raw patches to new local surfaces by shared world-voxel keys."""
    pairs = pd.read_excel(gt_xlsx, sheet_name="Pairs")
    station_by_name = {station.station: station for station in stations}
    pairs = pairs[pairs["station_a"].isin(station_by_name) &
                  pairs["station_b"].isin(station_by_name)].copy()
    requested = {}
    for _, row in pairs.iterrows():
        requested.setdefault(str(row["station_a"]), set()).add(int(row["patch_a"]))
        requested.setdefault(str(row["station_b"]), set()).add(int(row["patch_b"]))
    transfers = {}
    for name, patch_ids in requested.items():
        station = station_by_name[name]
        all_keys = np.concatenate([station.surfels(i) for i in range(station.S)])
        all_surface = np.concatenate([
            np.full(len(station.surfels(i)), station.surface_id[i], dtype=np.int32)
            for i in range(station.S)
        ])
        order = np.argsort(all_keys, kind="stable")
        keys_sorted, surface_sorted = all_keys[order], all_surface[order]
        unique_keys, starts, counts = np.unique(keys_sorted, return_index=True,
                                                return_counts=True)
        # A 10 cm world voxel shared by multiple local surfaces is not a
        # discriminating provenance key for GT transfer, so omit it.
        unique_surface = surface_sorted[starts]
        unambiguous = counts == 1
        keys_sorted = unique_keys[unambiguous]
        surface_sorted = unique_surface[unambiguous]
        legacy = np.load(os.path.join(legacy_cache_dir, f"{name}_fusion_cache_v3.npz"))
        old_keys, old_indptr = legacy["surf_keys"], legacy["surf_indptr"]
        for patch_id in patch_ids:
            if patch_id < 1 or patch_id >= len(old_indptr):
                transfers[(name, patch_id)] = (0, 0.0, 0)
                continue
            keys = old_keys[old_indptr[patch_id - 1]:old_indptr[patch_id]]
            pos = np.searchsorted(keys_sorted, keys)
            hit = pos < len(keys_sorted)
            hit[hit] &= keys_sorted[pos[hit]] == keys[hit]
            matched = surface_sorted[pos[hit]]
            if not len(matched):
                transfers[(name, patch_id)] = (0, 0.0, 0)
                continue
            values, counts = np.unique(matched, return_counts=True)
            best = int(np.argmax(counts))
            transfers[(name, patch_id)] = (int(values[best]), float(counts[best] / len(keys)), int(counts[best]))

    rows = []
    for _, pair in pairs.iterrows():
        a = (str(pair["station_a"]), int(pair["patch_a"]))
        b = (str(pair["station_b"]), int(pair["patch_b"]))
        surface_a, coverage_a, hits_a = transfers[a]
        surface_b, coverage_b, hits_b = transfers[b]
        rows.append({
            "pair_id": int(pair["pair_id"]),
            "station_a": a[0], "legacy_patch_a": a[1], "surface_a": surface_a,
            "transfer_coverage_a": round(coverage_a, 3), "shared_voxels_a": hits_a,
            "station_b": b[0], "legacy_patch_b": b[1], "surface_b": surface_b,
            "transfer_coverage_b": round(coverage_b, 3), "shared_voxels_b": hits_b,
            "same_surface_gt": str(pair["same_surface"]).lower(),
            "transfer_eligible": int(surface_a > 0 and surface_b > 0 and
                                     coverage_a >= GT_TRANSFER_MIN and coverage_b >= GT_TRANSFER_MIN),
        })
    return pd.DataFrame(rows)


def element_rgb(element_id: int) -> Tuple[int, int, int]:
    """Stable identity colour for visual inspection only."""
    value = (element_id * 2654435761) & 0xFFFFFFFF
    return (48 + ((value >> 0) & 0x7F),
            48 + ((value >> 8) & 0x7F),
            48 + ((value >> 16) & 0x7F))


def write_element_surfel_ply(stations: List[SurfaceStation], element: np.ndarray,
                              offsets: np.ndarray, theta: float,
                              output_dir: str) -> str:
    """Write a compact, station-preserving visualisation of temporary elements.

    Coordinates are 10 cm world-surfel centres rather than every source point.
    This avoids duplicating the full seven-station PLY while preserving the
    station/surface/element lineage needed for CloudCompare inspection.
    """
    total = sum(len(station.surfels(index))
                for station in stations for index in range(station.S))
    record = np.empty(total, dtype=np.dtype([
        ("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
        ("red", "u1"), ("green", "u1"), ("blue", "u1"),
        ("station_id", "u1"), ("surface_id", "<u4"),
        ("element_id", "<u4"), ("n_atomic_patches", "<u2"),
    ]))
    cursor = 0
    for station_index, (station, offset) in enumerate(zip(stations, offsets[:-1]), start=1):
        for index, surface_id in enumerate(station.surface_id):
            centres = unpack_centers(station.surfels(index), SURFEL)
            count = len(centres)
            element_id = int(element[offset + index])
            red, green, blue = element_rgb(element_id)
            out = record[cursor:cursor + count]
            out["x"], out["y"], out["z"] = centres[:, 0], centres[:, 1], centres[:, 2]
            out["red"], out["green"], out["blue"] = red, green, blue
            out["station_id"] = station_index
            out["surface_id"] = int(surface_id)
            out["element_id"] = element_id
            out["n_atomic_patches"] = int(station.n_atomic_patches[index])
            cursor += count
    if cursor != total:
        raise AssertionError("surfel PLY record count mismatch")
    name = f"surface_elements_theta_{int(theta):02d}deg.ply"
    path = os.path.join(output_dir, name)
    header = "\n".join([
        "ply", "format binary_little_endian 1.0", f"element vertex {total}",
        "property float x", "property float y", "property float z",
        "property uchar red", "property uchar green", "property uchar blue",
        "property uchar station_id", "property uint surface_id",
        "property uint element_id", "property ushort n_atomic_patches",
        "end_header", "",
    ]).encode("ascii")
    with open(path, "wb") as handle:
        handle.write(header)
        record.tofile(handle)
    return path


def evaluate_sweep(stations: List[SurfaceStation], candidates: List[Dict],
                   transferred_gt: pd.DataFrame, output_dir: str) -> pd.DataFrame:
    lookup = {(station.station, int(surface_id)): offset + index
              for offset, station in zip(global_offsets(stations)[:-1], stations)
              for index, surface_id in enumerate(station.surface_id)}
    rows = []
    for theta in THETA_SWEEP_DEG:
        element, offsets, n_edges = union_for_theta(stations, candidates, theta)
        element_by_surface = {key: int(element[index]) for key, index in lookup.items()}
        map_rows = []
        for station, offset in zip(stations, offsets[:-1]):
            for index, surface_id in enumerate(station.surface_id):
                map_rows.append({"station_id": station.station, "surface_id": int(surface_id),
                                 "element_id": int(element[offset + index])})
        pd.DataFrame(map_rows).to_csv(
            os.path.join(output_dir, f"surface_element_map_theta_{int(theta):02d}deg.csv"),
            index=False,
        )
        write_element_surfel_ply(stations, element, offsets, theta, output_dir)

        gt = transferred_gt.copy()
        gt["predicted_same_element"] = [
            int(element_by_surface[(row.station_a, int(row.surface_a))] ==
                element_by_surface[(row.station_b, int(row.surface_b))])
            if row.transfer_eligible else -1
            for row in gt.itertuples(index=False)
        ]
        gt.to_csv(os.path.join(output_dir, f"gt_predictions_theta_{int(theta):02d}deg.csv"), index=False)
        eligible = gt[gt.transfer_eligible.eq(1)]
        positive = eligible[eligible.same_surface_gt.eq("yes")]
        negative = eligible[eligible.same_surface_gt.eq("no")]
        correct = (eligible.predicted_same_element.eq(eligible.same_surface_gt.eq("yes")))
        n_elements = int(element.max())
        element_stations = [set() for _ in range(n_elements + 1)]
        for station, offset in zip(stations, offsets[:-1]):
            for index in range(station.S):
                element_stations[int(element[offset + index])].add(station.station)
        rows.append({
            "normal_angle_threshold_deg": theta,
            "delta_perp_m": DELTA_ASSOC_M,
            "accepted_surface_edges": n_edges,
            "elements": n_elements,
            "multi_station_elements": sum(len(v) > 1 for v in element_stations),
            "gt_pairs_total": len(gt),
            "gt_pairs_transfer_eligible": len(eligible),
            "gt_pair_accuracy": round(float(correct.mean()), 3) if len(eligible) else np.nan,
            "gt_positive_recall": round(float(positive.predicted_same_element.mean()), 3) if len(positive) else np.nan,
            "gt_negative_specificity": round(float((negative.predicted_same_element == 0).mean()), 3) if len(negative) else np.nan,
        })
    return pd.DataFrame(rows)


def main():
    args = parse_args()
    if os.path.exists(args.output_dir):
        raise FileExistsError(f"Refusing to overwrite existing output directory: {args.output_dir}")
    os.makedirs(os.path.join(args.output_dir, "cache"))
    stations = []
    for station in args.stations:
        cache_path = os.path.join(args.output_dir, "cache", f"{station}_surface_cache_v4.npz")
        build_surface_cache(station, args.pano_dir, args.atomic_dir, args.surface_dir, cache_path)
        stations.append(load_surface_station(station, cache_path))
    log(f"surface inventory: {sum(s.S for s in stations)} local surfaces across {len(stations)} stations")

    candidates = surface_candidates(stations)
    pd.DataFrame(candidates).to_csv(os.path.join(args.output_dir, "surface_association_candidates.csv"), index=False)
    log(f"candidate edges at delta <= {DELTA_ASSOC_M:.2f} m and theta <= {max(THETA_SWEEP_DEG):.1f} deg: {len(candidates)}")

    transferred_gt = transfer_legacy_pairs(stations, args.legacy_cache_dir, args.assoc_gt_xlsx)
    transferred_gt.to_csv(os.path.join(args.output_dir, "legacy_gt_to_surface_transfer.csv"), index=False)
    log(f"legacy GT transfer: {int(transferred_gt.transfer_eligible.sum())}/{len(transferred_gt)} pairs eligible")

    summary = evaluate_sweep(stations, candidates, transferred_gt, args.output_dir)
    summary.to_csv(os.path.join(args.output_dir, "normal_angle_sensitivity.csv"), index=False)
    with open(os.path.join(args.output_dir, "README.txt"), "w", encoding="utf-8") as handle:
        handle.write(
            "Step04 v09 two-station association diagnostic. Input unit is (station_id, surface_id), not raw patch.\n"
            "The selected gate is theta=7 degrees, delta_perp=0.15 m, and projected-gap limits at 0.60 m (vertical) / 2.00 m (horizontal).\n"
            "Plane normals are sensor-oriented: dot(n_a,n_b)>0 is a hard same-side compatibility test, while theta is the numerical co-planarity tolerance.\n"
            "surface_elements_theta_XXdeg.ply contains 10 cm world-surfel centres, RGB=temporary element identity, plus station_id, surface_id, element_id, and n_atomic_patches scalar fields. It is a compact review visualisation, not a deduplicated final point cloud.\n"
            "legacy_gt_to_surface_transfer.csv reports whether each historical raw-patch GT pair mapped to the new local-surface space with >=0.50 voxel-key coverage on both sides.\n"
            "No material fusion, ghost adjudication, final BIM class, room splitting, or IFC authoring occurs in this diagnostic.\n"
        )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
