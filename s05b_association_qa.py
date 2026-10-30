"""s05b -- component-level geometry QA of the s05a association (Layer 1.4).

Pipeline version of OpenScan2BIM_Step04_Fusion07.py (2026-10-30). The station
helpers come from s05a (byte-identical copies of the Fusion06 definitions the
legacy script imported); stations and paths are required arguments; the
historical-GT pair audit is written only when s05a produced its GT table.

Legacy description: OpenScan2BIM Step04 v07 -- component-level geometry QA at
the selected gate.

Consumes the v06 theta=7 degree provisional association.  It does not alter
the association, assign final BIM classes, split rooms, pool visual evidence,
or export an IFC file.  Its purpose is to measure whether every provisional
element remains geometrically explainable by one plane before a component-level
accept/review rule is calibrated.
"""
import argparse
import os
import time
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from s05a_association import (
    SurfaceStation,
    load_surface_station,
    normal_angle_deg,
)


SELECTED_THETA_DEG = 7.0
VALIDATION_REPS_PER_SURFACE = 1000


def log(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="OpenScan2BIM Step04 v07 component-level geometry QA"
    )
    parser.add_argument("--stations", nargs="+", required=True)
    parser.add_argument("--association_dir", required=True, help="s05a output")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--theta_deg", type=float, default=SELECTED_THETA_DEG)
    return parser.parse_args(argv)


def theta_tag(theta: float) -> str:
    if not float(theta).is_integer():
        raise ValueError("This diagnostic expects an integer-valued selected theta")
    return f"{int(theta):02d}deg"


def sample_representatives(points: np.ndarray) -> np.ndarray:
    """Keep every local surface represented equally enough for a stable fit."""
    if len(points) <= VALIDATION_REPS_PER_SURFACE:
        return points.astype(np.float64, copy=False)
    indices = np.linspace(0, len(points) - 1, VALIDATION_REPS_PER_SURFACE, dtype=np.int64)
    return points[indices].astype(np.float64, copy=False)


def weighted_plane(points: np.ndarray, weights: np.ndarray,
                   reference_normal: np.ndarray) -> Tuple[np.ndarray, float, np.ndarray]:
    """Total-least-squares plane; area weights stop tiny fragments dominating it."""
    weights = weights / weights.sum()
    centroid = (points * weights[:, None]).sum(axis=0)
    centered = points - centroid
    covariance = (centered * weights[:, None]).T @ centered
    _, vectors = np.linalg.eigh(covariance)
    normal = vectors[:, 0]
    if float(normal @ reference_normal) < 0.0:
        normal *= -1.0
    return normal, -float(normal @ centroid), centroid


def load_stations(station_names: List[str], association_dir: str) -> List[SurfaceStation]:
    stations = []
    for station in station_names:
        cache = os.path.join(association_dir, "cache", f"{station}_surface_cache_v4.npz")
        if not os.path.exists(cache):
            raise FileNotFoundError(f"Missing v06 cache: {cache}")
        stations.append(load_surface_station(station, cache))
    return stations


def build_member_inventory(stations: List[SurfaceStation], element_map: pd.DataFrame) -> Dict[int, List[Dict]]:
    required = {"station_id", "surface_id", "element_id"}
    if not required.issubset(element_map.columns):
        raise ValueError(f"Element map must contain {sorted(required)}")
    key_to_element = {
        (str(row.station_id), int(row.surface_id)): int(row.element_id)
        for row in element_map.itertuples(index=False)
    }
    expected = sum(station.S for station in stations)
    if len(key_to_element) != expected:
        raise ValueError(f"Element map has {len(key_to_element)} keys; expected {expected}")

    groups: Dict[int, List[Dict]] = defaultdict(list)
    for station in stations:
        for index, surface_id in enumerate(station.surface_id):
            key = (station.station, int(surface_id))
            if key not in key_to_element:
                raise KeyError(f"No element assignment for {key}")
            groups[key_to_element[key]].append({
                "station_id": station.station,
                "surface_id": int(surface_id),
                "index": index,
                "station": station,
            })
    return groups


def validate_components(groups: Dict[int, List[Dict]]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    element_rows = []
    member_rows = []
    for element_id in sorted(groups):
        members = groups[element_id]
        samples, weights, normals = [], [], []
        sample_meta = []
        for member in members:
            station = member["station"]
            index = member["index"]
            sample = sample_representatives(station.reps(index))
            samples.append(sample)
            weights.append(np.full(len(sample), max(float(station.area[index]), 1e-6) / len(sample)))
            normals.append(station.normal[index])
            sample_meta.append((member, sample))

        points = np.concatenate(samples, axis=0)
        area_weights = np.concatenate(weights, axis=0)
        reference_normal = np.sum(np.stack(normals), axis=0)
        reference_normal /= np.linalg.norm(reference_normal)
        normal, d, centroid = weighted_plane(points, area_weights, reference_normal)
        residual = np.abs(points @ normal + d)
        rms = float(np.sqrt(np.average(residual ** 2, weights=area_weights)))

        station_ids = set()
        total_area = 0.0
        quality_area_sum = 0.0
        max_centroid_offset = 0.0
        max_normal_angle = 0.0
        max_surface_rms = 0.0
        for member, sample in sample_meta:
            station = member["station"]
            index = member["index"]
            station_ids.add(member["station_id"])
            area = float(station.area[index])
            total_area += area
            quality_area_sum += area * float(station.mean_w[index])
            surface_residual = np.abs(sample @ normal + d)
            surface_rms = float(np.sqrt(np.mean(surface_residual ** 2)))
            centroid_offset = abs(float(station.centroid[index] @ normal + d))
            angle = normal_angle_deg(station.normal[index], normal)
            max_centroid_offset = max(max_centroid_offset, centroid_offset)
            max_normal_angle = max(max_normal_angle, angle)
            max_surface_rms = max(max_surface_rms, surface_rms)
            member_rows.append({
                "provisional_element_id": element_id,
                "station_id": member["station_id"],
                "surface_id": member["surface_id"],
                "n_atomic_patches": int(station.n_atomic_patches[index]),
                "area_m2": round(area, 4),
                "mean_w": round(float(station.mean_w[index]), 5),
                "member_normal_angle_to_component_deg": round(angle, 4),
                "member_centroid_offset_to_component_m": round(centroid_offset, 5),
                "member_sample_rms_to_component_m": round(surface_rms, 5),
            })

        element_rows.append({
            "provisional_element_id": element_id,
            "n_local_surfaces": len(members),
            "n_stations": len(station_ids),
            "n_atomic_patches": sum(row["n_atomic_patches"] for row in member_rows[-len(members):]),
            "area_sum_m2": round(total_area, 4),
            "area_weighted_mean_w": round(quality_area_sum / max(total_area, 1e-6), 5),
            "n_validation_representatives": len(points),
            "component_plane_normal_x": round(float(normal[0]), 7),
            "component_plane_normal_y": round(float(normal[1]), 7),
            "component_plane_normal_z": round(float(normal[2]), 7),
            "component_plane_d": round(d, 7),
            "component_plane_rms_m": round(rms, 5),
            "component_plane_p95_abs_residual_m": round(float(np.quantile(residual, 0.95)), 5),
            "component_plane_max_abs_residual_m": round(float(residual.max()), 5),
            "max_member_centroid_offset_m": round(max_centroid_offset, 5),
            "max_member_normal_angle_deg": round(max_normal_angle, 4),
            "max_member_sample_rms_m": round(max_surface_rms, 5),
            "geometry_status": "diagnostic_no_threshold",
        })
    return pd.DataFrame(element_rows), pd.DataFrame(member_rows)


def write_edge_audit(candidates: pd.DataFrame, element_map: pd.DataFrame,
                     theta: float, output_dir: str) -> None:
    selected = candidates[candidates.normal_angle_deg.astype(float) <= theta].copy()
    lookup = element_map.rename(columns={"element_id": "provisional_element_id"})
    left = lookup.rename(columns={"station_id": "station_a", "surface_id": "surface_a"})
    right = lookup.rename(columns={"station_id": "station_b", "surface_id": "surface_b"})
    selected = selected.merge(left, on=["station_a", "surface_a"], how="left")
    selected = selected.merge(right, on=["station_b", "surface_b"], how="left", suffixes=("", "_b"))
    if selected.provisional_element_id.isna().any() or selected.provisional_element_id_b.isna().any():
        raise AssertionError("Candidate edge could not be mapped to a provisional element")
    if not (selected.provisional_element_id == selected.provisional_element_id_b).all():
        raise AssertionError("Selected candidate edge crosses provisional element assignments")
    selected = selected.drop(columns=["provisional_element_id_b"])
    selected.to_csv(os.path.join(output_dir, "selected_candidate_edge_audit.csv"), index=False)


def write_gt_audit(association_dir: str, element_map: pd.DataFrame,
                   element_qa: pd.DataFrame, theta: float, output_dir: str) -> None:
    gt_path = os.path.join(association_dir, f"gt_predictions_theta_{theta_tag(theta)}.csv")
    if not os.path.exists(gt_path):  # s05a ran without the historical association GT
        return
    gt = pd.read_csv(gt_path)
    assignment = element_map.rename(columns={"element_id": "provisional_element_id"})
    left = assignment.rename(columns={"station_id": "station_a", "surface_id": "surface_a",
                                      "provisional_element_id": "element_a"})
    right = assignment.rename(columns={"station_id": "station_b", "surface_id": "surface_b",
                                       "provisional_element_id": "element_b"})
    gt = gt.merge(left, on=["station_a", "surface_a"], how="left")
    gt = gt.merge(right, on=["station_b", "surface_b"], how="left")
    compact_qa = element_qa[["provisional_element_id", "n_stations", "component_plane_rms_m",
                             "component_plane_p95_abs_residual_m", "max_member_normal_angle_deg"]]
    gt = gt.merge(compact_qa.rename(columns={
        "provisional_element_id": "element_a",
        "n_stations": "element_a_n_stations",
        "component_plane_rms_m": "element_a_rms_m",
        "component_plane_p95_abs_residual_m": "element_a_p95_m",
        "max_member_normal_angle_deg": "element_a_max_normal_angle_deg",
    }), on="element_a", how="left")
    gt = gt.merge(compact_qa.rename(columns={
        "provisional_element_id": "element_b",
        "n_stations": "element_b_n_stations",
        "component_plane_rms_m": "element_b_rms_m",
        "component_plane_p95_abs_residual_m": "element_b_p95_m",
        "max_member_normal_angle_deg": "element_b_max_normal_angle_deg",
    }), on="element_b", how="left")
    gt.to_csv(os.path.join(output_dir, "gt_pair_component_geometry_audit.csv"), index=False)


def main(argv=None):
    args = parse_args(argv)
    if os.path.exists(args.output_dir):
        raise FileExistsError(f"Refusing to overwrite existing output directory: {args.output_dir}")
    os.makedirs(args.output_dir)

    tag = theta_tag(args.theta_deg)
    map_path = os.path.join(args.association_dir, f"surface_element_map_theta_{tag}.csv")
    candidate_path = os.path.join(args.association_dir, "surface_association_candidates.csv")
    element_map = pd.read_csv(map_path)
    candidates = pd.read_csv(candidate_path)
    stations = load_stations(args.stations, args.association_dir)
    groups = build_member_inventory(stations, element_map)
    log(f"validating {len(groups)} provisional elements from {sum(s.S for s in stations)} local surfaces")

    element_qa, member_qa = validate_components(groups)
    element_qa.to_csv(os.path.join(args.output_dir, "provisional_element_geometry_qa.csv"), index=False)
    member_qa.to_csv(os.path.join(args.output_dir, "local_surface_component_geometry_qa.csv"), index=False)
    write_edge_audit(candidates, element_map, args.theta_deg, args.output_dir)
    write_gt_audit(args.association_dir, element_map, element_qa, args.theta_deg, args.output_dir)

    multi = element_qa[element_qa.n_stations > 1]
    summary = pd.DataFrame([{
        "selected_theta_deg": args.theta_deg,
        "provisional_elements": len(element_qa),
        "multi_station_elements": len(multi),
        "selected_candidate_edges": int((candidates.normal_angle_deg.astype(float) <= args.theta_deg).sum()),
        "multi_station_rms_median_m": round(float(multi.component_plane_rms_m.median()), 5),
        "multi_station_rms_p95_m": round(float(multi.component_plane_rms_m.quantile(0.95)), 5),
        "multi_station_max_rms_m": round(float(multi.component_plane_rms_m.max()), 5),
        "multi_station_max_member_offset_p95_m": round(float(multi.max_member_centroid_offset_m.quantile(0.95)), 5),
    }])
    summary.to_csv(os.path.join(args.output_dir, "component_geometry_summary.csv"), index=False)
    with open(os.path.join(args.output_dir, "README.txt"), "w", encoding="utf-8") as handle:
        handle.write(
            "Step04 v07 measures component-level plane consistency for the selected v06 association only.\n"
            "No RMS, residual, or angular value is used as an automatic rejection threshold in this run.\n"
            "The component plane is fitted with local-surface area weights; P95 and maximum residuals remain visible so small fragments cannot be hidden by that weighting.\n"
            "Room-scope splitting, final class assignment, VLM/material evidence fusion, uncertainty calibration, and IFC export are outside this diagnostic.\n"
        )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
