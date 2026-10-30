"""Door-only semantic-boundary preservation for Step01b atomic patches.

The geometry-first atomic patches remain the substrate. Grounded-SAM door
instances are used only to preserve a candidate subregion when planar
geometry alone has joined a door leaf to its host wall. No other semantic
class is modified and no final door label is assigned here.
"""

import argparse
import csv
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from plyfile import PlyData

from OpenScan2BIM_Step01b_Primitives04 import (
    VOXEL,
    classify_patch,
    factorized_planar_attributes,
    in_plane_axes,
)


PANO_DIR = Path("outputs18")
ATOMIC_DIR = Path("outputs49_step01b_atomic04_7station")
OUT_DIR = Path("outputs210_step01b2_door_preservation01_station5")
STATION = "2ndLab-5"
DOOR_SEMANTIC_ID = 3
WHOLE_PATCH_MIN_DOOR_FRACTION = 0.80
SPLIT_MIN_DOOR_FRACTION = 0.05
SPLIT_MIN_DOOR_POINTS = 20_000
SPLIT_MIN_DOOR_AREA_M2 = 0.50


def parse_args():
    parser = argparse.ArgumentParser(description="Door-only preservation of atomic planar patches")
    parser.add_argument("--station", default=STATION)
    parser.add_argument("--pano_dir", default=str(PANO_DIR))
    parser.add_argument("--atomic_dir", default=str(ATOMIC_DIR))
    parser.add_argument("--output_dir", default=str(OUT_DIR))
    return parser.parse_args()


def point_fields(path: Path):
    data = PlyData.read(path, mmap=True)["vertex"].data
    points = np.column_stack((data["x"], data["y"], data["z"])).astype(np.float64)
    semantic = np.asarray(data["semantic_class"], dtype=np.uint8)
    instance = np.asarray(data["instance_id"], dtype=np.uint16)
    return points, semantic, instance


def occupancy_and_hull_area(points, normal, voxel):
    axis_u, axis_v = in_plane_axes(normal)
    u = np.floor(points @ axis_u / voxel).astype(np.int64)
    v = np.floor(points @ axis_v / voxel).astype(np.int64)
    occupancy_area = float(len(np.unique(u * (1 << 32) + v))) * voxel * voxel
    stride = max(1, len(points) // 100_000)
    uv = np.column_stack((points[::stride] @ axis_u, points[::stride] @ axis_v))
    try:
        from scipy.spatial import ConvexHull
        hull_area = float(ConvexHull(uv).volume) if len(uv) >= 8 else 0.0
    except Exception:
        span = uv.max(axis=0) - uv.min(axis=0)
        hull_area = float(span[0] * span[1])
    return occupancy_area, max(hull_area, occupancy_area)


def describe_patch(source_row, point_indices, points, point_weight, ghost, status,
                   instance_ids, door_fraction):
    patch_points = points[point_indices]
    count = len(patch_points)
    centroid = patch_points.mean(axis=0)
    covariance = patch_points.T @ patch_points / count - np.outer(centroid, centroid)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    normal = eigenvectors[:, 0]
    source_normal = np.array([source_row["nx"], source_row["ny"], source_row["nz"]], dtype=float)
    if normal @ source_normal < 0:
        normal *= -1.0
    rms = float(np.sqrt(max(eigenvalues[0], 0.0)))
    area, hull = occupancy_and_hull_area(patch_points, normal, VOXEL)
    z02, z98 = np.percentile(patch_points[:, 2], [2, 98])
    guess, wall_tilt, slab_tilt = classify_patch(normal, patch_points[:, 2], area, 0.0, 0.0)
    row = {
        "patch_id": int(source_row["patch_id"]),
        **factorized_planar_attributes(normal, guess),
        "guess": guess,
        "n_points": count,
        "n_voxels": int(round(area / (VOXEL * VOXEL))),
        "area_m2": round(area, 4),
        "hull_area_m2": round(hull, 4),
        "coverage_ratio": round(min(area / max(hull, 1e-9), 1.0), 4),
        "nx": round(float(normal[0]), 6),
        "ny": round(float(normal[1]), 6),
        "nz": round(float(normal[2]), 6),
        "d": round(-float(normal @ centroid), 6),
        "cx": round(float(centroid[0]), 4),
        "cy": round(float(centroid[1]), 4),
        "cz": round(float(centroid[2]), 4),
        "z_mean": round(float(patch_points[:, 2].mean()), 4),
        "z_p02": round(float(z02), 4),
        "z_p98": round(float(z98), 4),
        "rms_m": round(rms, 5),
        "wall_tilt_deg": round(wall_tilt, 2),
        "slab_tilt_deg": round(slab_tilt, 2),
        "frac_above": 0.0,
        "frac_below": 0.0,
        "merged_fragments": int(source_row["merged_fragments"]),
        "mean_w": round(float(point_weight[point_indices].mean()), 4),
        "ghost_frac": round(float(ghost[point_indices].mean()), 4),
        "door_candidate": int(status in {"door_supported_whole_patch", "door_split_child"}),
        "door_evidence_status": status,
        "door_instance_ids": "|".join(str(value) for value in sorted(instance_ids)),
        "door_point_fraction": round(float(door_fraction), 6),
    }
    return row


def main():
    args = parse_args()
    station = args.station
    pano_dir = Path(args.pano_dir)
    atomic_dir = Path(args.atomic_dir)
    output_dir = Path(args.output_dir)
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing output directory: {output_dir}")
    output_dir.mkdir(parents=True)

    ply_path = pano_dir / station / f"{station}_pano_labeled.ply"
    metadata = pd.read_csv(pano_dir / station / f"{station}_pano_metadata.csv")
    valid_door_instances = set(
        metadata.loc[metadata["Label"].str.lower().eq("door"), "Instance ID"].astype(int)
    )
    points, semantic, instance = point_fields(ply_path)
    original_patch = np.load(atomic_dir / f"{station}_patch_ids.npy", mmap_mode="r")
    revised_patch = np.asarray(original_patch, dtype=np.int32).copy()
    quality = np.load(atomic_dir / f"{station}_point_quality.npz")
    point_weight = quality["w_uint8"].astype(np.float32) / 255.0
    ghost = (quality["flags"] & 4) > 0
    source_rows = pd.read_csv(atomic_dir / f"{station}_patches.csv")

    door_mask = (semantic == DOOR_SEMANTIC_ID) & np.isin(instance, list(valid_door_instances))
    affected = sorted(int(value) for value in np.unique(revised_patch[door_mask]) if value > 0)
    split_specs = []
    whole_candidates = {}
    next_patch_id = int(revised_patch.max())
    audit_rows = []
    for patch_id in affected:
        patch_mask = revised_patch == patch_id
        total = int(patch_mask.sum())
        ids, counts = np.unique(instance[patch_mask & door_mask], return_counts=True)
        for door_instance, door_count in zip(ids.astype(int), counts.astype(int)):
            fraction = door_count / max(total, 1)
            door_indices = np.flatnonzero(patch_mask & (instance == door_instance) & door_mask)
            source_row = source_rows.loc[source_rows.patch_id.eq(patch_id)].iloc[0]
            normal = np.array([source_row.nx, source_row.ny, source_row.nz], dtype=float)
            door_area, _ = occupancy_and_hull_area(points[door_indices], normal, VOXEL)
            action = "insufficient_support"
            child_id = 0
            if fraction >= WHOLE_PATCH_MIN_DOOR_FRACTION:
                whole_candidates.setdefault(patch_id, set()).add(door_instance)
                action = "preserve_whole_patch"
            elif (fraction >= SPLIT_MIN_DOOR_FRACTION
                  and door_count >= SPLIT_MIN_DOOR_POINTS
                  and door_area >= SPLIT_MIN_DOOR_AREA_M2):
                next_patch_id += 1
                child_id = next_patch_id
                revised_patch[door_indices] = child_id
                split_specs.append((patch_id, child_id, door_instance, door_indices, fraction))
                action = "split_door_child"
            audit_rows.append({
                "station_id": station,
                "source_patch_id": patch_id,
                "door_instance_id": door_instance,
                "source_patch_points": total,
                "door_points": door_count,
                "door_point_fraction": round(fraction, 6),
                "door_area_m2": round(door_area, 4),
                "action": action,
                "door_child_patch_id": child_id,
            })

    split_parents = {spec[0] for spec in split_specs}
    output_rows = []
    for source_row in source_rows.to_dict("records"):
        patch_id = int(source_row["patch_id"])
        if patch_id in split_parents:
            indices = np.flatnonzero(revised_patch == patch_id)
            original_door = int(np.count_nonzero(door_mask & (original_patch == patch_id)))
            original_total = int(np.count_nonzero(original_patch == patch_id))
            output_rows.append(describe_patch(
                source_row, indices, points, point_weight, ghost, "host_after_door_split", set(),
                original_door / max(original_total, 1)))
        elif patch_id in whole_candidates:
            row = dict(source_row)
            row.update({
                "door_candidate": 1,
                "door_evidence_status": "door_supported_whole_patch",
                "door_instance_ids": "|".join(str(value) for value in sorted(whole_candidates[patch_id])),
                "door_point_fraction": round(
                    float(np.count_nonzero(door_mask & (revised_patch == patch_id))) /
                    max(int(np.count_nonzero(revised_patch == patch_id)), 1), 6),
            })
            output_rows.append(row)
        else:
            row = dict(source_row)
            row.update({
                "door_candidate": 0,
                "door_evidence_status": "none",
                "door_instance_ids": "",
                "door_point_fraction": 0.0,
            })
            output_rows.append(row)

    source_by_id = source_rows.set_index("patch_id").to_dict("index")
    for parent_id, child_id, door_instance, indices, fraction in split_specs:
        source_row = dict(source_by_id[parent_id])
        source_row["patch_id"] = child_id
        output_rows.append(describe_patch(
            source_row, indices, points, point_weight, ghost, "door_split_child",
            {door_instance}, fraction))

    output_rows.sort(key=lambda row: int(row["patch_id"]))
    if [int(row["patch_id"]) for row in output_rows] != list(range(1, next_patch_id + 1)):
        raise AssertionError("Refined patch ids must remain contiguous")
    np.save(output_dir / f"{station}_patch_ids.npy", revised_patch)
    pd.DataFrame(output_rows).to_csv(
        output_dir / f"{station}_patches.csv", index=False, encoding="utf-8")
    pd.DataFrame(audit_rows).to_csv(
        output_dir / f"{station}_door_refinement.csv", index=False, encoding="utf-8")
    shutil.copy2(
        atomic_dir / f"{station}_point_quality.npz",
        output_dir / f"{station}_point_quality.npz",
    )
    manifest = {
        "record_version": "Step01b2_DoorPreservation01",
        "station": station,
        "scope": "door only; no other semantic class changes atomic geometry",
        "source_atomic_patches": int(len(source_rows)),
        "refined_atomic_patches": int(len(output_rows)),
        "whole_patch_door_candidates": int(len(whole_candidates)),
        "split_door_children": int(len(split_specs)),
        "thresholds": {
            "whole_patch_min_door_fraction": WHOLE_PATCH_MIN_DOOR_FRACTION,
            "split_min_door_fraction": SPLIT_MIN_DOOR_FRACTION,
            "split_min_door_points": SPLIT_MIN_DOOR_POINTS,
            "split_min_door_area_m2": SPLIT_MIN_DOOR_AREA_M2,
        },
        "not_a_final_class": True,
    }
    (output_dir / f"{station}_door_preservation_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    print(pd.DataFrame(audit_rows).to_string(index=False))


if __name__ == "__main__":
    main()
