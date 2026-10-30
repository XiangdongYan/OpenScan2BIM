"""OpenScan2BIM Step04c v01 -- auditable native-panorama evidence regions.

For each Step04b station-local member surface, project its atomic-patch points
onto the matching native-camera equirectangular panorama.  The output is a
view-evidence record plus an untouched image crop, an exact projected-point
mask, and a contour overlay for review.  This is geometry-to-image provenance
only: it does not run Grounded-SAM, Qwen, CLIP, semantic classification, or
material classification.
"""
import argparse
import csv
import hashlib
import json
import os
from math import pi
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import cv2
import numpy as np
import pandas as pd
from plyfile import PlyData


RECORD_DIR = "./outputs67_step04b_element_records02_station5_6"
ATOMIC_DIR = "./outputs49_step01b_atomic04_7station"
SURFACE_DIR = "./outputs55_step01c_local_surfaces12_7station"
E57_DIR = "./assets/Level02Lab"
OUT_DIR = "./outputs69_step04c_panorama_projection01_station5_6"

CROP_MAX_PX = 2048
CROP_MARGIN_PX = 48
MASK_CLOSE_PX = 21
MASK_DILATE_PX = 7
MIN_PROJECTED_POINTS = 30
COMPLETE_POINT_FRACTION = 0.98


def parse_args():
    parser = argparse.ArgumentParser(description="OpenScan2BIM Step04c panorama projection")
    parser.add_argument("--record_dir", default=RECORD_DIR)
    parser.add_argument("--atomic_dir", default=ATOMIC_DIR)
    parser.add_argument("--surface_dir", default=SURFACE_DIR)
    parser.add_argument("--e57_dir", default=E57_DIR)
    parser.add_argument("--output_dir", default=OUT_DIR)
    parser.add_argument("--limit", type=int, default=0,
                        help="Process the first N ordered view slots for a smoke test; 0 means all.")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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


def bbox_with_wrap(u: np.ndarray, v: np.ndarray, width: int) -> Tuple[int, int, int, int, int]:
    """Robust projected bbox using the established Step01/Step02 seam convention."""
    v0, v1 = int(np.percentile(v, 2)), int(np.percentile(v, 98))
    u0, u1 = int(np.percentile(u, 2)), int(np.percentile(u, 98))
    roll = 0
    if u1 - u0 > width * 0.6:
        rolled_u = (u + width // 2) % width
        rolled_u0, rolled_u1 = int(np.percentile(rolled_u, 2)), int(np.percentile(rolled_u, 98))
        if rolled_u1 - rolled_u0 < u1 - u0:
            u0, u1, roll = rolled_u0, rolled_u1, width // 2
    return u0, u1, v0, v1, roll


def cap_axis(lo: int, hi: int, coordinates: np.ndarray, limit: int, size: int) -> Tuple[int, int]:
    """Take a deterministic principal window when a panoramic support is too broad."""
    if hi - lo <= limit:
        return lo, hi
    center = int(np.median(coordinates))
    out_lo = max(0, center - limit // 2)
    out_hi = min(size, out_lo + limit)
    out_lo = max(0, out_hi - limit)
    return out_lo, out_hi


def stable_name(element_id: int, station: str, surface_id: int) -> str:
    return f"geom-{element_id:04d}_{station}_surface-{surface_id:04d}"


def parse_patch_ids(value: object) -> List[int]:
    return [int(token) for token in str(value).split("|") if token]


def point_indices_for_patches(order: np.ndarray, starts: np.ndarray,
                              patch_ids: Iterable[int]) -> np.ndarray:
    chunks = [order[starts[patch_id]:starts[patch_id + 1]] for patch_id in patch_ids]
    if not chunks or any(len(chunk) == 0 for chunk in chunks):
        raise ValueError("One or more member atomic patches have no aligned source points")
    return np.concatenate(chunks)


def load_patch_surface_lookup(path: Path) -> Dict[int, int]:
    with open(path, newline="", encoding="utf-8") as handle:
        return {int(row["patch_id"]): int(row["surface_id"])
                for row in csv.DictReader(handle)}


def make_evidence_region(pano_rgb: np.ndarray, u: np.ndarray, v: np.ndarray,
                         quality: np.ndarray, ranges: np.ndarray,
                         element_id: int, station: str, surface_id: int,
                         output_dir: Path) -> Dict:
    height, width = pano_rgb.shape[:2]
    raw_pixels_full = np.unique(v.astype(np.int64) * width + u.astype(np.int64))
    u0, u1, v0, v1, roll = bbox_with_wrap(u, v, width)
    rolled_u = (u - roll) % width
    u0, u1 = cap_axis(u0, u1, rolled_u, CROP_MAX_PX, width)
    v0, v1 = cap_axis(v0, v1, v, CROP_MAX_PX, height)
    u0 = max(0, u0 - CROP_MARGIN_PX)
    u1 = min(width, u1 + CROP_MARGIN_PX + 1)
    v0 = max(0, v0 - CROP_MARGIN_PX)
    v1 = min(height, v1 + CROP_MARGIN_PX + 1)
    rolled_pano = pano_rgb if roll == 0 else np.roll(pano_rgb, -roll, axis=1)
    selected = (rolled_u >= u0) & (rolled_u < u1) & (v >= v0) & (v < v1)
    selected_count = int(selected.sum())
    if selected_count < MIN_PROJECTED_POINTS:
        return {
            "projection_status": "insufficient_points_in_principal_region",
            "full_member_points": int(len(u)),
            "projected_points_in_region": selected_count,
            "projected_point_fraction_in_region": round(selected_count / max(len(u), 1), 6),
            "raw_projected_pixels_full": int(len(raw_pixels_full)),
        }

    local_u = rolled_u[selected] - u0
    local_v = v[selected] - v0
    crop_width, crop_height = u1 - u0, v1 - v0
    local_pixels = np.unique(local_v.astype(np.int64) * crop_width + local_u.astype(np.int64))
    raw_mask = np.zeros((crop_height, crop_width), dtype=np.uint8)
    raw_mask[local_pixels // crop_width, local_pixels % crop_width] = 255
    display_mask = cv2.morphologyEx(
        raw_mask, cv2.MORPH_CLOSE, np.ones((MASK_CLOSE_PX, MASK_CLOSE_PX), dtype=np.uint8))
    display_mask = cv2.dilate(display_mask, np.ones((MASK_DILATE_PX, MASK_DILATE_PX), dtype=np.uint8))
    crop = rolled_pano[v0:v1, u0:u1].copy()
    overlay = crop.copy()
    contours, _ = cv2.findContours(display_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (255, 0, 0), 5)

    name = stable_name(element_id, station, surface_id)
    crop_path = output_dir / "native_crops" / f"{name}.jpg"
    mask_path = output_dir / "projected_point_masks" / f"{name}.png"
    overlay_path = output_dir / "audit_overlays" / f"{name}.jpg"
    cv2.imwrite(str(crop_path), cv2.cvtColor(crop, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 92])
    cv2.imwrite(str(mask_path), raw_mask)
    cv2.imwrite(str(overlay_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 92])

    visible_source_rgb = crop[local_v, local_u]
    nonblack_fraction = float((visible_source_rgb.max(axis=1) > 3).mean())
    fraction = selected_count / max(len(u), 1)
    return {
        "projection_status": "projected_complete" if fraction >= COMPLETE_POINT_FRACTION
                             else "projected_principal_region_only",
        "native_crop_path": crop_path.as_posix(),
        "projected_point_mask_path": mask_path.as_posix(),
        "audit_overlay_path": overlay_path.as_posix(),
        "pano_width_px": width,
        "pano_height_px": height,
        "pano_roll_px": roll,
        "crop_u0_rolled_px": u0,
        "crop_u1_rolled_px": u1,
        "crop_v0_px": v0,
        "crop_v1_px": v1,
        "crop_width_px": crop_width,
        "crop_height_px": crop_height,
        "full_member_points": int(len(u)),
        "projected_points_in_region": selected_count,
        "projected_point_fraction_in_region": round(fraction, 6),
        "raw_projected_pixels_full": int(len(raw_pixels_full)),
        "raw_projected_pixels_in_region": int(len(local_pixels)),
        "raw_mask_area_px": int(raw_mask.sum() // 255),
        "display_mask_area_px": int(display_mask.sum() // 255),
        "raw_mask_fraction_of_crop": round(float((raw_mask > 0).mean()), 8),
        "source_nonblack_fraction_at_projected_points": round(nonblack_fraction, 6),
        "mean_point_quality": round(float(quality[selected].mean()), 6),
        "mean_range_m": round(float(ranges[selected].mean()), 5),
        "p02_range_m": round(float(np.quantile(ranges[selected], 0.02)), 5),
        "p98_range_m": round(float(np.quantile(ranges[selected], 0.98)), 5),
    }


def process_station(station: str, slots: pd.DataFrame, args, output_dir: Path) -> List[Dict]:
    root = Path(".")
    point_ply = root / slots.iloc[0].labeled_point_cloud_path
    native_pano = root / slots.iloc[0].native_panorama_path
    patch_path = Path(args.atomic_dir) / f"{station}_patch_ids.npy"
    quality_path = Path(args.atomic_dir) / f"{station}_point_quality.npz"
    patch_surface_path = Path(args.surface_dir) / f"{station}_patch_to_surface.csv"
    e57_path = Path(args.e57_dir) / f"{station}.e57"
    required = [point_ply, native_pano, patch_path, quality_path, patch_surface_path, e57_path]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing Step04c source assets: " + "; ".join(missing))

    pano_bgr = cv2.imread(str(native_pano), cv2.IMREAD_COLOR)
    if pano_bgr is None:
        raise ValueError(f"Could not read native panorama: {native_pano}")
    pano_rgb = cv2.cvtColor(pano_bgr, cv2.COLOR_BGR2RGB)
    sensor = read_sensor(e57_path)
    native_hash = sha256_file(native_pano)
    patch_ids = np.load(patch_path, mmap_mode="r")
    quality_npz = np.load(quality_path)
    quality = quality_npz["w_uint8"].astype(np.float32) / 255.0
    if len(quality) != len(patch_ids):
        raise AssertionError(f"Point quality does not align with patch IDs for {station}")
    patch_surface = load_patch_surface_lookup(patch_surface_path)

    ply = PlyData.read(str(point_ply), mmap=True)
    vertex = ply["vertex"].data
    if len(vertex) != len(patch_ids):
        raise AssertionError(f"Point PLY and patch IDs do not align for {station}")
    order = np.argsort(patch_ids, kind="stable")
    sorted_patch_ids = patch_ids[order]
    patch_max = int(patch_ids.max())
    starts = np.searchsorted(sorted_patch_ids, np.arange(patch_max + 2))

    rows = []
    for slot in slots.itertuples(index=False):
        patch_list = parse_patch_ids(slot.member_patch_ids)
        if any(patch_surface.get(patch_id) != int(slot.surface_id) for patch_id in patch_list):
            raise AssertionError(f"Patch-to-surface provenance mismatch for {slot.member_record_id}")
        indices = point_indices_for_patches(order, starts, patch_list)
        points = np.column_stack((np.asarray(vertex["x"])[indices],
                                  np.asarray(vertex["y"])[indices],
                                  np.asarray(vertex["z"])[indices])).astype(np.float32, copy=False)
        u, v, ranges = pano_pixels(points, sensor, pano_rgb.shape[1], pano_rgb.shape[0])
        projection = make_evidence_region(
            pano_rgb, u, v, quality[indices], ranges,
            int(slot.provisional_geometric_element_id), station, int(slot.surface_id), output_dir)
        rows.append({
            "evidence_region_id": f"{slot.view_slot_id}:region-01",
            "view_slot_id": slot.view_slot_id,
            "member_record_id": slot.member_record_id,
            "provisional_geometric_element_id": int(slot.provisional_geometric_element_id),
            "station_id": station,
            "surface_id": int(slot.surface_id),
            "member_patch_ids": slot.member_patch_ids,
            "native_panorama_path": slot.native_panorama_path,
            "native_panorama_sha256": native_hash,
            "sensor_x_m": round(float(sensor[0]), 6),
            "sensor_y_m": round(float(sensor[1]), 6),
            "sensor_z_m": round(float(sensor[2]), 6),
            **projection,
        })
    del vertex, ply, patch_ids, quality, order, sorted_patch_ids, pano_rgb
    return rows


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing output directory: {output_dir}")
    for folder in (output_dir, output_dir / "native_crops", output_dir / "projected_point_masks",
                   output_dir / "audit_overlays"):
        folder.mkdir(parents=True, exist_ok=False if folder == output_dir else True)

    record_dir = Path(args.record_dir)
    slots = pd.read_csv(record_dir / "element_view_slots.csv")
    members = pd.read_csv(record_dir / "local_surface_member_records.csv")
    required_columns = {
        "view_slot_id", "member_record_id", "provisional_geometric_element_id", "station_id",
        "surface_id", "native_panorama_path", "labeled_point_cloud_path",
    }
    if not required_columns.issubset(slots.columns):
        raise ValueError("Element view slots do not have the required Step04b schema")
    if not {"member_record_id", "member_patch_ids"}.issubset(members.columns):
        raise ValueError("Local-surface member records do not have patch provenance")
    slots = slots.merge(
        members[["member_record_id", "member_patch_ids"]],
        on="member_record_id",
        how="left",
        validate="one_to_one",
    )
    if slots.member_patch_ids.isna().any():
        raise AssertionError("Every Step04c view slot must resolve to one local-surface patch list")
    slots = slots.sort_values(["provisional_geometric_element_id", "station_id", "surface_id"]).reset_index(drop=True)
    if args.limit:
        slots = slots.iloc[:args.limit].copy()
    if slots.empty:
        raise ValueError("No view slots selected for projection")

    all_rows = []
    for station, station_slots in slots.groupby("station_id", sort=True):
        print(f"[{station}] projecting {len(station_slots)} view slots", flush=True)
        all_rows.extend(process_station(station, station_slots.reset_index(drop=True), args, output_dir))
    evidence = pd.DataFrame(all_rows)
    if len(evidence) != len(slots) or evidence.view_slot_id.duplicated().any():
        raise AssertionError("Projection output must contain exactly one evidence-region row per selected view slot")
    evidence.to_csv(output_dir / "projected_view_evidence_regions.csv", index=False)

    updated_slots = slots.merge(
        evidence[["view_slot_id", "evidence_region_id", "projection_status", "native_crop_path",
                  "projected_point_mask_path", "audit_overlay_path"]],
        on="view_slot_id", how="left", validate="one_to_one",
    )
    updated_slots["view_slot_status"] = updated_slots.projection_status
    updated_slots.to_csv(output_dir / "element_view_slots_projected.csv", index=False)
    element_summary = evidence.groupby("provisional_geometric_element_id", as_index=False).agg(
        n_projected_view_slots=("view_slot_id", "count"),
        n_complete_regions=("projection_status", lambda values: int((values == "projected_complete").sum())),
        n_principal_regions=("projection_status", lambda values: int((values == "projected_principal_region_only").sum())),
        n_insufficient_regions=("projection_status", lambda values: int((values == "insufficient_points_in_principal_region").sum())),
        projected_station_ids=("station_id", lambda values: "|".join(sorted(set(values)))),
        projected_raw_pixels=("raw_projected_pixels_in_region", "sum"),
    )
    element_summary.to_csv(output_dir / "element_projection_summary.csv", index=False)
    manifest = {
        "record_version": "Step04c_PanoramaProjection01",
        "input_record_dir": args.record_dir,
        "selected_view_slots": int(len(slots)),
        "processed_stations": sorted(evidence.station_id.unique().tolist()),
        "parameters": {
            "crop_max_px": CROP_MAX_PX,
            "crop_margin_px": CROP_MARGIN_PX,
            "mask_close_px": MASK_CLOSE_PX,
            "mask_dilate_px": MASK_DILATE_PX,
            "minimum_projected_points": MIN_PROJECTED_POINTS,
            "complete_point_fraction": COMPLETE_POINT_FRACTION,
        },
        "status_counts": {str(key): int(value) for key, value in evidence.projection_status.value_counts().items()},
        "not_performed": [
            "semantic VLM inference", "object detection", "finish-region segmentation",
            "material classification", "multi-view posterior fusion", "uncertainty calibration",
        ],
    }
    with open(output_dir / "projection_manifest.json", "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
    with open(output_dir / "README.txt", "w", encoding="utf-8") as handle:
        handle.write(
            "Step04c v01 projects each selected station-local surface to its native equirectangular panorama.\n"
            "native_crops contains unmodified image evidence; projected_point_masks contains exact raw point support; audit_overlays contains only a red contour of a morphology-solidified display mask.\n"
            "A projected_principal_region_only status means the surface spans more than one CROP_MAX window. It is a valid partial observation, not a whole-surface material claim.\n"
            "No VLM inference, class label, opening extraction, finish region, material label, or UQ result is created here.\n"
        )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
