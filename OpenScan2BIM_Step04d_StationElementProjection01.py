"""Build station-level element image evidence from current local-surface members.

Input geometry remains untouched.  For each (provisional element, station),
all member atomic-patch points are reprojected directly into the native
panorama, avoiding the per-local-surface ROI cap used by Step04c.  The result
is one union context image for model evidence and one whole-panorama overlay
for human review.
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from plyfile import PlyData

import OpenScan2BIM_Step04c_PanoramaProjection01 as step04c


RECORD_DIR = "outputs67_step04b_element_records02_station5_6"
ATOMIC_DIR = "outputs49_step01b_atomic04_7station"
E57_DIR = "assets/Level02Lab"
OUT_DIR = "outputs88_step04d_station_element_projection01_smoke"

CROP_MAX_PX = 2048
CROP_MARGIN_PX = 96
FULL_PANO_DISPLAY_WIDTH_PX = 2048
MASK_CLOSE_PX = 21
MASK_DILATE_PX = 7
COMPLETE_POINT_FRACTION = 0.98


def parse_args():
    parser = argparse.ArgumentParser(description="OpenScan2BIM station-element panorama projection")
    parser.add_argument("--record_dir", default=RECORD_DIR)
    parser.add_argument("--atomic_dir", default=ATOMIC_DIR)
    parser.add_argument("--e57_dir", default=E57_DIR)
    parser.add_argument("--output_dir", default=OUT_DIR)
    parser.add_argument("--element_ids", default="", help="Comma-separated element IDs; empty means all.")
    return parser.parse_args()


def parse_ids(value: str) -> set[int]:
    return {int(token.strip()) for token in value.split(",") if token.strip()}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_patch_ids(value: object) -> list[int]:
    return [int(token) for token in str(value).split("|") if token and token != "nan"]


def display_mask(raw_mask: np.ndarray) -> np.ndarray:
    closed = cv2.morphologyEx(
        raw_mask, cv2.MORPH_CLOSE, np.ones((MASK_CLOSE_PX, MASK_CLOSE_PX), dtype=np.uint8)
    )
    return cv2.dilate(closed, np.ones((MASK_DILATE_PX, MASK_DILATE_PX), dtype=np.uint8))


def image_bbox_with_wrap(u: np.ndarray, v: np.ndarray, width: int) -> tuple[int, int, int, int, int]:
    """Return a compact panoramic bbox using the established Step04c seam convention."""
    v0, v1 = int(np.percentile(v, 2)), int(np.percentile(v, 98))
    u0, u1 = int(np.percentile(u, 2)), int(np.percentile(u, 98))
    roll = 0
    if u1 - u0 > width * 0.6:
        rolled_u = (u + width // 2) % width
        rolled_u0, rolled_u1 = int(np.percentile(rolled_u, 2)), int(np.percentile(rolled_u, 98))
        if rolled_u1 - rolled_u0 < u1 - u0:
            u0, u1, roll = rolled_u0, rolled_u1, width // 2
    return u0, u1, v0, v1, roll


def crop_limits(lo: int, hi: int, coordinates: np.ndarray, limit: int, size: int) -> tuple[int, int]:
    if hi - lo <= limit:
        return lo, hi
    center = int(np.median(coordinates))
    out_lo = max(0, center - limit // 2)
    out_hi = min(size, out_lo + limit)
    return max(0, out_hi - limit), out_hi


def raw_mask_from_points(u: np.ndarray, v: np.ndarray, width: int, height: int) -> np.ndarray:
    pixels = np.unique(v.astype(np.int64) * width + u.astype(np.int64))
    mask = np.zeros((height, width), dtype=np.uint8)
    mask[pixels // width, pixels % width] = 255
    return mask


def render_context(native_rgb: np.ndarray, u: np.ndarray, v: np.ndarray) -> tuple[dict, np.ndarray, np.ndarray]:
    height, width = native_rgb.shape[:2]
    u0, u1, v0, v1, roll = image_bbox_with_wrap(u, v, width)
    rolled_u = (u - roll) % width
    u0, u1 = crop_limits(u0, u1, rolled_u, CROP_MAX_PX, width)
    v0, v1 = crop_limits(v0, v1, v, CROP_MAX_PX, height)
    u0 = max(0, u0 - CROP_MARGIN_PX)
    u1 = min(width, u1 + CROP_MARGIN_PX + 1)
    v0 = max(0, v0 - CROP_MARGIN_PX)
    v1 = min(height, v1 + CROP_MARGIN_PX + 1)
    selected = (rolled_u >= u0) & (rolled_u < u1) & (v >= v0) & (v < v1)
    local_u = rolled_u[selected] - u0
    local_v = v[selected] - v0
    crop_width, crop_height = u1 - u0, v1 - v0
    local_mask = raw_mask_from_points(local_u, local_v, crop_width, crop_height)
    crop_source = native_rgb if roll == 0 else np.roll(native_rgb, -roll, axis=1)
    crop = crop_source[v0:v1, u0:u1].copy()
    overlay = crop.copy()
    contours, _ = cv2.findContours(display_mask(local_mask), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (255, 0, 0), 5)
    metadata = {
        "pano_roll_px": roll,
        "crop_u0_rolled_px": u0,
        "crop_u1_rolled_px": u1,
        "crop_v0_px": v0,
        "crop_v1_px": v1,
        "crop_width_px": crop_width,
        "crop_height_px": crop_height,
        "full_member_points": int(len(u)),
        "projected_points_in_context_crop": int(selected.sum()),
        "projected_point_fraction_in_context_crop": round(float(selected.mean()), 8),
        "context_projection_status": (
            "projected_complete" if selected.mean() >= COMPLETE_POINT_FRACTION
            else "projected_principal_context_only"
        ),
        "raw_mask_area_px": int(local_mask.sum() // 255),
    }
    return metadata, local_mask, overlay


def render_full_panorama(native_rgb: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    raw_mask = raw_mask_from_points(u, v, native_rgb.shape[1], native_rgb.shape[0])
    overlay = native_rgb.copy()
    contours, _ = cv2.findContours(display_mask(raw_mask), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (255, 0, 0), 10)
    height, width = overlay.shape[:2]
    display_height = int(round(height * FULL_PANO_DISPLAY_WIDTH_PX / width))
    return cv2.resize(overlay, (FULL_PANO_DISPLAY_WIDTH_PX, display_height), interpolation=cv2.INTER_AREA)


def station_assets(station: str, slots: pd.DataFrame, args):
    first = slots.iloc[0]
    point_ply = Path(first["labeled_point_cloud_path"])
    native_pano = Path(first["native_panorama_path"])
    patch_path = Path(args.atomic_dir) / f"{station}_patch_ids.npy"
    e57_path = Path(args.e57_dir) / f"{station}.e57"
    required = [point_ply, native_pano, patch_path, e57_path]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing station-element projection asset: " + "; ".join(missing))
    native_bgr = cv2.imread(str(native_pano), cv2.IMREAD_COLOR)
    if native_bgr is None:
        raise FileNotFoundError(native_pano)
    native_rgb = cv2.cvtColor(native_bgr, cv2.COLOR_BGR2RGB)
    patch_ids = np.load(patch_path, mmap_mode="r")
    ply = PlyData.read(str(point_ply), mmap=True)
    vertex = ply["vertex"].data
    if len(vertex) != len(patch_ids):
        raise AssertionError(f"Patch IDs do not align with source point cloud for {station}")
    order = np.argsort(patch_ids, kind="stable")
    sorted_patch_ids = patch_ids[order]
    starts = np.searchsorted(sorted_patch_ids, np.arange(int(patch_ids.max()) + 2))
    return native_rgb, step04c.read_sensor(e57_path), vertex, order, starts, sha256_file(native_pano)


def main():
    args = parse_args()
    record_dir = Path(args.record_dir)
    output_dir = Path(args.output_dir)
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing output directory: {output_dir}")
    member_path = record_dir / "local_surface_member_records.csv"
    element_path = record_dir / "provisional_geometric_element_records.csv"
    for path in (member_path, element_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    members = pd.read_csv(member_path)
    elements = pd.read_csv(element_path)
    selected_ids = parse_ids(args.element_ids)
    if selected_ids:
        members = members[members["provisional_geometric_element_id"].isin(selected_ids)].copy()
        elements = elements[elements["provisional_geometric_element_id"].isin(selected_ids)].copy()
    if len(members) == 0:
        raise ValueError("No element members selected")

    output_dir.mkdir(parents=True)
    for directory in ("context_crops", "context_audit_overlays", "context_point_masks", "full_pano_overlays"):
        (output_dir / directory).mkdir()
    rows = []
    for station, station_members in members.groupby("station_id", sort=True):
        native_rgb, sensor, vertex, order, starts, pano_hash = station_assets(station, station_members, args)
        for element_id, group in station_members.groupby("provisional_geometric_element_id", sort=True):
            patch_ids = sorted({patch_id for value in group["member_patch_ids"] for patch_id in parse_patch_ids(value)})
            indices = step04c.point_indices_for_patches(order, starts, patch_ids)
            points = np.column_stack((
                np.asarray(vertex["x"])[indices], np.asarray(vertex["y"])[indices], np.asarray(vertex["z"])[indices],
            )).astype(np.float32, copy=False)
            u, v, _ = step04c.pano_pixels(points, sensor, native_rgb.shape[1], native_rgb.shape[0])
            context, local_mask, context_overlay = render_context(native_rgb, u, v)
            element_id = int(element_id)
            prefix = f"geom-{element_id:04d}_{station}_station-element"
            crop_path = output_dir / "context_crops" / f"{prefix}.jpg"
            overlay_path = output_dir / "context_audit_overlays" / f"{prefix}.jpg"
            mask_path = output_dir / "context_point_masks" / f"{prefix}.png"
            full_pano_path = output_dir / "full_pano_overlays" / f"{prefix}.jpg"
            for path, image in ((crop_path, context_overlay), (overlay_path, context_overlay),
                                (full_pano_path, render_full_panorama(native_rgb, u, v))):
                if not cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR),
                                   [cv2.IMWRITE_JPEG_QUALITY, 92]):
                    raise OSError(f"Could not write {path}")
            if not cv2.imwrite(str(mask_path), local_mask):
                raise OSError(f"Could not write {mask_path}")
            area = group["area_m2"].astype(float).to_numpy()
            quality = group["mean_w"].astype(float).to_numpy()
            weighted_quality = float(np.average(quality, weights=np.maximum(area, 1e-6)))
            rows.append({
                "station_element_evidence_id": f"geom-{element_id:04d}:{station}:station-element-context",
                "provisional_geometric_element_id": element_id,
                "element_record_id": f"geom-{element_id:04d}",
                "station_id": station,
                "n_member_local_surfaces": int(len(group)),
                "member_surface_ids": "|".join(str(int(value)) for value in group.sort_values("surface_id")["surface_id"]),
                "member_record_ids": "|".join(group.sort_values("surface_id")["member_record_id"].astype(str)),
                "member_patch_ids": "||".join(group.sort_values("surface_id")["member_patch_ids"].astype(str)),
                "n_atomic_patches": int(len(patch_ids)),
                "area_m2": round(float(area.sum()), 6),
                "mean_point_quality": round(weighted_quality, 6),
                "native_panorama_path": str(group.iloc[0]["native_panorama_path"]),
                "native_panorama_sha256": pano_hash,
                "native_context_crop_path": crop_path.as_posix(),
                "context_audit_overlay_path": overlay_path.as_posix(),
                "context_point_mask_path": mask_path.as_posix(),
                "full_panorama_overlay_path": full_pano_path.as_posix(),
                "pano_width_px": native_rgb.shape[1],
                "pano_height_px": native_rgb.shape[0],
                **context,
            })
    evidence = pd.DataFrame(rows).sort_values(["provisional_geometric_element_id", "station_id"])
    evidence.to_csv(output_dir / "station_element_evidence_regions.csv", index=False, encoding="utf-8")
    manifest = {
        "record_version": "Step04d_StationElementProjection01",
        "scope": "station-level union evidence for selected current provisional elements",
        "input_sha256": {member_path.name: sha256_file(member_path), element_path.name: sha256_file(element_path)},
        "selected_element_ids": sorted(int(value) for value in elements["provisional_geometric_element_id"].unique()),
        "decision_unit": "one (provisional element, station) union of all member local-surface atomic-patch points",
        "context_crop": "native panorama crop around the union projection, red contour = union support",
        "full_pano_overlay": "human audit only; red contour = same union support in the complete native panorama",
        "not_performed": ["geometry modification", "semantic inference", "material inference", "cross-element region grouping"],
        "n_station_element_evidence_regions": int(len(evidence)),
    }
    (output_dir / "station_element_projection_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
