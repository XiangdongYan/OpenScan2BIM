"""s06b -- target-centred rectilinear views and metric geometry (Layer 2 inputs).

Pipeline version of OpenScan2BIM_Step04d_RectilinearElementProjection05.py
(2026-10-30). Helpers now come from lib/panorama.py, lib/rectilinear.py
(v02 views + v03 metric geometry, verbatim) and s06a; module aliases were
renamed accordingly. Paths and stations are required arguments.

Legacy description: build corrected rectilinear evidence for an arbitrary
station subset.
"""

import argparse
import json
from math import ceil
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

import lib.panorama as panorama
import lib.rectilinear as rect
import s06a_station_projection as station_projection


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Build full rectilinear metric evidence")
    parser.add_argument("--record_dir", required=True, help="s05c output")
    parser.add_argument("--atomic_dir", required=True, help="s02 output")
    parser.add_argument("--reference_dir", required=True, help="s04a output (per-station reference bands)")
    parser.add_argument("--source_projection_dir", required=True, help="s06a output")
    parser.add_argument("--e57_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--stations", nargs="+", required=True)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    output_dir = Path(args.output_dir)
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing output directory: {output_dir}")

    record_dir = Path(args.record_dir)
    members = pd.read_csv(record_dir / "local_surface_member_records.csv")
    members = members[members["station_id"].isin(args.stations)].copy()
    elements = pd.read_csv(record_dir / "provisional_geometric_element_records.csv")
    elements = elements.set_index("provisional_geometric_element_id")
    source_regions = pd.read_csv(
        Path(args.source_projection_dir) / "station_element_evidence_regions.csv"
    ).set_index(["provisional_geometric_element_id", "station_id"])
    expected_pairs = members[["provisional_geometric_element_id", "station_id"]].drop_duplicates()
    expected_element_count = int(members["provisional_geometric_element_id"].nunique())
    expected_pair_count = len(expected_pairs)
    if len(elements) != expected_element_count or expected_pair_count == 0:
        raise AssertionError("Element records and station-element members are inconsistent")

    output_dir.mkdir(parents=True)
    for name in ("rectilinear_clean", "rectilinear_overlays", "rectilinear_ordered_audit"):
        (output_dir / name).mkdir()

    loader_args = SimpleNamespace(atomic_dir=args.atomic_dir, e57_dir=args.e57_dir)
    rows = []
    station_references = {}
    for station in args.stations:
        station_members = members[members["station_id"].eq(station)].copy()
        refs = pd.read_csv(Path(args.reference_dir) / f"{station}_elevation_reference_bands.csv")
        z_floor = float(refs.loc[refs["role"].eq("floor_reference"), "z_ref_m"].min())
        z_top = float(refs.loc[refs["role"].eq("upper_reference"), "z_ref_m"].max())
        if not np.isfinite(z_floor) or not np.isfinite(z_top) or z_top <= z_floor:
            raise AssertionError(f"Invalid vertical reference interval for {station}")
        station_references[station] = {"floor_z_m": z_floor, "top_z_m": z_top}

        pano, sensor, vertex, order, starts, _ = station_projection.station_assets(
            station, station_members, loader_args
        )
        quality_all = np.load(
            Path(args.atomic_dir) / f"{station}_point_quality.npz"
        )["w_uint8"]
        station_groups = list(station_members.groupby("provisional_geometric_element_id", sort=True))
        for station_ordinal, (element_id, group) in enumerate(station_groups, start=1):
            element_id = int(element_id)
            patch_ids = rect.parse_patch_ids(group["member_patch_ids"])
            indices = panorama.point_indices_for_patches(order, starts, patch_ids)
            if len(indices) == 0:
                raise AssertionError(f"No point support for geom-{element_id:04d}/{station}")
            stride = max(1, int(ceil(len(indices) / rect.MAX_RAW_GEOMETRY_SAMPLES)))
            sampled_indices = indices[::stride]
            points = np.column_stack((
                np.asarray(vertex["x"])[sampled_indices],
                np.asarray(vertex["y"])[sampled_indices],
                np.asarray(vertex["z"])[sampled_indices],
            )).astype(np.float64, copy=False)
            quality = quality_all[sampled_indices].astype(np.float64) / 254.0
            finite = np.isfinite(points).all(axis=1)
            points, quality = points[finite], quality[finite]
            voxel_points, voxel_weights = rect.quality_weighted_voxels(points, quality)
            geometry = rect.metric_geometry(
                elements.loc[element_id], voxel_points, voxel_weights, z_floor, z_top
            )

            mask_points = voxel_points
            if len(mask_points) > rect.MAX_MASK_VOXELS:
                selected = np.linspace(
                    0, len(mask_points) - 1, rect.MAX_MASK_VOXELS, dtype=np.int64
                )
                mask_points = mask_points[selected]
            relative = mask_points - sensor[None, :]
            rays = relative / np.maximum(np.linalg.norm(relative, axis=1, keepdims=True), 1e-12)
            centers, cluster_coverage, view_sequence = rect.choose_view_centers(rays)
            if not centers:
                raise AssertionError(f"No rectilinear view center for geom-{element_id:04d}/{station}")

            clean_tiles, overlay_tiles = [], []
            covered = np.zeros(len(rays), dtype=bool)
            for center in centers:
                clean, overlay, visible = rect.render_rectilinear(pano, center, rays)
                clean_tiles.append(clean)
                overlay_tiles.append(overlay)
                covered |= visible
            clean_montage = rect.montage(clean_tiles)
            overlay_montage = rect.montage(overlay_tiles)
            audit_montage = rect.ordered_audit_montage(overlay_tiles, view_sequence)

            prefix = f"geom-{element_id:04d}_{station}_rectilinear"
            clean_path = output_dir / "rectilinear_clean" / f"{prefix}.jpg"
            overlay_path = output_dir / "rectilinear_overlays" / f"{prefix}.jpg"
            audit_path = output_dir / "rectilinear_ordered_audit" / f"{prefix}.jpg"
            rect.write_rgb(clean_path, clean_montage)
            rect.write_rgb(overlay_path, overlay_montage)
            rect.write_rgb(audit_path, audit_montage)

            source = source_regions.loc[(element_id, station)]
            rows.append({
                "record_version": "Step04d_RectilinearElementProjection05",
                "station_element_evidence_id": source["station_element_evidence_id"],
                "provisional_geometric_element_id": element_id,
                "element_record_id": f"geom-{element_id:04d}",
                "station_id": station,
                "rectilinear_clean_path": clean_path.as_posix(),
                "rectilinear_overlay_path": overlay_path.as_posix(),
                "rectilinear_ordered_audit_path": audit_path.as_posix(),
                "rectilinear_view_sequence_json": json.dumps(view_sequence, separators=(",", ":")),
                "rectilinear_view_order": "row_major_greedy_residual_target_ray_coverage",
                "prior_equirectangular_detail_path": source["context_audit_overlay_path"],
                "full_panorama_overlay_path": source["full_panorama_overlay_path"],
                "semantic_evidence_weight": float(
                    np.sqrt(max(float(source["area_m2"]), 0.01))
                    * max(float(source["mean_point_quality"]), 0.0)
                    * max(float(source["projected_point_fraction_in_context_crop"]), 0.0)
                ),
                "n_rectilinear_views": len(centers),
                "rectilinear_fov_deg": rect.FOV_DEG,
                "rectilinear_target_ray_coverage": float(covered.mean()),
                "view_center_selection_coverage": cluster_coverage,
                "raw_support_point_count": int(len(indices)),
                "geometry_raw_sample_stride": stride,
                "geometry_sampled_point_count": int(len(points)),
                "geometry_quality_weighted_voxel_count": int(len(voxel_points)),
                "geometry_voxel_m": rect.VOXEL_M,
                **geometry,
            })
            print(
                f"Rectilinear {station} {station_ordinal}/{len(station_groups)}: "
                f"geom-{element_id:04d}, views={len(centers)}, coverage={covered.mean():.3f}",
                flush=True,
            )
        del pano, vertex, order, starts, quality_all

    result = pd.DataFrame(rows).sort_values(
        ["provisional_geometric_element_id", "station_id"]
    ).reset_index(drop=True)
    if (len(result) != expected_pair_count
            or result["provisional_geometric_element_id"].nunique() != expected_element_count):
        raise AssertionError("Full rectilinear output is incomplete")
    if (result["rectilinear_target_ray_coverage"] <= 0).any():
        raise AssertionError("A rectilinear record has zero target-ray coverage")
    result.to_csv(output_dir / "rectilinear_metric_evidence.csv", index=False, encoding="utf-8")
    rect.write_index(output_dir, result)
    manifest = {
        "record_version": "Step04d_RectilinearElementProjection05",
        "stations": list(args.stations),
        "n_elements": int(result["provisional_geometric_element_id"].nunique()),
        "n_station_element_records": int(len(result)),
        "n_multi_station_elements": int((result.groupby("provisional_geometric_element_id").size() > 1).sum()),
        "tile_px": rect.TILE_PX,
        "fov_deg": rect.FOV_DEG,
        "max_views": rect.MAX_VIEWS,
        "mean_target_ray_coverage": float(result["rectilinear_target_ray_coverage"].mean()),
        "min_target_ray_coverage": float(result["rectilinear_target_ray_coverage"].min()),
        "vertical_reference_source": args.reference_dir,
        "station_references": station_references,
        "geometry_support": "quality-weighted 2 cm station-local atomic-patch voxels",
    }
    (output_dir / "rectilinear_metric_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
