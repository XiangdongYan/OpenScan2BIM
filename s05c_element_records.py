"""s05c -- provisional geometric element records (Framework v0.1, Layer 1.4).

Pipeline version of OpenScan2BIM_Step04b_ElementRecords02.py (2026-10-30) for
any station list; paths and stations are required arguments, code unchanged.

Legacy description: OpenScan2BIM Step04b v02 -- station-5/6 geometry element
records.

This is a provenance and schema layer.  It turns the selected Step04 station-5/6
geometric association into records that later semantic, opening, and material
evidence can attach to.  It intentionally performs no VLM inference and does
not assign a final BIM class, finish region, material, or IFC entity.
"""
import argparse
import json
import os
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


NORMAL_ANGLE_DEG = 7.0
DELTA_PERP_M = 0.15
G_PARALLEL_VERTICAL_M = 0.60
G_PARALLEL_HORIZONTAL_M = 2.00
RMS_QA_BAND_M = 0.20
HORIZONTAL_NORMAL_Z = float(np.cos(np.deg2rad(15.0)))
VERTICAL_NORMAL_Z = float(np.sin(np.deg2rad(15.0)))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="OpenScan2BIM Step04b element records")
    parser.add_argument("--stations", nargs="+", required=True)
    parser.add_argument("--fusion_dir", required=True, help="s05a output")
    parser.add_argument("--qa_dir", required=True, help="s05b output")
    parser.add_argument("--surface_dir", required=True, help="s03 output")
    parser.add_argument("--pano_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    return parser.parse_args(argv)


def relative_path(path: Path) -> str:
    return path.as_posix()


def plane_role(normal_z: float) -> str:
    """Geometry evidence only; this is not a final semantic BIM class."""
    absolute_z = abs(normal_z)
    if absolute_z >= HORIZONTAL_NORMAL_Z:
        return "horizontal_plane"
    if absolute_z <= VERTICAL_NORMAL_Z:
        return "vertical_plane"
    return "oblique_plane"


def join_unique(values: pd.Series, ignore: set = None) -> str:
    ignore = ignore or set()
    ordered = []
    for value in values.astype(str):
        if value not in ignore and value not in ordered:
            ordered.append(value)
    return "|".join(ordered) if ordered else "NA"


def source_paths(station: str, pano_dir: Path) -> Dict[str, str]:
    base = pano_dir / station
    paths = {
        "native_panorama_path": base / f"{station}_pano_NATIVE.jpg",
        "segmented_panorama_path": base / f"{station}_pano_segmented.jpg",
        "pano_mask_path": base / f"{station}_pano_mask.tif",
        "labeled_point_cloud_path": base / f"{station}_pano_labeled.ply",
        "pano_metadata_path": base / f"{station}_pano_metadata.csv",
    }
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing station source evidence: " + "; ".join(missing))
    return {name: relative_path(path) for name, path in paths.items()}


def load_surface_rows(stations: List[str], surface_dir: Path) -> pd.DataFrame:
    frames = []
    for station in stations:
        path = surface_dir / f"{station}_surfaces.csv"
        frame = pd.read_csv(path)
        frame.insert(0, "station_id", station)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def write_schema(path: Path) -> None:
    schema = {
        "record_version": "Step04b_ElementRecords02",
        "record_unit": "provisional_geometric_element",
        "scope": "station 5 and station 6 selected geometry association",
        "meaning": {
            "provisional_geometric_element_id": (
                "Cross-station geometric association identifier. It is not a final BIM/IFC identifier."
            ),
            "geometry_role_evidence": (
                "horizontal_plane, vertical_plane, or oblique_plane inferred only from the component normal."
            ),
            "semantic_class_label": "Reserved for later final semantic inference; unassigned in this run.",
            "visible_finish_label": "Reserved for later finish-region inference; unassigned in this run.",
            "view_slot": (
                "One station-local surface projection slot. It points to source imagery but contains no crop, VLM, or material result yet."
            ),
        },
        "fixed_association_gate": {
            "oriented_normal_dot": "> 0",
            "normal_angle_deg_max": NORMAL_ANGLE_DEG,
            "delta_perp_m_max": DELTA_PERP_M,
            "g_parallel_vertical_m_max": G_PARALLEL_VERTICAL_M,
            "g_parallel_horizontal_m_max": G_PARALLEL_HORIZONTAL_M,
        },
        "current_qa": {
            "component_rms_band_m": RMS_QA_BAND_M,
            "automatic_rejection": False,
        },
        "not_performed": [
            "semantic class inference",
            "opening/object instance extraction",
            "finish-region construction",
            "visible-finish material inference",
            "multi-view visual posterior fusion",
            "uncertainty calibration",
            "IFC authoring",
        ],
    }
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(schema, handle, ensure_ascii=False, indent=2)


def main(argv=None):
    args = parse_args(argv)
    output_dir = Path(args.output_dir)
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing output directory: {output_dir}")
    output_dir.mkdir(parents=True)

    fusion_dir = Path(args.fusion_dir)
    qa_dir = Path(args.qa_dir)
    surface_dir = Path(args.surface_dir)
    pano_dir = Path(args.pano_dir)

    element_map = pd.read_csv(fusion_dir / "surface_element_map_theta_07deg.csv")
    element_qa = pd.read_csv(qa_dir / "provisional_element_geometry_qa.csv")
    member_qa = pd.read_csv(qa_dir / "local_surface_component_geometry_qa.csv")
    surfaces = load_surface_rows(args.stations, surface_dir)

    expected_members = len(surfaces)
    if len(element_map) != expected_members or len(member_qa) != expected_members:
        raise AssertionError("Association and QA rows must cover every selected local surface")
    if element_map.duplicated(["station_id", "surface_id"]).any():
        raise AssertionError("A local surface may not map to more than one provisional element")
    if surfaces.duplicated(["station_id", "surface_id"]).any():
        raise AssertionError("Surface source inventory contains duplicate station/surface keys")

    member = element_map.rename(columns={"element_id": "provisional_geometric_element_id"}).copy()
    qa_source = member_qa.rename(columns={"provisional_element_id": "provisional_geometric_element_id"})
    key_columns = ["provisional_geometric_element_id", "station_id", "surface_id"]
    qa_lookup = {
        (int(row.provisional_geometric_element_id), str(row.station_id), int(row.surface_id)): row._asdict()
        for row in qa_source.itertuples(index=False)
    }
    source_lookup = {
        (str(row.station_id), int(row.surface_id)): row._asdict()
        for row in surfaces.itertuples(index=False)
    }
    member_keys = [
        (int(row.provisional_geometric_element_id), str(row.station_id), int(row.surface_id))
        for row in member.itertuples(index=False)
    ]
    if len(qa_lookup) != len(qa_source) or any(key not in qa_lookup for key in member_keys):
        raise AssertionError("Member QA provenance keys do not cover the association map")
    source_keys = [(station, surface) for _, station, surface in member_keys]
    if len(source_lookup) != len(surfaces) or any(key not in source_lookup for key in source_keys):
        raise AssertionError("Local-surface source provenance keys do not cover the association map")
    for column in qa_source.columns:
        if column not in key_columns:
            member[column] = [qa_lookup[key][column] for key in member_keys]
    for column in surfaces.columns:
        if column not in {"station_id", "surface_id"}:
            target = column if column not in member.columns else f"{column}_surface"
            member[target] = [source_lookup[key][column] for key in source_keys]
    required_member_columns = [
        "provisional_geometric_element_id", "station_id", "surface_id",
        "n_atomic_patches", "member_normal_angle_to_component_deg",
        "member_centroid_offset_to_component_m", "member_sample_rms_to_component_m",
        "member_patch_ids", "geom_state", "planar_orientation",
    ]
    if len(member) != expected_members or member[required_member_columns].isna().any().any():
        raise AssertionError("Member record join lost required provenance or geometry evidence")
    if not (member.n_atomic_patches == member.n_patches_merged).all():
        raise AssertionError("QA atomic-patch count differs from local-surface provenance")

    source_by_station = {station: source_paths(station, pano_dir) for station in args.stations}
    for key in next(iter(source_by_station.values())):
        member[key] = member.station_id.map(lambda station: source_by_station[station][key])

    station_counts = member.groupby("provisional_geometric_element_id")["station_id"].nunique()
    member["n_stations_in_element"] = member.provisional_geometric_element_id.map(station_counts)
    member["is_cross_station_geometric_element"] = (member.n_stations_in_element > 1).astype(np.uint8)
    member["record_version"] = "Step04b_ElementRecords02"
    member["member_record_id"] = member.apply(
        lambda row: f"geom-{int(row.provisional_geometric_element_id):04d}:{row.station_id}:surface-{int(row.surface_id):04d}",
        axis=1,
    )
    member["view_slot_id"] = member.member_record_id + ":native_pano_projection"
    member["view_slot_status"] = "pending_surface_projection"
    member["object_evidence_status"] = "not_attached"
    member["finish_evidence_status"] = "not_attached"

    qa = element_qa.rename(columns={"provisional_element_id": "provisional_geometric_element_id"})
    if qa.provisional_geometric_element_id.duplicated().any():
        raise AssertionError("Element QA must have exactly one row per provisional element")
    grouped = member.groupby("provisional_geometric_element_id", sort=True)
    summary = grouped.agg(
        n_member_local_surfaces=("surface_id", "count"),
        member_station_ids=("station_id", lambda values: "|".join(sorted(set(values)))),
        member_surface_ids=("surface_id", lambda values: "|".join(str(int(value)) for value in values)),
        member_patch_ids=("member_patch_ids", lambda values: "||".join(str(value) for value in values)),
        member_area_sum_m2=("area_m2", "sum"),
        member_geometry_states=("geom_state", lambda values: join_unique(values)),
        member_planar_orientations=("planar_orientation", lambda values: join_unique(values)),
        member_elevation_bands=("elevation_band", lambda values: join_unique(values, {"NA"})),
        member_local_occupancy_contexts=("local_occupancy_context", lambda values: join_unique(values, {"NA"})),
    ).reset_index()
    records = qa.copy()
    summary_lookup = {
        int(row.provisional_geometric_element_id): row._asdict()
        for row in summary.itertuples(index=False)
    }
    qa_ids = [int(value) for value in records.provisional_geometric_element_id]
    if len(summary_lookup) != len(summary) or any(value not in summary_lookup for value in qa_ids):
        raise AssertionError("Element summary does not cover every geometry-QA record")
    for column in summary.columns:
        if column != "provisional_geometric_element_id":
            records[column] = [summary_lookup[value][column] for value in qa_ids]
    if len(records) != len(qa):
        raise AssertionError("Element summary did not cover every geometry-QA record")
    if not (records.n_local_surfaces == records.n_member_local_surfaces).all():
        raise AssertionError("Element QA member count differs from source provenance")
    if not np.allclose(records.area_sum_m2, records.member_area_sum_m2, atol=1e-4):
        raise AssertionError("Element QA area sum differs from source provenance")

    records["record_version"] = "Step04b_ElementRecords02"
    records["element_record_id"] = records.provisional_geometric_element_id.map(
        lambda value: f"geom-{int(value):04d}")
    records["geometry_role_evidence"] = records.component_plane_normal_z.map(plane_role)
    records["is_cross_station_geometric_element"] = (records.n_stations > 1).astype(np.uint8)
    records["within_current_rms_20cm_band"] = (records.component_plane_rms_m <= RMS_QA_BAND_M).astype(np.uint8)
    records["association_normal_angle_deg_max"] = NORMAL_ANGLE_DEG
    records["association_delta_perp_m_max"] = DELTA_PERP_M
    records["association_g_parallel_vertical_m_max"] = G_PARALLEL_VERTICAL_M
    records["association_g_parallel_horizontal_m_max"] = G_PARALLEL_HORIZONTAL_M
    records["semantic_class_label"] = "unassigned"
    records["semantic_evidence_status"] = "not_attached"
    records["opening_object_status"] = "not_extracted"
    records["visible_finish_region_status"] = "not_created"
    records["visible_finish_label"] = "unassigned"
    records["material_evidence_status"] = "not_attached"
    records["semantic_uq_status"] = "not_computed"
    records["material_uq_status"] = "not_computed"
    records["n_view_slots"] = records.n_local_surfaces

    element_columns = [
        "record_version", "element_record_id", "provisional_geometric_element_id",
        "is_cross_station_geometric_element", "n_stations", "member_station_ids",
        "n_member_local_surfaces", "member_surface_ids", "n_atomic_patches",
        "member_patch_ids", "geometry_role_evidence", "member_geometry_states",
        "member_planar_orientations", "member_elevation_bands", "member_local_occupancy_contexts",
        "area_sum_m2", "member_area_sum_m2", "area_weighted_mean_w",
        "component_plane_normal_x", "component_plane_normal_y", "component_plane_normal_z",
        "component_plane_d", "component_plane_rms_m", "component_plane_p95_abs_residual_m",
        "component_plane_max_abs_residual_m", "max_member_centroid_offset_m",
        "max_member_normal_angle_deg", "within_current_rms_20cm_band", "geometry_status",
        "association_normal_angle_deg_max", "association_delta_perp_m_max",
        "association_g_parallel_vertical_m_max", "association_g_parallel_horizontal_m_max",
        "n_view_slots", "semantic_class_label", "semantic_evidence_status",
        "opening_object_status", "visible_finish_region_status", "visible_finish_label",
        "material_evidence_status", "semantic_uq_status", "material_uq_status",
    ]
    member_columns = [
        "record_version", "member_record_id", "view_slot_id", "provisional_geometric_element_id",
        "station_id", "surface_id", "n_stations_in_element", "is_cross_station_geometric_element",
        "n_atomic_patches", "member_patch_ids", "area_m2", "mean_w", "geom_state",
        "planar_orientation", "local_occupancy_context", "elevation_band", "z_ref_m", "zspan_m",
        "resid_rms_m", "member_normal_angle_to_component_deg",
        "member_centroid_offset_to_component_m", "member_sample_rms_to_component_m",
        "native_panorama_path", "segmented_panorama_path", "pano_mask_path",
        "labeled_point_cloud_path", "pano_metadata_path", "view_slot_status",
        "object_evidence_status", "finish_evidence_status",
    ]
    view_columns = [
        "view_slot_id", "member_record_id", "provisional_geometric_element_id", "station_id",
        "surface_id", "native_panorama_path", "segmented_panorama_path", "pano_mask_path",
        "labeled_point_cloud_path", "pano_metadata_path", "view_slot_status",
        "object_evidence_status", "finish_evidence_status",
    ]
    records[element_columns].sort_values("provisional_geometric_element_id").to_csv(
        output_dir / "provisional_geometric_element_records.csv", index=False)
    member[member_columns].sort_values(["provisional_geometric_element_id", "station_id", "surface_id"]).to_csv(
        output_dir / "local_surface_member_records.csv", index=False)
    member[view_columns].sort_values(["provisional_geometric_element_id", "station_id", "surface_id"]).to_csv(
        output_dir / "element_view_slots.csv", index=False)
    write_schema(output_dir / "element_record_schema.json")

    validation = {
        "record_version": "Step04b_ElementRecords02",
        "stations": args.stations,
        "provisional_geometric_elements": int(len(records)),
        "cross_station_geometric_elements": int(records.is_cross_station_geometric_element.sum()),
        "local_surface_member_records": int(len(member)),
        "view_slots": int(len(member)),
        "all_members_have_source_paths": True,
        "all_members_have_unassigned_semantic_and_finish_slots": True,
        "all_elements_within_current_rms_20cm_band": bool(records.within_current_rms_20cm_band.all()),
    }
    with open(output_dir / "validation_summary.json", "w", encoding="utf-8") as handle:
        json.dump(validation, handle, ensure_ascii=False, indent=2)
    with open(output_dir / "README.txt", "w", encoding="utf-8") as handle:
        handle.write(
            "Step04b v02 creates a stable provenance layer for the station-5/6 selected geometric association.\n"
            "provisional_geometric_element_records.csv is one row per geometry element.\n"
            "local_surface_member_records.csv preserves every member (station_id, surface_id) and its atomic-patch, geometry-QA, and source-file provenance.\n"
            "element_view_slots.csv lists pending native-panorama surface-projection slots; it contains no generated crop or VLM result.\n"
            "Semantic labels, opening instances, finish regions, material labels, UQ, and IFC output are intentionally unassigned.\n"
        )
    print(json.dumps(validation, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
