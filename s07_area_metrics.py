"""s07 -- deduplicated station and fused-element planar area metrics (Layer 2.1).

Pipeline version of OpenScan2BIM_Step04e_ElementAreaMetrics02.py (2026-10-30):
paths and stations are required arguments; code unchanged.

Legacy description: compute deduplicated station and fused-element planar area
metrics.

Each atomic-patch point is projected into the provisional element's fused
plane and quantized on a shared 2 cm grid.  Station area is the occupied-cell
union within one station; element area is the union of those same cell keys
across stations.  The envelope is the convex hull of the occupied cell
squares and is therefore an explicit upper-envelope diagnostic, not an
authoritative BIM surface area.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from plyfile import PlyData
from scipy.spatial import ConvexHull, QhullError

import lib.rectilinear as projection_helpers


AREA_GRID_M = 0.02
POINT_CHUNK = 2_000_000


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Compute deduplicated element areas")
    parser.add_argument("--record_dir", required=True, help="s05c output")
    parser.add_argument("--atomic_dir", required=True, help="s02 output")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--stations", nargs="+", required=True)
    parser.add_argument("--area_grid_m", type=float, default=AREA_GRID_M)
    return parser.parse_args(argv)


def plane_axes(row):
    normal = np.array(
        [
            float(row["component_plane_normal_x"]),
            float(row["component_plane_normal_y"]),
            float(row["component_plane_normal_z"]),
        ],
        dtype=np.float64,
    )
    normal /= max(np.linalg.norm(normal), 1e-12)
    reference = np.array([0.0, 0.0, 1.0]) if abs(normal[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    first = np.cross(normal, reference)
    first /= max(np.linalg.norm(first), 1e-12)
    second = np.cross(normal, first)
    second /= max(np.linalg.norm(second), 1e-12)
    return first, second


def pack_cells(first_index, second_index):
    first = first_index.astype(np.int64, copy=False)
    second = second_index.astype(np.int64, copy=False)
    return (first << 32) | (second & np.int64(0xFFFFFFFF))


def unpack_cells(keys):
    first = (keys >> 32).astype(np.int64)
    second = (keys & np.int64(0xFFFFFFFF)).astype(np.int64)
    second[second >= 2**31] -= 2**32
    return np.column_stack((first, second))


def patch_slices(order, starts, patch_ids):
    for patch_id in patch_ids:
        if patch_id <= 0 or patch_id + 1 >= len(starts):
            raise ValueError(f"Invalid atomic patch id: {patch_id}")
        selected = order[starts[patch_id] : starts[patch_id + 1]]
        for start in range(0, len(selected), POINT_CHUNK):
            yield selected[start : start + POINT_CHUNK]


def occupied_keys(vertex, order, starts, patch_ids, axes, grid_m):
    first_axis, second_axis = axes
    parts = []
    n_points = 0
    for indices in patch_slices(order, starts, patch_ids):
        points = np.column_stack(
            (
                np.asarray(vertex["x"])[indices],
                np.asarray(vertex["y"])[indices],
                np.asarray(vertex["z"])[indices],
            )
        ).astype(np.float64, copy=False)
        points = points[np.isfinite(points).all(axis=1)]
        if len(points) == 0:
            continue
        first = np.floor((points @ first_axis) / grid_m).astype(np.int64)
        second = np.floor((points @ second_axis) / grid_m).astype(np.int64)
        parts.append(np.unique(pack_cells(first, second)))
        n_points += len(points)
    if not parts:
        raise AssertionError("Element has no finite point support")
    return np.unique(np.concatenate(parts)), n_points


def boundary_cells(cells):
    first, second = cells[:, 0], cells[:, 1]
    selected = []
    order_first = np.lexsort((second, first))
    sorted_first = first[order_first]
    _, starts, counts = np.unique(sorted_first, return_index=True, return_counts=True)
    selected.extend(order_first[starts])
    selected.extend(order_first[starts + counts - 1])
    order_second = np.lexsort((first, second))
    sorted_second = second[order_second]
    _, starts, counts = np.unique(sorted_second, return_index=True, return_counts=True)
    selected.extend(order_second[starts])
    selected.extend(order_second[starts + counts - 1])
    return cells[np.unique(np.asarray(selected, dtype=np.int64))]


def envelope_area(keys, grid_m):
    cells = unpack_cells(keys)
    boundary = boundary_cells(cells)
    offsets = np.array(
        [[-0.5, -0.5], [-0.5, 0.5], [0.5, -0.5], [0.5, 0.5]], dtype=np.float64
    )
    corners = (boundary[:, None, :] + offsets[None, :, :]).reshape(-1, 2) * grid_m
    try:
        area = float(ConvexHull(corners).volume)
    except QhullError:
        span = cells.max(axis=0) - cells.min(axis=0) + 1
        area = float(span[0] * span[1]) * grid_m * grid_m
    observed = len(keys) * grid_m * grid_m
    return max(area, observed), len(boundary)


def main(argv=None):
    args = parse_args(argv)
    output_dir = Path(args.output_dir)
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing output directory: {output_dir}")
    if args.area_grid_m <= 0:
        raise ValueError("area_grid_m must be positive")

    record_dir = Path(args.record_dir)
    members = pd.read_csv(record_dir / "local_surface_member_records.csv")
    members = members[members["station_id"].isin(args.stations)].copy()
    elements = pd.read_csv(record_dir / "provisional_geometric_element_records.csv")
    element_lookup = elements.set_index("provisional_geometric_element_id")
    expected_pairs = members[["provisional_geometric_element_id", "station_id"]].drop_duplicates()
    expected_element_count = int(members["provisional_geometric_element_id"].nunique())
    expected_pair_count = len(expected_pairs)
    if len(elements) != expected_element_count or expected_pair_count == 0:
        raise AssertionError("Element records and station-element members are inconsistent")

    station_rows = []
    element_key_parts = {int(element_id): [] for element_id in element_lookup.index}
    for station in args.stations:
        station_members = members[members["station_id"].eq(station)].copy()
        first_row = station_members.iloc[0]
        point_path = Path(first_row["labeled_point_cloud_path"])
        patch_path = Path(args.atomic_dir) / f"{station}_patch_ids.npy"
        if not point_path.is_file() or not patch_path.is_file():
            raise FileNotFoundError(f"Missing point or patch data for {station}")
        patch_ids = np.load(patch_path, mmap_mode="r")
        vertex = PlyData.read(str(point_path), mmap=True)["vertex"].data
        if len(vertex) != len(patch_ids):
            raise AssertionError(f"Point and patch arrays do not align for {station}")
        order = np.argsort(patch_ids, kind="stable")
        sorted_patch_ids = patch_ids[order]
        starts = np.searchsorted(sorted_patch_ids, np.arange(int(patch_ids.max()) + 2))

        groups = list(station_members.groupby("provisional_geometric_element_id", sort=True))
        for ordinal, (element_id, group) in enumerate(groups, start=1):
            element_id = int(element_id)
            patch_members = projection_helpers.parse_patch_ids(group["member_patch_ids"])
            keys, point_count = occupied_keys(
                vertex,
                order,
                starts,
                patch_members,
                plane_axes(element_lookup.loc[element_id]),
                args.area_grid_m,
            )
            element_key_parts[element_id].append(keys)
            station_rows.append(
                {
                    "record_version": "Step04e_ElementAreaMetrics02",
                    "element_record_id": f"geom-{element_id:04d}",
                    "provisional_geometric_element_id": element_id,
                    "station_id": station,
                    "station_observed_area_m2": len(keys) * args.area_grid_m**2,
                    "station_observed_cell_count": int(len(keys)),
                    "station_support_point_count": int(point_count),
                    "area_grid_m": args.area_grid_m,
                }
            )
            print(
                f"Area {station} {ordinal}/{len(groups)}: geom-{element_id:04d}, "
                f"observed={len(keys) * args.area_grid_m**2:.3f} m2",
                flush=True,
            )
        del vertex, patch_ids, order, sorted_patch_ids, starts

    station_metrics = pd.DataFrame(station_rows).sort_values(
        ["provisional_geometric_element_id", "station_id"]
    ).reset_index(drop=True)
    element_rows = []
    for element_id in sorted(element_key_parts):
        parts = element_key_parts[element_id]
        if not parts:
            raise AssertionError(f"No station area support for geom-{element_id:04d}")
        union_keys = np.unique(np.concatenate(parts))
        union_area = len(union_keys) * args.area_grid_m**2
        envelope, boundary_count = envelope_area(union_keys, args.area_grid_m)
        coverage = union_area / max(envelope, args.area_grid_m**2)
        element = element_lookup.loc[element_id]
        station_sum = station_metrics.loc[
            station_metrics["provisional_geometric_element_id"].eq(element_id),
            "station_observed_area_m2",
        ].sum()
        element_rows.append(
            {
                "record_version": "Step04e_ElementAreaMetrics02",
                "element_record_id": f"geom-{element_id:04d}",
                "provisional_geometric_element_id": element_id,
                "n_stations": int(element["n_stations"]),
                "station_observed_area_sum_m2": float(station_sum),
                "element_observed_union_area_m2": float(union_area),
                "element_envelope_area_m2": float(envelope),
                "element_coverage_ratio": float(min(max(coverage, 0.0), 1.0)),
                "cross_station_redundancy_ratio": float(station_sum / max(union_area, 1e-12)),
                "legacy_summed_member_area_m2": float(element["area_sum_m2"]),
                "element_observed_union_cell_count": int(len(union_keys)),
                "element_envelope_boundary_cell_count": int(boundary_count),
                "area_grid_m": args.area_grid_m,
                "envelope_method": "convex_hull_of_observed_grid_cell_squares_on_fused_plane",
            }
        )

    element_metrics = pd.DataFrame(element_rows).sort_values(
        "provisional_geometric_element_id"
    ).reset_index(drop=True)
    if len(station_metrics) != expected_pair_count or len(element_metrics) != expected_element_count:
        raise AssertionError("Area metric output is incomplete")
    station_sum = station_metrics.groupby("provisional_geometric_element_id")[
        "station_observed_area_m2"
    ].sum()
    union = element_metrics.set_index("provisional_geometric_element_id")[
        "element_observed_union_area_m2"
    ]
    if (union - station_sum > 1e-8).any():
        raise AssertionError("Element union area exceeds summed station area")
    if (element_metrics["element_observed_union_area_m2"] - element_metrics["element_envelope_area_m2"] > 1e-8).any():
        raise AssertionError("Observed union area exceeds envelope area")

    output_dir.mkdir(parents=True)
    station_metrics.to_csv(output_dir / "station_element_area_metrics.csv", index=False)
    element_metrics.to_csv(output_dir / "element_area_metrics.csv", index=False)
    manifest = {
        "record_version": "Step04e_ElementAreaMetrics02",
        "record_dir": args.record_dir,
        "atomic_dir": args.atomic_dir,
        "stations": list(args.stations),
        "n_station_element_records": int(len(station_metrics)),
        "n_elements": int(len(element_metrics)),
        "area_grid_m": args.area_grid_m,
        "station_area": "union of occupied planar grid cells for one station-element observation",
        "element_union_area": "union of the same fused-plane grid keys across stations",
        "element_envelope_area": "convex hull of observed grid-cell squares; upper-envelope diagnostic",
        "coverage_ratio": "element_observed_union_area_m2 / element_envelope_area_m2",
        "legacy_area_status": "retained only as legacy_summed_member_area_m2; not a decision metric",
        "union_area_range_m2": [
            float(element_metrics["element_observed_union_area_m2"].min()),
            float(element_metrics["element_observed_union_area_m2"].max()),
        ],
        "coverage_ratio_range": [
            float(element_metrics["element_coverage_ratio"].min()),
            float(element_metrics["element_coverage_ratio"].max()),
        ],
    }
    (output_dir / "element_area_metrics_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
