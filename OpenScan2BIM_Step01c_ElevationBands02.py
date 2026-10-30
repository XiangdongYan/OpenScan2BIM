"""Reclassify station elevation bands with a 2.30 m upper-zone clearance."""

import argparse
import json
from pathlib import Path

import pandas as pd

import OpenScan2BIM_Step01c_LocalSurfaces14 as local_v14


SOURCE_DIR = "outputs197_step01c_local_surfaces14_station5_6_7"
OUT_DIR = "outputs201_step01c_elevation_bands02_station5_6_7"
STATIONS = ["2ndLab-5", "2ndLab-6", "2ndLab-7"]
UPPER_MIN_HEIGHT_ABOVE_FLOOR_M = 2.30


def parse_args():
    parser = argparse.ArgumentParser(
        description="Recompute floor/intermediate/upper elevation bands without rerunning geometry"
    )
    parser.add_argument("--source_dir", default=SOURCE_DIR)
    parser.add_argument("--output_dir", default=OUT_DIR)
    parser.add_argument("--stations", nargs="+", default=STATIONS)
    return parser.parse_args()


def main():
    args = parse_args()
    source_dir = Path(args.source_dir)
    output_dir = Path(args.output_dir)
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing output directory: {output_dir}")
    output_dir.mkdir(parents=True)

    local_v14.ELEVATION_MIN_CLEARANCE = UPPER_MIN_HEIGHT_ABOVE_FLOOR_M
    summaries = []
    for station in args.stations:
        surface_path = source_dir / f"{station}_surfaces.csv"
        old_band_path = source_dir / f"{station}_elevation_reference_bands.csv"
        surface_rows = pd.read_csv(surface_path).to_dict(orient="records")
        old_bands = pd.read_csv(old_band_path)
        new_bands = local_v14.attach_station_elevation_bands(surface_rows)
        new_band_table = pd.DataFrame(new_bands)

        pd.DataFrame(surface_rows).to_csv(
            output_dir / f"{station}_surfaces.csv", index=False, encoding="utf-8"
        )
        new_band_table.to_csv(
            output_dir / f"{station}_elevation_reference_bands.csv",
            index=False,
            encoding="utf-8",
        )

        joined = old_bands[["z_ref_m", "role"]].merge(
            new_band_table[["z_ref_m", "role"]], on="z_ref_m", suffixes=("_old", "_new")
        )
        changed = joined.loc[joined["role_old"] != joined["role_new"]]
        summaries.append({
            "station": station,
            "floor_reference_z_m": float(
                new_band_table.loc[new_band_table["role"] == "floor_reference", "z_ref_m"].iloc[0]
            ),
            "n_bands": int(len(new_band_table)),
            "n_intermediate_references": int((new_band_table["role"] == "intermediate_reference").sum()),
            "n_upper_references": int((new_band_table["role"] == "upper_reference").sum()),
            "n_roles_changed": int(len(changed)),
            "changed_z_m": changed["z_ref_m"].astype(float).tolist(),
        })

    manifest = {
        "record_version": "Step01c_ElevationBands02",
        "source_geometry_version": "Step01c_LocalSurfaces14",
        "source_dir": str(source_dir),
        "stations": list(args.stations),
        "horizontal_support_min_area_m2": local_v14.ELEVATION_SUPPORT_MIN_AREA_M2,
        "elevation_cluster_tolerance_m": local_v14.ELEVATION_CLUSTER_TOL,
        "upper_min_height_above_floor_m": UPPER_MIN_HEIGHT_ABOVE_FLOOR_M,
        "elevation_assignment_tolerance_m": local_v14.ELEVATION_ASSIGN_TOL,
        "station_summaries": summaries,
        "interpretation": (
            "The lowest broad horizontal band is floor_reference. Bands below floor+2.30 m "
            "are intermediate_reference; bands at or above floor+2.30 m are upper_reference. "
            "These remain geometry-derived elevation candidates, not final semantic classes."
        ),
    }
    (output_dir / "elevation_band_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
