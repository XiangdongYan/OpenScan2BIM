"""Region-level elevation references (Framework v0.1, Layer 0).

All stations of one region are pooled before the existing band logic of
Step01c v14 runs, so every station of the region shares one floor reference
z_f, one set of upper references, and one upper-zone boundary z_u = z_f + c.
The clearance c is a per-region parameter: it must lie above door heads and
below the region's lowest broad upper reference, which this script checks.
"""

import argparse
import json
from pathlib import Path

import pandas as pd

import OpenScan2BIM_Step01c_LocalSurfaces14 as local_v14


SOURCE_DIR = "outputs216_door_integrated_inputs/surfaces"
REGION = "Level02Lab"
STATIONS = ["2ndLab-5", "2ndLab-6", "2ndLab-7"]
UPPER_ZONE_CLEARANCE_M = 2.30


def parse_args():
    parser = argparse.ArgumentParser(description="Pool station surfaces into one region elevation reference")
    parser.add_argument("--source_dir", default=SOURCE_DIR)
    parser.add_argument("--region", default=REGION)
    parser.add_argument("--stations", nargs="+", default=STATIONS)
    parser.add_argument("--clearance_m", type=float, default=UPPER_ZONE_CLEARANCE_M)
    parser.add_argument("--output_dir", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    source_dir = Path(args.source_dir)
    output_dir = Path(args.output_dir)
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing output directory: {output_dir}")

    rows, station_floor = [], {}
    for station in args.stations:
        table = pd.read_csv(source_dir / f"{station}_surfaces.csv")
        station_floor[station] = float(table["station_floor_reference_z_m"].dropna().iloc[0])
        table.insert(0, "station_id", station)
        table["surface_id"] = station + ":" + table["surface_id"].astype(str)
        rows += table.to_dict(orient="records")

    local_v14.ELEVATION_MIN_CLEARANCE = args.clearance_m
    bands = pd.DataFrame(local_v14.attach_station_elevation_bands(rows))
    z_f = float(bands.loc[bands["role"] == "floor_reference", "z_ref_m"].iloc[0])
    upper = bands.loc[bands["role"] == "upper_reference", "z_ref_m"].astype(float).tolist()
    z_u = round(z_f + args.clearance_m, 3)
    if not upper or z_u >= min(upper):
        raise ValueError(f"z_u = {z_u} is not below the lowest upper reference {upper}; choose another clearance")

    references = {
        "record_version": "Step01c_ElevationBands03",
        "region": args.region,
        "stations": list(args.stations),
        "source_dir": str(source_dir),
        "floor_reference_z_m": z_f,
        "upper_zone_clearance_m": args.clearance_m,
        "upper_zone_min_z_m": z_u,
        "upper_reference_z_m": upper,
        "lowest_upper_reference_above_floor_m": round(min(upper) - z_f, 3),
        "station_floor_reference_z_m": station_floor,
        "horizontal_support_min_area_m2": local_v14.ELEVATION_SUPPORT_MIN_AREA_M2,
        "elevation_cluster_tolerance_m": local_v14.ELEVATION_CLUSTER_TOL,
        "elevation_assignment_tolerance_m": local_v14.ELEVATION_ASSIGN_TOL,
        "interpretation": (
            "Pooled over the region's stations. The lowest broad horizontal band is the floor "
            "reference z_f; bands at or above z_u = z_f + clearance are upper references. "
            "z_u bounds the upper (ceiling/service) zone used by the Final policy and the point labels."
        ),
    }
    output_dir.mkdir(parents=True)
    bands.to_csv(output_dir / "region_elevation_reference_bands.csv", index=False, encoding="utf-8")
    pd.DataFrame(rows).to_csv(output_dir / "region_surfaces.csv", index=False, encoding="utf-8")
    (output_dir / "region_references.json").write_text(json.dumps(references, indent=2), encoding="utf-8")
    print(json.dumps(references, indent=2))
    print(bands.to_string(index=False))


if __name__ == "__main__":
    main()
