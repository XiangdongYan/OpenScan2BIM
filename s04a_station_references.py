"""s04a -- per-station elevation reference bands with the region clearance c.

Pipeline version of OpenScan2BIM_Step01c_ElevationBands02.py (2026-10-30). The
station references feed the s06b geometry card (station floor/top references)
that Qwen Stage 2 reads; the Final policy and point labels use the region
references of s04b instead. The clearance (legacy constant 2.30 m) is now the
required --clearance_m and is passed to s03.attach_station_elevation_bands
instead of patching a module global.
"""

import argparse
import json
from pathlib import Path

import pandas as pd

import s03_local_surfaces as local_v14


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Recompute floor/intermediate/upper elevation bands without rerunning geometry"
    )
    parser.add_argument("--source_dir", required=True, help="s03 output")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--stations", nargs="+", required=True)
    parser.add_argument("--clearance_m", type=float, required=True,
                        help="upper references lie at least this far above the floor band (legacy 2.30)")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    source_dir = Path(args.source_dir)
    output_dir = Path(args.output_dir)
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing output directory: {output_dir}")
    output_dir.mkdir(parents=True)

    summaries = []
    for station in args.stations:
        surface_path = source_dir / f"{station}_surfaces.csv"
        old_band_path = source_dir / f"{station}_elevation_reference_bands.csv"
        surface_rows = pd.read_csv(surface_path).to_dict(orient="records")
        old_bands = pd.read_csv(old_band_path)
        new_bands = local_v14.attach_station_elevation_bands(surface_rows, min_clearance=args.clearance_m)
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
        "record_version": "pipeline_s04a_station_references (= Step01c_ElevationBands02)",
        "source_geometry_version": "pipeline_s03_local_surfaces (= Step01c_LocalSurfaces16)",
        "source_dir": str(source_dir),
        "stations": list(args.stations),
        "horizontal_support_min_area_m2": local_v14.ELEVATION_SUPPORT_MIN_AREA_M2,
        "elevation_cluster_tolerance_m": local_v14.ELEVATION_CLUSTER_TOL,
        "upper_min_height_above_floor_m": args.clearance_m,
        "elevation_assignment_tolerance_m": local_v14.ELEVATION_ASSIGN_TOL,
        "station_summaries": summaries,
        "interpretation": (
            f"The lowest broad horizontal band is floor_reference. Bands below floor+{args.clearance_m:.2f} m "
            f"are intermediate_reference; bands at or above floor+{args.clearance_m:.2f} m are upper_reference. "
            "These remain geometry-derived elevation candidates, not final semantic classes."
        ),
    }
    (output_dir / "elevation_band_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
