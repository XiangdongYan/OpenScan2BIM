"""s09 -- seven-class element semantic Final with one region reference (Layer 2.3).

Station-image records are constrained by lib/final_policy.py and pooled to each
element with the existing semantic evidence weights. Pipeline version of
OpenScan2BIM_Step05d_ElementSemanticFinal07.py (2026-10-30): input paths are
required arguments; the decision code is unchanged.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import lib.final_policy as policy


CLASSES = policy.CLASSES


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Create seven-class element semantic Final records")
    parser.add_argument("--score_dir", required=True)
    parser.add_argument("--region_refs", required=True)
    parser.add_argument("--config", default=None,
                        help="JSON object overriding entries of lib/final_policy.DEFAULT_CONFIG")
    parser.add_argument("--output_dir", required=True)
    return parser.parse_args(argv)


def load_refs(path):
    refs = json.loads(Path(path).read_text(encoding="utf-8"))
    return {"z_f": refs["floor_reference_z_m"], "z_u": refs["upper_zone_min_z_m"],
            "upper": refs["upper_reference_z_m"], "region": refs["region"]}


def run(station_rows, refs, cfg):
    """Constrain every station-image record and pool to elements."""
    station_final = pd.concat([station_rows.reset_index(drop=True), pd.DataFrame(
        [policy.constrain_row(row, refs, cfg) for _, row in station_rows.iterrows()])], axis=1)
    element_rows = []
    for element_id, group in station_final.groupby("provisional_geometric_element_id", sort=True):
        weights = group["semantic_evidence_weight"].astype(float).to_numpy()
        weights = weights / weights.sum() if weights.sum() > 0 else np.full(len(group), 1 / len(group))
        pooled = (group[[f"final_p_{c}" for c in CLASSES]].to_numpy() * weights[:, None]).sum(axis=0)
        classes = group["final_semantic_class"].astype(str).tolist()
        element_rows.append({
            "provisional_geometric_element_id": int(element_id),
            "station_ids": "|".join(group["station_id"].astype(str)),
            "n_station_images": len(group),
            "element_observed_union_area_m2": float(group["element_observed_union_area_m2"].iloc[0]),
            "station_image_final_classes": "|".join(classes),
            "station_image_final_agreement": len(set(classes)) == 1,
            "station_decision_routes": "|".join(group["decision_route"].astype(str)),
            "final_semantic_class": CLASSES[int(np.argmax(pooled))],
            "pooled_final_top1_probability": float(pooled.max()),
            **{f"pooled_final_p_{c}": float(v) for c, v in zip(CLASSES, pooled)},
        })
    return station_final, pd.DataFrame(element_rows)


def main(argv=None):
    args = parse_args(argv)
    output_dir = Path(args.output_dir)
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing output directory: {output_dir}")
    cfg = {**policy.DEFAULT_CONFIG, **(json.loads(args.config) if args.config else {})}
    unknown = set(cfg) - set(policy.DEFAULT_CONFIG)
    if unknown:
        raise ValueError(f"unknown config keys: {sorted(unknown)}")
    refs = load_refs(args.region_refs)
    station_rows = pd.read_csv(Path(args.score_dir) / "station_image_semantic_scores.csv")
    station_final, element_final = run(station_rows, refs, cfg)

    output_dir.mkdir(parents=True)
    keep = ["provisional_geometric_element_id", "station_id", "semantic_evidence_weight",
            "geometry_role_evidence", "z_world_q025_m", "z_world_q50_m", "z_world_q975_m", "vertical_span_m",
            "horizontal_in_plane_width_m", "element_observed_union_area_m2", "input_top1", "incompatible_classes",
            "decision_route", "geometry_constraint_status", "final_semantic_class", *[f"final_p_{c}" for c in CLASSES]]
    station_final[keep].to_csv(output_dir / "station_image_semantic_final.csv", index=False, encoding="utf-8")
    element_final.to_csv(output_dir / "element_semantic_final.csv", index=False, encoding="utf-8")
    manifest = {
        "record_version": "pipeline_s09_final (= Step05d_ElementSemanticFinal07)",
        "score_dir": args.score_dir, "region_refs": args.region_refs, "region_reference": refs,
        "n_elements": int(len(element_final)), "n_station_image_records": int(len(station_final)),
        "decision_routes": station_final["decision_route"].value_counts().to_dict(),
        "n_station_reassigned": int(station_final["geometry_constraint_status"].eq("reassigned").sum()),
        "element_class_counts": element_final["final_semantic_class"].value_counts().sort_index().to_dict(),
        "policy": policy.policy_manifest(cfg),
        "station_evidence_weight": "semantic_evidence_weight from the score table (unchanged)",
        "interpretation_limit": "Probabilities remain uncalibrated; pooled values are evidence, not confidence.",
    }
    (output_dir / "element_semantic_final_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({k: manifest[k] for k in ("n_elements", "decision_routes", "element_class_counts")}, indent=2))


if __name__ == "__main__":
    main()
