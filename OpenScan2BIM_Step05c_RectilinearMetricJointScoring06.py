"""Rerun full Station-5/6 Stage 2 with deduplicated element-area metrics."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import OpenScan2BIM_Step05c_RectilinearMetricJointScoring01 as stage_helpers
import OpenScan2BIM_Step05c_RectilinearMetricJointScoring03 as scoring_base


SOURCE_DIR = "outputs176_step05c_rectilinear_metric_joint07_station5_6_full"
AREA_DIR = "outputs179_step04e_element_area_metrics01_station5_6"
OUT_DIR = "outputs180_step05c_rectilinear_metric_joint08_area_station5_6_full"
MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"
SEMANTIC_CLASSES = scoring_base.SEMANTIC_CLASSES


def parse_args():
    parser = argparse.ArgumentParser(description="Rerun Stage 2 with deduplicated area evidence")
    parser.add_argument("--source_dir", default=SOURCE_DIR)
    parser.add_argument("--area_dir", default=AREA_DIR)
    parser.add_argument("--output_dir", default=OUT_DIR)
    parser.add_argument("--model_id", default=MODEL_ID)
    parser.add_argument("--max_image_px", type=int, default=1024)
    return parser.parse_args()


def log(message):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def format_value(value, digits=3):
    value = float(value)
    return "not available" if not np.isfinite(value) else f"{value:.{digits}f}"


def metric_geometry_text(row):
    if str(row["dimension_schema"]) == "horizontal_planar_major_minor":
        dimensions = (
            f"horizontal planar major extent={format_value(row['planar_major_extent_m'])} m; "
            f"minor extent={format_value(row['planar_minor_extent_m'])} m; "
            f"planar elongation major/minor={format_value(row['planar_elongation'], 2)}"
        )
    else:
        dimensions = (
            f"robust world-Z vertical span={format_value(row['vertical_span_m'])} m; "
            f"horizontal in-plane width={format_value(row['horizontal_in_plane_width_m'])} m; "
            f"vertical slenderness height/width={format_value(row['vertical_slenderness'], 2)}; "
            f"story coverage={format_value(row['story_coverage'], 3)}; "
            f"planar elongation major/minor={format_value(row['planar_elongation'], 2)}"
        )
    return (
        "- World frame: registered scan X-Y is horizontal and +Z is upward.\n"
        f"- Measured planar orientation: {row['geometry_role_evidence']}; unit normal="
        f"({format_value(row['plane_normal_x'], 4)}, {format_value(row['plane_normal_y'], 4)}, "
        f"{format_value(row['plane_normal_z'], 4)}); tilt from horizontal="
        f"{format_value(row['plane_tilt_from_horizontal_deg'], 2)} deg; normal azimuth modulo 180="
        f"{format_value(row['plane_normal_azimuth_mod180_deg'], 2)} deg.\n"
        f"- Direction-specific dimensions: {dimensions}.\n"
        "- Observed area and completeness: station observed area="
        f"{format_value(row['station_observed_area_m2'])} m2; cross-station deduplicated element union area="
        f"{format_value(row['element_observed_union_area_m2'])} m2; element envelope area="
        f"{format_value(row['element_envelope_area_m2'])} m2; element coverage ratio="
        f"{format_value(row['element_coverage_ratio'], 3)}.\n"
        f"- Support provenance: {int(row['n_atomic_patches'])} atomic patches and "
        f"{int(row['n_member_local_surfaces'])} local surfaces observed at "
        f"{int(row['n_stations'])} station(s).\n"
        f"- Station references: floor Z={format_value(row['station_floor_reference_z_m'])} m; "
        f"top-reference Z={format_value(row['station_top_reference_z_m'])} m.\n"
        "- Absolute elevation: world-Z q2.5 / q50 / q97.5="
        f"{format_value(row['z_world_q025_m'])} / {format_value(row['z_world_q50_m'])} / "
        f"{format_value(row['z_world_q975_m'])} m.\n"
        "- Weighted support fractions by normalized height (floor=0 and highest upper reference=1): "
        f"below 0={format_value(row['fraction_below_reference'])}; "
        f"0-0.10={format_value(row['fraction_bottom_00_10'])}; "
        f"0.10-0.30={format_value(row['fraction_lower_10_30'])}; "
        f"0.30-0.70={format_value(row['fraction_middle_30_70'])}; "
        f"0.70-0.90={format_value(row['fraction_upper_70_90'])}; "
        f"0.90-1.00={format_value(row['fraction_top_90_100'])}; "
        f"above 1={format_value(row['fraction_above_reference'])}."
    )


def joint_prompt(row, evidence, class_order):
    evidence_text = "\n".join(
        f"- {field.replace('_', ' ')}: {evidence[field]}" for field in stage_helpers.EVIDENCE_FIELDS
    )
    return (
        "Classify the RED-outlined point-cloud surface support. Reinspect the attached target-centered "
        "RECTILINEAR panel or montage, then use the fallible Stage-1 observations and neutral LiDAR measurements "
        "below. All panels refer to the same target. Correct the Stage-1 text when the image disagrees. Geometry "
        "describes measured orientation, extent, elevation, observed area, and completeness but does not name the "
        "object. Identify the surface's architectural or visible-object role, not material or unseen substrate.\n\n"
        "Stage-1 observable image evidence:\n"
        f"{evidence_text}\n\nQuantitative metric geometry:\n{metric_geometry_text(row)}\n\nOptions:\n"
        f"{scoring_base.options_text(class_order)}\n\nAnswer with only the option letter."
    )


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing output directory: {output_dir}")
    rows = pd.read_csv(Path(args.source_dir) / "station_image_semantic_scores.csv")
    station_area = pd.read_csv(Path(args.area_dir) / "station_element_area_metrics.csv")
    element_area = pd.read_csv(Path(args.area_dir) / "element_area_metrics.csv")
    station_columns = [
        "provisional_geometric_element_id", "station_id", "station_observed_area_m2"
    ]
    element_columns = [
        "provisional_geometric_element_id", "element_observed_union_area_m2",
        "element_envelope_area_m2", "element_coverage_ratio",
    ]
    rows = rows.drop(
        columns=[
            "area_sum_m2", "station_observed_area_m2", "element_observed_union_area_m2",
            "element_envelope_area_m2", "element_coverage_ratio",
        ],
        errors="ignore",
    )
    rows = rows.merge(
        station_area[station_columns],
        on=["provisional_geometric_element_id", "station_id"],
        how="left",
        validate="one_to_one",
    ).merge(
        element_area[element_columns],
        on="provisional_geometric_element_id",
        how="left",
        validate="many_to_one",
    )
    if len(rows) != 245 or rows[station_columns[2:] + element_columns[1:]].isna().any().any():
        raise AssertionError("Area metrics do not cover all 245 station-image records")

    stale = [
        column for column in rows.columns
        if column.startswith("joint_") or column in {
            "metric_geometry_card", "provisional_semantic_class", "semantic_result_source",
            "external_final_policy_applied", "experiment_status",
        }
    ]
    rows = rows.drop(columns=stale, errors="ignore")
    processor, model, option_token_ids = stage_helpers.load_model(args.model_id)
    results = []
    try:
        ordered = rows.sort_values(["provisional_geometric_element_id", "station_id"])
        for ordinal, (_, row) in enumerate(ordered.iterrows(), start=1):
            image = stage_helpers.resize_image(Path(row["rectilinear_overlay_path"]), args.max_image_px)
            class_order = [
                label for label in str(row["qwen_option_order"]).split("|") if label in SEMANTIC_CLASSES
            ]
            if set(class_order) != set(SEMANTIC_CLASSES):
                raise AssertionError("Option order does not contain the eight-class taxonomy")
            evidence = {field: str(row[field]) for field in stage_helpers.EVIDENCE_FIELDS}
            joint = scoring_base.score_prompt(
                processor,
                model,
                option_token_ids,
                image,
                joint_prompt(row, evidence, class_order),
                class_order,
            )
            result = row.to_dict()
            scoring_base.add_probability_fields(result, "joint", joint)
            result.update(
                {
                    "metric_geometry_card": metric_geometry_text(row),
                    "provisional_semantic_class": result["joint_top1"],
                    "semantic_result_source": "qwen_joint_with_deduplicated_element_area_metrics",
                    "external_final_policy_applied": False,
                    "experiment_status": "uncalibrated_no_ground_truth",
                }
            )
            results.append(result)
            log(
                f"Stage 2 {ordinal}/245: {row['element_record_id']} / {row['station_id']} "
                f"joint={result['joint_top1']} ({result['joint_top1_probability']:.3f})"
            )
    finally:
        del model, processor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    result = pd.DataFrame(results)
    joint_columns = [f"joint_p_{label}" for label in SEMANTIC_CLASSES]
    if len(result) != 245 or not np.allclose(result[joint_columns].sum(axis=1), 1.0, atol=1e-6):
        raise AssertionError("Full area-aware Joint output is incomplete")
    if "area_sum_m2" in result.columns:
        raise AssertionError("Deprecated area_sum_m2 leaked into the active semantic record")
    output_dir.mkdir(parents=True)
    result.to_csv(output_dir / "station_image_semantic_scores.csv", index=False, encoding="utf-8")
    manifest = {
        "record_version": "Step05c_RectilinearMetricJointScoring06",
        "source_dir": args.source_dir,
        "area_dir": args.area_dir,
        "n_elements": int(result["element_record_id"].nunique()),
        "n_station_image_records": int(len(result)),
        "model_id": args.model_id,
        "semantic_classes": list(SEMANTIC_CLASSES),
        "stage1_image_probabilities_reused_unchanged": True,
        "stage1_observable_evidence_reused_unchanged": True,
        "stage2_rerun": True,
        "area_metrics_in_geometry_card": [
            "station_observed_area_m2", "element_observed_union_area_m2",
            "element_envelope_area_m2", "element_coverage_ratio",
        ],
        "deprecated_area_sum_m2_in_active_record": False,
        "external_final_policy_applied": False,
        "interpretation_limit": "No independent semantic GT; Qwen option probabilities are uncalibrated.",
    }
    (output_dir / "rectilinear_metric_joint_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
