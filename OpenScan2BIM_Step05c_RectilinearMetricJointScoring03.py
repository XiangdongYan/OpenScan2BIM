"""Rerun eight-class Stage 2 with corrected metric geometry fields."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import OpenScan2BIM_Step05a_ElementSemanticEvidence02 as semantic_source
import OpenScan2BIM_Step05c_RectilinearMetricJointScoring01 as v01


SOURCE_DIR = "outputs155_step05c_rectilinear_metric_joint03_smoke_station5_000_016"
GEOMETRY_DIR = "outputs162_step04d_rectilinear_metric03_smoke_station5_000_016"
OUT_DIR = "outputs163_step05c_rectilinear_metric_joint04_smoke_station5_000_016"
MODEL_ID = semantic_source.MODEL_ID
SEMANTIC_CLASSES = (
    "wall", "floor", "ceiling", "column", "door", "window", "furniture", "other"
)
CLASS_TEXT = {
    "wall": "a surface belonging to a vertical wall or architectural partition",
    "floor": "a floor or accessible lower walking surface",
    "ceiling": "an overhead ceiling/service-zone surface, including overhead MEP",
    "column": "a surface belonging to a finite-section structural column or support",
    "door": "a door leaf or door assembly at a wall opening",
    "window": "a window, glazed opening, or inclined glazed facade panel",
    "furniture": "furniture, casework, workbench, cabinet, or another bounded interior object",
    "other": "another positively identifiable surface outside the classes above",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Rerun corrected eight-class metric Joint scoring")
    parser.add_argument("--source_dir", default=SOURCE_DIR)
    parser.add_argument("--geometry_dir", default=GEOMETRY_DIR)
    parser.add_argument("--output_dir", default=OUT_DIR)
    parser.add_argument("--model_id", default=MODEL_ID)
    parser.add_argument("--max_image_px", type=int, default=1024)
    return parser.parse_args()


def log(message):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def format_value(value, digits=3):
    value = float(value)
    return "not available" if not np.isfinite(value) else f"{value:.{digits}f}"


def options_text(class_order):
    return "\n".join(
        f"{letter}. {CLASS_TEXT[label]}"
        for letter, label in zip(semantic_source.OPTION_LETTERS, class_order)
    )


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
        f"- Observed support area: {format_value(row['area_sum_m2'])} m2; support is assembled from "
        f"{int(row['n_atomic_patches'])} atomic patches and {int(row['n_member_local_surfaces'])} local surfaces "
        f"observed at {int(row['n_stations'])} station(s).\n"
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
        f"- {field.replace('_', ' ')}: {evidence[field]}" for field in v01.EVIDENCE_FIELDS
    )
    return (
        "Classify the RED-outlined point-cloud surface support. Reinspect the attached target-centered "
        "RECTILINEAR panel or montage, then use the fallible Stage-1 observations and neutral LiDAR measurements "
        "below. All panels refer to the same target. Correct the Stage-1 text when the image disagrees. Geometry "
        "describes measured orientation, extent, and elevation but does not name the object. Identify the surface's "
        "architectural or visible-object role, not material or unseen substrate.\n\nStage-1 observable image evidence:\n"
        f"{evidence_text}\n\nQuantitative metric geometry:\n{metric_geometry_text(row)}\n\nOptions:\n"
        f"{options_text(class_order)}\n\nAnswer with only the option letter."
    )


def score_prompt(processor, model, option_token_ids, image, prompt, class_order):
    message = [{"role": "user", "content": [
        {"type": "image", "image": image},
        {"type": "text", "text": prompt},
    ]}]
    chat = processor.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[chat], images=[image], return_tensors="pt").to(model.device)
    with torch.no_grad():
        logits = model(**inputs).logits[0, -1, option_token_ids[:len(class_order)]].float().cpu().numpy()
    option_probabilities = np.exp(logits - logits.max())
    option_probabilities /= option_probabilities.sum()
    probabilities = np.zeros(len(SEMANTIC_CLASSES), dtype=np.float64)
    for probability, label in zip(option_probabilities, class_order):
        probabilities[SEMANTIC_CLASSES.index(label)] = probability
    return probabilities


def condition_source(row, prefix):
    values = np.array([float(row[f"{prefix}_{label}"]) for label in SEMANTIC_CLASSES])
    return values / values.sum()


def entropy(values):
    positive = values[values > 0]
    return float(-(positive * np.log(positive)).sum() / np.log(len(values)))


def add_probability_fields(result, prefix, values):
    order = np.argsort(values)[::-1]
    result[f"{prefix}_top1"] = SEMANTIC_CLASSES[int(order[0])]
    result[f"{prefix}_top1_probability"] = float(values[order[0]])
    result[f"{prefix}_top2"] = SEMANTIC_CLASSES[int(order[1])]
    result[f"{prefix}_top1_margin"] = float(values[order[0]] - values[order[1]])
    result[f"{prefix}_entropy_normalized"] = entropy(values)
    for label, value in zip(SEMANTIC_CLASSES, values):
        result[f"{prefix}_p_{label}"] = float(value)


def merge_geometry(source, geometry):
    key = ["provisional_geometric_element_id", "station_id"]
    geometry = geometry.copy()
    replacement = [column for column in geometry.columns if column not in key]
    source = source.drop(columns=[column for column in replacement if column in source], errors="ignore")
    stale_dimensions = [
        "observed_length_m", "observed_width_m", "observed_height_m",
        "observed_major_m", "observed_minor_m", "observed_aspect_ratio",
    ]
    source = source.drop(columns=stale_dimensions, errors="ignore")
    return source.merge(geometry, on=key, how="inner", validate="one_to_one")


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing output directory: {output_dir}")
    source = pd.read_csv(Path(args.source_dir) / "rectilinear_metric_joint_scores.csv")
    geometry = pd.read_csv(Path(args.geometry_dir) / "rectilinear_metric_evidence.csv")
    rows = merge_geometry(source, geometry)
    if len(rows) != 16 or not rows["stage1_evidence_parse_status"].eq("ok").all():
        raise AssertionError("Expected 16 complete corrected-geometry smoke records")

    stale = [
        column for column in rows.columns
        if column.startswith(("joint_", "final_")) or column in {
            "metric_geometry_card", "provisional_semantic_class", "semantic_result_source",
            "external_final_policy_applied", "experiment_status", "current_final_provisional",
            "previous_smoke_provisional_semantic_class",
        }
    ]
    rows = rows.drop(columns=stale, errors="ignore")
    processor, model, option_token_ids = v01.load_model(args.model_id)
    results = []
    try:
        for ordinal, (_, row) in enumerate(rows.sort_values("provisional_geometric_element_id").iterrows(), 1):
            image = v01.resize_image(Path(row["rectilinear_overlay_path"]), args.max_image_px)
            class_order = [
                label for label in str(row["qwen_option_order"]).split("|") if label in SEMANTIC_CLASSES
            ]
            if set(class_order) != set(SEMANTIC_CLASSES):
                raise AssertionError("Option order does not contain the eight-class taxonomy")
            evidence = {field: str(row[field]) for field in v01.EVIDENCE_FIELDS}
            joint = score_prompt(
                processor, model, option_token_ids, image,
                joint_prompt(row, evidence, class_order), class_order,
            )
            result = row.to_dict()
            result["semantic_taxonomy"] = "|".join(SEMANTIC_CLASSES)
            result["qwen_option_order"] = "|".join(class_order)
            result["conditioned_option_order"] = "|".join(
                label for label in str(row["conditioned_option_order"]).split("|") if label in SEMANTIC_CLASSES
            )
            baseline = condition_source(row, "baseline_p")
            rectilinear = condition_source(row, "rectilinear_image_p")
            add_probability_fields(result, "baseline_image_only", baseline)
            add_probability_fields(result, "rectilinear_image", rectilinear)
            add_probability_fields(result, "joint", joint)
            for prefix in ("baseline_p", "rectilinear_image_p"):
                for label in SEMANTIC_CLASSES:
                    result[f"{prefix}_{label}"] = float(
                        baseline[SEMANTIC_CLASSES.index(label)] if prefix == "baseline_p"
                        else rectilinear[SEMANTIC_CLASSES.index(label)]
                    )
            result.update({
                "metric_geometry_card": metric_geometry_text(row),
                "provisional_semantic_class": result["joint_top1"],
                "semantic_result_source": "qwen_corrected_metric_joint_top1",
                "external_final_policy_applied": False,
                "experiment_status": "uncalibrated_no_ground_truth",
            })
            results.append(result)
            log(
                f"Stage 2 {ordinal}/16: {row['element_record_id']} "
                f"rect={result['rectilinear_image_top1']} joint={result['joint_top1']} "
                f"({result['joint_top1_probability']:.3f})"
            )
    finally:
        del model, processor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    result = pd.DataFrame(results)
    removable = [
        column for column in result.columns
        if column.startswith(("baseline_p_", "rectilinear_image_p_", "joint_p_"))
        and column.rsplit("_", 1)[-1] not in SEMANTIC_CLASSES
    ]
    result = result.drop(columns=removable)
    joint_columns = [f"joint_p_{label}" for label in SEMANTIC_CLASSES]
    if not np.allclose(result[joint_columns].sum(axis=1), 1.0, atol=1e-6):
        raise AssertionError("Eight-class Joint probabilities do not sum to one")
    output_dir.mkdir(parents=True)
    result.to_csv(output_dir / "rectilinear_metric_joint_scores.csv", index=False, encoding="utf-8")
    manifest = {
        "record_version": "Step05c_RectilinearMetricJointScoring03",
        "source_dir": args.source_dir,
        "geometry_dir": args.geometry_dir,
        "n_records": int(len(result)),
        "model_id": args.model_id,
        "semantic_classes": list(SEMANTIC_CLASSES),
        "stage1_evidence_reused_unchanged": True,
        "stage2_rerun": True,
        "geometry_card": (
            "corrected vertical_span, horizontal_in_plane_width, height/width vertical_slenderness, "
            "story_coverage, and separate major/minor planar_elongation"
        ),
        "external_final_policy_applied": False,
        "interpretation_limit": "No independent semantic GT; Qwen option probabilities are uncalibrated.",
    }
    (output_dir / "rectilinear_metric_joint_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
