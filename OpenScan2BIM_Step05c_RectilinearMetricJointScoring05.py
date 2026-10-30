"""Run the frozen two-stage eight-class Qwen pipeline on all Station-5/6 views.

Version 05 keeps the resumable v04 scoring contract but does not abort the full
batch when one Stage-1 response remains non-JSON after three deterministic
attempts.  The affected record is explicitly marked and Stage 2 receives a
neutral unavailable-evidence card rather than inferred or fabricated evidence.
"""

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import OpenScan2BIM_Step05a_ElementSemanticEvidence02 as semantic_source
import OpenScan2BIM_Step05c_RectilinearMetricJointScoring01 as stage_helpers
import OpenScan2BIM_Step05c_RectilinearMetricJointScoring03 as corrected


PROJECTION_DIR = "outputs168_step04d_rectilinear_metric04_station5_6_full"
ELEMENT_RECORD_DIR = "outputs67_step04b_element_records02_station5_6"
PANORAMA_BASELINE_DIR = "outputs95_step05a_element_semantic03_station5_6"
OUT_DIR = "outputs173_step05c_rectilinear_metric_joint06_station5_6_full"
MODEL_ID = semantic_source.MODEL_ID
SEMANTIC_CLASSES = corrected.SEMANTIC_CLASSES


def parse_args():
    parser = argparse.ArgumentParser(description="Run full resumable rectilinear two-stage Qwen scoring")
    parser.add_argument("--projection_dir", default=PROJECTION_DIR)
    parser.add_argument("--element_record_dir", default=ELEMENT_RECORD_DIR)
    parser.add_argument("--panorama_baseline_dir", default=PANORAMA_BASELINE_DIR)
    parser.add_argument("--output_dir", default=OUT_DIR)
    parser.add_argument("--model_id", default=MODEL_ID)
    parser.add_argument("--max_image_px", type=int, default=1024)
    return parser.parse_args()


def log(message):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def class_order_for(record_id):
    order = list(SEMANTIC_CLASSES)
    random.Random(semantic_source.stable_seed(f"{record_id}:eight-class-rectilinear")).shuffle(order)
    return order


def options_text(class_order):
    return "\n".join(
        f"{letter}. {corrected.CLASS_TEXT[label]}"
        for letter, label in zip(semantic_source.OPTION_LETTERS, class_order)
    )


def image_only_prompt(class_order):
    return (
        "Classify the RED-outlined point-cloud surface support in this real indoor TLS image. The image contains "
        "one or more target-centered RECTILINEAR panels reprojected from the same spherical panorama; every panel "
        "shows a different view direction for the same target. Rectilinear projection removes equirectangular "
        "curvature, although perspective and panel seams remain. Everything outside the red outline is context. "
        "Identify the architectural or visible-object role, not material or unseen substrate. Choose exactly one "
        "of the eight classes below.\n\nOptions:\n"
        f"{options_text(class_order)}\n\nAnswer with only the option letter."
    )


def normalized_eight_class_baseline(row):
    values = np.array([float(row[f"vision_raw_p_{label}"]) for label in SEMANTIC_CLASSES])
    total = float(values.sum())
    if total <= 0:
        raise AssertionError("Panorama baseline has no eight-class probability mass")
    return values / total


def add_summary(result, prefix, values):
    corrected.add_probability_fields(result, prefix, values)


def json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    if pd.isna(value):
        return None
    raise TypeError(type(value).__name__)


def save_checkpoint(path, result):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(result, ensure_ascii=False, default=json_default), encoding="utf-8")
    temporary.replace(path)


def main():
    args = parse_args()
    projection = pd.read_csv(Path(args.projection_dir) / "rectilinear_metric_evidence.csv")
    elements = pd.read_csv(
        Path(args.element_record_dir) / "provisional_geometric_element_records.csv"
    )
    baseline = pd.read_csv(
        Path(args.panorama_baseline_dir) / "station_element_semantic_evidence.csv"
    )
    if len(projection) != 245 or projection["provisional_geometric_element_id"].nunique() != 229:
        raise AssertionError("Expected 245 station-image records for 229 elements")

    element_columns = [
        "provisional_geometric_element_id", "geometry_role_evidence", "area_sum_m2",
        "n_atomic_patches", "n_member_local_surfaces", "n_stations",
    ]
    rows = projection.merge(
        elements[element_columns], on="provisional_geometric_element_id", how="left", validate="many_to_one"
    )
    baseline_columns = [
        "station_element_evidence_id", "provisional_geometric_element_id", "station_id",
        "context_audit_overlay_path", "full_panorama_overlay_path", "semantic_evidence_weight",
    ] + [f"vision_raw_p_{label}" for label in SEMANTIC_CLASSES]
    rows = rows.merge(
        baseline[baseline_columns],
        on=["station_element_evidence_id", "provisional_geometric_element_id", "station_id"],
        how="left",
        validate="one_to_one",
        suffixes=("", "_baseline"),
    )
    if rows["semantic_evidence_weight"].isna().any():
        raise AssertionError("A rectilinear record is missing its panorama baseline/weight")

    output_dir = Path(args.output_dir)
    final_manifest = output_dir / "rectilinear_metric_joint_manifest.json"
    if final_manifest.exists():
        raise FileExistsError(f"Completed output already exists: {output_dir}")
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    ordered_rows = rows.sort_values(["provisional_geometric_element_id", "station_id"])
    completed = {
        path.stem: json.loads(path.read_text(encoding="utf-8"))
        for path in checkpoint_dir.glob("*.json")
    }
    log(f"Resuming with {len(completed)}/245 station-image checkpoints")

    processor = model = None
    try:
        if len(completed) < len(ordered_rows):
            processor, model, option_token_ids = stage_helpers.load_model(args.model_id)
        for ordinal, (_, row) in enumerate(ordered_rows.iterrows(), start=1):
            checkpoint_key = f"geom-{int(row['provisional_geometric_element_id']):04d}_{row['station_id']}"
            if checkpoint_key in completed:
                continue
            image_path = Path(row["rectilinear_overlay_path"])
            if not image_path.is_file():
                raise FileNotFoundError(image_path)
            image = stage_helpers.resize_image(image_path, args.max_image_px)
            class_order = class_order_for(str(row["station_element_evidence_id"]))

            image_probabilities = corrected.score_prompt(
                processor, model, option_token_ids, image,
                image_only_prompt(class_order), class_order,
            )
            evidence_raw = ""
            evidence = None
            for _ in range(3):
                evidence_raw = stage_helpers.generate_evidence(processor, model, image)
                evidence = stage_helpers.parse_evidence(evidence_raw)
                if evidence["stage1_evidence_parse_status"] == "ok":
                    break
            if evidence["stage1_evidence_parse_status"] != "ok":
                evidence = {
                    "stage1_evidence_parse_status": "unavailable_after_three_attempts",
                    **{
                        field: "Stage-1 observable evidence unavailable because the response did not satisfy the JSON schema."
                        for field in stage_helpers.EVIDENCE_FIELDS
                    },
                }
            joint_probabilities = corrected.score_prompt(
                processor, model, option_token_ids, image,
                corrected.joint_prompt(row, evidence, class_order), class_order,
            )
            panorama_probabilities = normalized_eight_class_baseline(row)

            result = row.to_dict()
            result.update({
                "semantic_taxonomy": "|".join(SEMANTIC_CLASSES),
                "qwen_option_order": "|".join(class_order),
                "stage1_evidence_raw": evidence_raw,
                **evidence,
                "metric_geometry_card": corrected.metric_geometry_text(row),
                "semantic_result_source": "qwen_rectilinear_joint_evidence_before_element_pooling",
                "external_final_policy_applied": False,
                "experiment_status": "uncalibrated_no_ground_truth",
            })
            add_summary(result, "baseline_image", panorama_probabilities)
            add_summary(result, "rectilinear_image", image_probabilities)
            add_summary(result, "joint", joint_probabilities)
            save_checkpoint(checkpoint_dir / f"{checkpoint_key}.json", result)
            completed[checkpoint_key] = result
            log(
                f"Qwen {ordinal}/245 {checkpoint_key}: pano={result['baseline_image_top1']} "
                f"rect={result['rectilinear_image_top1']} joint={result['joint_top1']}"
            )
    finally:
        if model is not None:
            del model, processor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    result = pd.DataFrame(completed.values()).sort_values(
        ["provisional_geometric_element_id", "station_id"]
    ).reset_index(drop=True)
    if len(result) != 245:
        raise AssertionError("Full two-stage Qwen output is incomplete")
    for prefix in ("baseline_image", "rectilinear_image", "joint"):
        columns = [f"{prefix}_p_{label}" for label in SEMANTIC_CLASSES]
        if not np.allclose(result[columns].astype(float).sum(axis=1), 1.0, atol=1e-5):
            raise AssertionError(f"{prefix} probabilities do not sum to one")
    output_path = output_dir / "station_image_semantic_scores.csv"
    result.to_csv(output_path, index=False, encoding="utf-8")
    manifest = {
        "record_version": "Step05c_RectilinearMetricJointScoring05",
        "projection_dir": args.projection_dir,
        "panorama_baseline_dir": args.panorama_baseline_dir,
        "n_elements": int(result["provisional_geometric_element_id"].nunique()),
        "n_station_image_records": int(len(result)),
        "n_multi_station_elements": int((result.groupby("provisional_geometric_element_id").size() > 1).sum()),
        "model_id": args.model_id,
        "max_image_px": args.max_image_px,
        "semantic_classes": list(SEMANTIC_CLASSES),
        "stage1_evidence_json_valid": int(result["stage1_evidence_parse_status"].eq("ok").sum()),
        "stage1_evidence_schema_fallback": int(
            result["stage1_evidence_parse_status"].ne("ok").sum()
        ),
        "panorama_baseline": "existing full station-element context Qwen probabilities, conditioned to eight classes",
        "rectilinear_stage1": "eight-class image-only logits plus class-neutral observable evidence",
        "rectilinear_stage2": "same image plus Stage-1 evidence plus corrected quantitative geometry",
        "external_final_policy_applied": False,
        "checkpoint_count": int(len(completed)),
        "interpretation_limit": "Qwen probabilities are uncalibrated; Final policy and element pooling occur downstream.",
    }
    final_manifest.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
