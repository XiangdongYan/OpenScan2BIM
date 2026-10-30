"""s08 -- Qwen3-VL two-stage scoring of every station-image record (Layer 2.2).

Stage 1 scores the rectilinear image alone and extracts blind observable
evidence; Stage 2 rescores the same image with that evidence and the
metric-geometry card. Pipeline version of
OpenScan2BIM_Step05c_RectilinearMetricJointScoring09.py (2026-10-30): the
legacy reuse of earlier score tables (frozen score/record directories) is
removed, so every record is scored by this code in this run. The scoring path
is the legacy `new_two_stage` path unchanged. An interrupted run resumes from
the per-record checkpoints in the same output directory.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from lib.qwen_scoring import (
    EVIDENCE_FIELDS, SEMANTIC_CLASSES, add_probability_fields, class_order_for, generate_evidence,
    image_only_prompt, joint_prompt, json_default, load_model, metric_geometry_text, parse_evidence,
    resize_image, score_prompt,
)


MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"
CLASSES = SEMANTIC_CLASSES


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Two-stage Qwen3-VL scoring of station-image records")
    parser.add_argument("--projection_dir", required=True)
    parser.add_argument("--element_record_dir", required=True)
    parser.add_argument("--area_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--model_id", default=MODEL_ID)
    parser.add_argument("--max_image_px", type=int, default=1024)
    return parser.parse_args(argv)


def merge_current_rows(projection_dir, element_record_dir, area_dir):
    rows = pd.read_csv(Path(projection_dir) / "rectilinear_metric_evidence.csv")
    elements = pd.read_csv(Path(element_record_dir) / "provisional_geometric_element_records.csv")
    element_columns = [
        "provisional_geometric_element_id", "geometry_role_evidence",
        "n_atomic_patches", "n_member_local_surfaces", "n_stations",
    ]
    rows = rows.merge(elements[element_columns], on="provisional_geometric_element_id",
                      how="left", validate="many_to_one")
    station_area = pd.read_csv(Path(area_dir) / "station_element_area_metrics.csv")
    element_area = pd.read_csv(Path(area_dir) / "element_area_metrics.csv")
    rows = rows.merge(
        station_area[["provisional_geometric_element_id", "station_id", "station_observed_area_m2"]],
        on=["provisional_geometric_element_id", "station_id"], how="left", validate="one_to_one",
    ).merge(
        element_area[["provisional_geometric_element_id", "element_observed_union_area_m2",
                      "element_envelope_area_m2", "element_coverage_ratio"]],
        on="provisional_geometric_element_id", how="left", validate="many_to_one",
    )
    if rows.isna().any()[["station_observed_area_m2", "element_observed_union_area_m2",
                          "element_envelope_area_m2", "element_coverage_ratio"]].any():
        raise AssertionError("Area metrics do not cover every station-element record")
    return rows


def fallback_evidence():
    return {
        "stage1_evidence_parse_status": "unavailable_after_three_attempts",
        **{field: "Stage-1 observable evidence unavailable because the response did not satisfy the JSON schema."
           for field in EVIDENCE_FIELDS},
    }


def save_checkpoint(path, row):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(row, ensure_ascii=False, default=json_default), encoding="utf-8"
    )
    temporary.replace(path)


def main(argv=None):
    args = parse_args(argv)
    output_dir = Path(args.output_dir)
    final_manifest = output_dir / "rectilinear_metric_joint_manifest.json"
    if final_manifest.exists():
        raise FileExistsError(f"Completed output already exists: {output_dir}")
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    rows = merge_current_rows(args.projection_dir, args.element_record_dir, args.area_dir)
    completed = {
        path.stem: json.loads(path.read_text(encoding="utf-8"))
        for path in checkpoint_dir.glob("*.json")
    }
    processor = model = option_token_ids = None
    counts = {"scored": 0, "resumed_from_checkpoint": 0}
    ordered = rows.sort_values(["provisional_geometric_element_id", "station_id"])
    try:
        for ordinal, (_, row) in enumerate(ordered.iterrows(), start=1):
            element_id = int(row["provisional_geometric_element_id"])
            station = str(row["station_id"])
            key = f"geom-{element_id:04d}_{station}"
            if key in completed:
                counts["resumed_from_checkpoint"] += 1
                continue

            result = row.to_dict()
            if processor is None:
                processor, model, option_token_ids = load_model(args.model_id)
            image = resize_image(Path(row["rectilinear_overlay_path"]), args.max_image_px)
            class_order = class_order_for(str(row["station_element_evidence_id"]))
            image_probabilities = score_prompt(
                processor, model, option_token_ids, image,
                image_only_prompt(class_order), class_order,
            )
            evidence_raw = ""
            evidence = None
            for _ in range(3):
                evidence_raw = generate_evidence(processor, model, image)
                evidence = parse_evidence(evidence_raw)
                if evidence["stage1_evidence_parse_status"] == "ok":
                    break
            if evidence["stage1_evidence_parse_status"] != "ok":
                evidence = fallback_evidence()
            result.update({
                "semantic_taxonomy": "|".join(CLASSES),
                "qwen_option_order": "|".join(class_order),
                "stage1_evidence_raw": evidence_raw,
                **evidence,
            })
            add_probability_fields(result, "rectilinear_image", image_probabilities)

            joint = score_prompt(
                processor, model, option_token_ids, image,
                joint_prompt(row, evidence, class_order), class_order,
            )
            add_probability_fields(result, "joint", joint)

            result.update({
                "record_version": "pipeline_s08_qwen_scoring",
                "element_record_id": f"geom-{element_id:04d}",
                "metric_geometry_card": metric_geometry_text(row),
                "provisional_semantic_class": result["joint_top1"],
                "semantic_result_source": "two_stage_scored_in_this_run",
                "external_final_policy_applied": False,
                "experiment_status": "uncalibrated_pending_human_review",
            })
            save_checkpoint(checkpoint_dir / f"{key}.json", result)
            completed[key] = result
            counts["scored"] += 1
            print(f"Semantic {ordinal}/{len(ordered)} {key}: joint={result['joint_top1']}", flush=True)
    finally:
        if model is not None:
            del model, processor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    result = pd.DataFrame(completed.values()).sort_values(
        ["provisional_geometric_element_id", "station_id"]
    ).reset_index(drop=True)
    if len(result) != len(rows):
        raise AssertionError("Semantic output is incomplete")
    for prefix in ("rectilinear_image", "joint"):
        columns = [f"{prefix}_p_{label}" for label in CLASSES]
        if not np.allclose(result[columns].astype(float).sum(axis=1), 1.0, atol=1e-5):
            raise AssertionError(f"{prefix} probabilities do not sum to one")
    result.to_csv(output_dir / "station_image_semantic_scores.csv", index=False, encoding="utf-8")
    manifest = {
        "record_version": "pipeline_s08_qwen_scoring",
        "scope": "every station-image record scored with Stage 1 and Stage 2 in this run; no reuse of earlier runs",
        "n_elements": int(result["provisional_geometric_element_id"].nunique()),
        "n_station_image_records": int(len(result)),
        "counts": counts,
        "model_id": args.model_id,
        "max_image_px": args.max_image_px,
        "semantic_classes": list(CLASSES),
        "external_final_policy_applied": False,
        "interpretation_limit": "Uncalibrated Qwen probabilities; the Final policy (s09) applies geometry constraints.",
    }
    final_manifest.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
