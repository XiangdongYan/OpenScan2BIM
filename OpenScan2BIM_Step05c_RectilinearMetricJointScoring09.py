"""Rerun changed Station-5 door views without reusing stale visual evidence.

An unchanged patch ID is not enough for reuse after a door's points have been
split out. Reuse Stage 1 only for byte-identical target images, and reuse
Stage 2 only when the quantitative geometry card is also identical.
"""

import argparse
import hashlib
import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import OpenScan2BIM_Step05c_RectilinearMetricJointScoring01 as helpers
import OpenScan2BIM_Step05c_RectilinearMetricJointScoring03 as scorer
import OpenScan2BIM_Step05c_RectilinearMetricJointScoring05 as stage1_base
import OpenScan2BIM_Step05c_RectilinearMetricJointScoring06 as stage2_base


PROJECTION_DIR = "outputs222_step04d_rectilinear_door_station5_6_7"
ELEMENT_RECORD_DIR = "outputs219_step04b_door_station5_6_7"
AREA_DIR = "outputs223_step04e_area_door_station5_6_7"
FROZEN_SCORE_DIR = "outputs205_step05c_rectilinear_metric_joint10_station5_6_7"
FROZEN_RECORD_DIR = "outputs200_step04b_element_records_station5_6_7"
OUT_DIR = "outputs224_step05c_door_station5_6_7"
MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"
CLASSES = scorer.SEMANTIC_CLASSES


def parse_args():
    parser = argparse.ArgumentParser(description="Extend frozen semantic scoring to Station 5/6/7")
    parser.add_argument("--projection_dir", default=PROJECTION_DIR)
    parser.add_argument("--element_record_dir", default=ELEMENT_RECORD_DIR)
    parser.add_argument("--area_dir", default=AREA_DIR)
    parser.add_argument("--frozen_score_dir", default=FROZEN_SCORE_DIR)
    parser.add_argument("--frozen_record_dir", default=FROZEN_RECORD_DIR)
    parser.add_argument("--output_dir", default=OUT_DIR)
    parser.add_argument("--model_id", default=MODEL_ID)
    parser.add_argument("--max_image_px", type=int, default=1024)
    return parser.parse_args()


def member_signatures(record_dir):
    members = pd.read_csv(Path(record_dir) / "local_surface_member_records.csv")
    atomic = []
    for row in members.itertuples():
        for patch_id in str(row.member_patch_ids).split("|"):
            atomic.append({
                "provisional_geometric_element_id": int(row.provisional_geometric_element_id),
                "station_id": str(row.station_id),
                "patch_id": int(patch_id),
            })
    atomic = pd.DataFrame(atomic).drop_duplicates()
    view = (
        atomic.groupby(["provisional_geometric_element_id", "station_id"])["patch_id"]
        .apply(lambda values: tuple(sorted(map(int, values))))
        .reset_index(name="view_signature")
    )
    element = {
        int(element_id): tuple(sorted(
            (str(row.station_id), int(row.patch_id)) for row in group.itertuples()
        ))
        for element_id, group in atomic.groupby("provisional_geometric_element_id")
    }
    return view, element


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


def stage1_fields(old_row):
    names = [
        "semantic_taxonomy", "qwen_option_order", "stage1_evidence_raw",
        "stage1_evidence_parse_status", *helpers.EVIDENCE_FIELDS,
        "rectilinear_image_top1", "rectilinear_image_top1_probability",
        "rectilinear_image_top2", "rectilinear_image_top1_margin",
        "rectilinear_image_entropy_normalized",
        *[f"rectilinear_image_p_{label}" for label in CLASSES],
    ]
    return {name: old_row[name] for name in names}


def full_score_fields(old_row):
    result = stage1_fields(old_row)
    names = [
        "joint_top1", "joint_top1_probability", "joint_top2", "joint_top1_margin",
        "joint_entropy_normalized", *[f"joint_p_{label}" for label in CLASSES],
    ]
    result.update({name: old_row[name] for name in names})
    return result


def fallback_evidence():
    return {
        "stage1_evidence_parse_status": "unavailable_after_three_attempts",
        **{field: "Stage-1 observable evidence unavailable because the response did not satisfy the JSON schema."
           for field in helpers.EVIDENCE_FIELDS},
    }


def save_checkpoint(path, row):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(row, ensure_ascii=False, default=stage1_base.json_default), encoding="utf-8"
    )
    temporary.replace(path)


@lru_cache(maxsize=None)
def image_digest(path):
    return hashlib.sha256(Path(path).read_bytes()).digest()


def same_image(row, old_row):
    return image_digest(str(row["rectilinear_overlay_path"])) == image_digest(
        str(old_row["rectilinear_overlay_path"])
    )


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    final_manifest = output_dir / "rectilinear_metric_joint_manifest.json"
    if final_manifest.exists():
        raise FileExistsError(f"Completed output already exists: {output_dir}")
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    rows = merge_current_rows(args.projection_dir, args.element_record_dir, args.area_dir)
    frozen = pd.read_csv(Path(args.frozen_score_dir) / "station_image_semantic_scores.csv")
    old_views, old_elements = member_signatures(args.frozen_record_dir)
    new_views, new_elements = member_signatures(args.element_record_dir)
    old_element_by_signature = {signature: element_id for element_id, signature in old_elements.items()}
    old_view_by_signature = {
        (str(row.station_id), row.view_signature): int(row.provisional_geometric_element_id)
        for row in old_views.itertuples()
    }
    new_view_signature = {
        (int(row.provisional_geometric_element_id), str(row.station_id)): row.view_signature
        for row in new_views.itertuples()
    }
    frozen_lookup = frozen.set_index(["provisional_geometric_element_id", "station_id"])

    completed = {
        path.stem: json.loads(path.read_text(encoding="utf-8"))
        for path in checkpoint_dir.glob("*.json")
    }
    processor = model = option_token_ids = None
    counts = {"full_score_reuse": 0, "stage1_reuse_stage2_rerun": 0, "new_two_stage": 0}
    ordered = rows.sort_values(["provisional_geometric_element_id", "station_id"])
    try:
        for ordinal, (_, row) in enumerate(ordered.iterrows(), start=1):
            element_id = int(row["provisional_geometric_element_id"])
            station = str(row["station_id"])
            key = f"geom-{element_id:04d}_{station}"
            if key in completed:
                counts[str(completed[key]["extension_inference_status"])] += 1
                continue

            result = row.to_dict()
            element_signature = new_elements[element_id]
            old_element_id = old_element_by_signature.get(element_signature)
            view_signature = new_view_signature[(element_id, station)]
            old_view_element_id = old_view_by_signature.get((station, view_signature))

            old_element_row = (
                frozen_lookup.loc[(old_element_id, station)] if old_element_id is not None else None
            )
            old_view_row = (
                frozen_lookup.loc[(old_view_element_id, station)]
                if old_view_element_id is not None else None
            )
            current_card = stage2_base.metric_geometry_text(row)
            if (old_element_row is not None and same_image(row, old_element_row)
                    and current_card == str(old_element_row["metric_geometry_card"])):
                old_row = old_element_row
                result.update(full_score_fields(old_row))
                status = "full_score_reuse"
            else:
                if processor is None:
                    processor, model, option_token_ids = helpers.load_model(args.model_id)
                image = helpers.resize_image(Path(row["rectilinear_overlay_path"]), args.max_image_px)
                if old_view_row is not None and same_image(row, old_view_row):
                    old_row = old_view_row
                    result.update(stage1_fields(old_row))
                    class_order = str(old_row["qwen_option_order"]).split("|")
                    evidence = {field: str(old_row[field]) for field in helpers.EVIDENCE_FIELDS}
                    status = "stage1_reuse_stage2_rerun"
                else:
                    class_order = stage1_base.class_order_for(str(row["station_element_evidence_id"]))
                    image_probabilities = scorer.score_prompt(
                        processor, model, option_token_ids, image,
                        stage1_base.image_only_prompt(class_order), class_order,
                    )
                    evidence_raw = ""
                    evidence = None
                    for _ in range(3):
                        evidence_raw = helpers.generate_evidence(processor, model, image)
                        evidence = helpers.parse_evidence(evidence_raw)
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
                    scorer.add_probability_fields(result, "rectilinear_image", image_probabilities)
                    status = "new_two_stage"

                joint = scorer.score_prompt(
                    processor, model, option_token_ids, image,
                    stage2_base.joint_prompt(row, evidence, class_order), class_order,
                )
                scorer.add_probability_fields(result, "joint", joint)

            result.update({
                "record_version": "Step05c_RectilinearMetricJointScoring09",
                "element_record_id": f"geom-{element_id:04d}",
                "metric_geometry_card": current_card,
                "provisional_semantic_class": result["joint_top1"],
                "semantic_result_source": "station567_door_rerun_with_image_and_geometry_reuse_checks",
                "external_final_policy_applied": False,
                "experiment_status": "uncalibrated_extension_pending_human_review",
                "extension_inference_status": status,
            })
            save_checkpoint(checkpoint_dir / f"{key}.json", result)
            completed[key] = result
            counts[status] += 1
            print(f"Semantic {ordinal}/{len(ordered)} {key}: {status}, joint={result['joint_top1']}", flush=True)
    finally:
        if model is not None:
            del model, processor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    result = pd.DataFrame(completed.values()).sort_values(
        ["provisional_geometric_element_id", "station_id"]
    ).reset_index(drop=True)
    if len(result) != len(rows):
        raise AssertionError("Extension semantic output is incomplete")
    for prefix in ("rectilinear_image", "joint"):
        columns = [f"{prefix}_p_{label}" for label in CLASSES]
        if not np.allclose(result[columns].astype(float).sum(axis=1), 1.0, atol=1e-5):
            raise AssertionError(f"{prefix} probabilities do not sum to one")
    result.to_csv(output_dir / "station_image_semantic_scores.csv", index=False, encoding="utf-8")
    manifest = {
        "record_version": "Step05c_RectilinearMetricJointScoring09",
        "scope": "Station 5/6/7 door rerun; reuse requires exact target image and geometry card",
        "n_elements": int(result["provisional_geometric_element_id"].nunique()),
        "n_station_image_records": int(len(result)),
        "reuse_counts": counts,
        "frozen_score_dir": args.frozen_score_dir,
        "frozen_record_dir": args.frozen_record_dir,
        "model_id": args.model_id,
        "semantic_classes": list(CLASSES),
        "external_final_policy_applied": False,
        "interpretation_limit": "Station-7 and changed cross-station elements remain pending human semantic review.",
    }
    final_manifest.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
