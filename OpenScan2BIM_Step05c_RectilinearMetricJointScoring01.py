"""Two-stage rectilinear image + metric-geometry semantic smoke test.

Stage 1 independently scores the rectilinear image and extracts observable
evidence. Stage 2 sees the same image again, the Stage-1 evidence, and neutral
metric LiDAR measurements. The previous semantic class, route-compatible class
list, and review outcome are withheld. A bounded policy emits one concrete
provisional class while retaining every upstream probability for audit.
"""

import argparse
import json
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

import OpenScan2BIM_Step05a_ElementSemanticEvidence02 as semantic_v02


PROJECTION_DIR = "outputs150_step04d_rectilinear_metric01_smoke_station5_000_016"
PANORAMA_BASELINE_DIR = "outputs148_step05c_neutral_geometry_joint01_smoke_station5_000_016"
OUT_DIR = "outputs151_step05c_rectilinear_metric_joint01_smoke_station5_000_016"
MODEL_ID = semantic_v02.MODEL_ID
VISUAL_ANCHOR_THRESHOLD = 0.95
JOINT_WEIGHT = 0.50
GEOMETRY_SOFT_PENALTY = 0.50
EPSILON = 1e-8

EVIDENCE_FIELDS = (
    "target_shape_evidence",
    "target_boundary_attachment_evidence",
    "target_context_evidence",
    "occlusion_mixing_evidence",
    "evidence_limitations",
)


def parse_args():
    parser = argparse.ArgumentParser(description="Run rectilinear metric-geometry joint Qwen scoring")
    parser.add_argument("--projection_dir", default=PROJECTION_DIR)
    parser.add_argument("--panorama_baseline_dir", default=PANORAMA_BASELINE_DIR)
    parser.add_argument("--output_dir", default=OUT_DIR)
    parser.add_argument("--model_id", default=MODEL_ID)
    parser.add_argument("--max_image_px", type=int, default=1024)
    parser.add_argument("--visual_anchor_threshold", type=float, default=VISUAL_ANCHOR_THRESHOLD)
    parser.add_argument("--joint_weight", type=float, default=JOINT_WEIGHT)
    parser.add_argument("--geometry_soft_penalty", type=float, default=GEOMETRY_SOFT_PENALTY)
    return parser.parse_args()


def log(message):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def load_model(model_id):
    from transformers import AutoProcessor
    try:
        from transformers import AutoModelForImageTextToText as ModelClass
    except ImportError:
        from transformers import Qwen3VLForConditionalGeneration as ModelClass
    log(f"Loading {model_id}")
    processor = AutoProcessor.from_pretrained(model_id, local_files_only=True)
    model = ModelClass.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="cuda" if torch.cuda.is_available() else "cpu",
        local_files_only=True,
    ).eval()
    option_token_ids = []
    for letter in semantic_v02.OPTION_LETTERS:
        ids = processor.tokenizer(letter, add_special_tokens=False).input_ids
        if len(ids) != 1:
            raise RuntimeError(f"Option {letter!r} is not one tokenizer token: {ids}")
        option_token_ids.append(ids[0])
    return processor, model, option_token_ids


def resize_image(path, max_image_px):
    image = Image.open(path).convert("RGB")
    if max(image.size) > max_image_px:
        scale = max_image_px / max(image.size)
        image = image.resize(
            (max(1, round(image.width * scale)), max(1, round(image.height * scale))),
            Image.LANCZOS,
        )
    return image


def options_text(class_order):
    return "\n".join(
        f"{letter}. {semantic_v02.CLASS_TEXT[label]}"
        for letter, label in zip(semantic_v02.OPTION_LETTERS, class_order)
    )


def rectilinear_image_prompt(class_order):
    return (
        "Classify the RED-outlined point-cloud surface support in this real indoor TLS image. The image contains "
        "one or more target-centered RECTILINEAR panels reprojected from the same spherical panorama; every panel "
        "shows a different view direction for the same target. Rectilinear projection removes equirectangular "
        "curvature, although perspective and panel seams remain. Everything outside the red outline is context. "
        "Identify the architectural or visible-object role, not its material or unseen substrate. Choose unknown "
        "only when the outlined support is genuinely mixed or visually undecidable.\n\nOptions:\n"
        f"{options_text(class_order)}\n\nAnswer with only the option letter."
    )


def blind_evidence_prompt():
    return (
        "Inspect the RED-outlined point-cloud surface support in this real indoor TLS image. The image contains "
        "one or more target-centered RECTILINEAR panels reprojected from the same spherical panorama; all panels "
        "refer to the same target. Extract only observable evidence without assigning an architectural class. "
        "Do not infer unseen substrate and do not provide hidden chain-of-thought. Return exactly one JSON object "
        "and no markdown:\n"
        "{\n"
        "  \"target_shape_evidence\": \"one concise sentence about visible shape and orientation\",\n"
        "  \"target_boundary_attachment_evidence\": \"one concise sentence about visible boundaries and attachments\",\n"
        "  \"target_context_evidence\": \"one concise sentence distinguishing the target from its surroundings\",\n"
        "  \"occlusion_mixing_evidence\": \"one concise sentence about occlusion or mixed-object support\",\n"
        "  \"evidence_limitations\": \"one concise sentence about what the image cannot establish\"\n"
        "}"
    )


def parse_evidence(text):
    match = re.search(r"\{.*\}", text.strip(), flags=re.DOTALL)
    try:
        parsed = json.loads(match.group(0) if match else text.strip())
    except (json.JSONDecodeError, AttributeError):
        return {"stage1_evidence_parse_status": "invalid_json", **{field: "" for field in EVIDENCE_FIELDS}}
    return {
        "stage1_evidence_parse_status": "ok",
        **{field: str(parsed.get(field, "")).strip() for field in EVIDENCE_FIELDS},
    }


def generate_evidence(processor, model, image):
    message = [{"role": "user", "content": [
        {"type": "image", "image": image},
        {"type": "text", "text": blind_evidence_prompt()},
    ]}]
    chat = processor.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[chat], images=[image], return_tensors="pt").to(model.device)
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=240, do_sample=False)
    output = output[:, inputs["input_ids"].shape[1]:]
    return processor.batch_decode(
        output, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0].strip()


def score_image_prompt(processor, model, option_token_ids, image, prompt, class_order):
    message = [{"role": "user", "content": [
        {"type": "image", "image": image},
        {"type": "text", "text": prompt},
    ]}]
    chat = processor.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[chat], images=[image], return_tensors="pt").to(model.device)
    with torch.no_grad():
        output = model(**inputs)
    logits = output.logits[0, -1, option_token_ids].float().cpu().numpy()
    option_probabilities = np.exp(logits - logits.max())
    option_probabilities /= option_probabilities.sum()
    probabilities = np.zeros(len(semantic_v02.SEMANTIC_CLASSES), dtype=np.float64)
    for probability, label in zip(option_probabilities, class_order):
        probabilities[semantic_v02.SEMANTIC_CLASSES.index(label)] = probability
    return probabilities


def format_value(value, digits=3):
    value = float(value)
    return "not available" if not np.isfinite(value) else f"{value:.{digits}f}"


def metric_geometry_text(row):
    schema = str(row["dimension_schema"])
    if schema == "vertical_width_height":
        dimensions = (
            f"vertical-plane observed width={format_value(row['observed_width_m'])} m; "
            f"observed height={format_value(row['observed_height_m'])} m; "
            f"width/height={format_value(row['observed_aspect_ratio'], 2)}"
        )
    elif schema == "horizontal_length_width":
        dimensions = (
            f"horizontal-plane observed major length={format_value(row['observed_length_m'])} m; "
            f"minor width={format_value(row['observed_width_m'])} m; "
            f"length/width={format_value(row['observed_aspect_ratio'], 2)}"
        )
    else:
        dimensions = (
            f"oblique-plane observed major extent={format_value(row['observed_major_m'])} m; "
            f"minor extent={format_value(row['observed_minor_m'])} m; "
            f"major/minor={format_value(row['observed_aspect_ratio'], 2)}"
        )
    return (
        "- World frame: registered scan X-Y is horizontal and +Z is upward.\n"
        f"- Measured planar orientation: {row['geometry_role_evidence']}; unit normal="
        f"({format_value(row['plane_normal_x'], 4)}, {format_value(row['plane_normal_y'], 4)}, "
        f"{format_value(row['plane_normal_z'], 4)}); tilt from horizontal="
        f"{format_value(row['plane_tilt_from_horizontal_deg'], 2)} deg; normal azimuth modulo 180="
        f"{format_value(row['plane_normal_azimuth_mod180_deg'], 2)} deg.\n"
        f"- Direction-appropriate dimensions: {dimensions}.\n"
        f"- Observed support area: {format_value(row['area_sum_m2'])} m2; support is assembled from "
        f"{int(row['n_atomic_patches'])} atomic patches and {int(row['n_member_local_surfaces'])} local surfaces "
        f"observed at {int(row['n_stations'])} station(s).\n"
        f"- Station vertical references: floor Z={format_value(row['station_floor_reference_z_m'])} m; "
        f"highest upper reference Z={format_value(row['station_top_reference_z_m'])} m.\n"
        f"- Target world-Z distribution: weighted mean={format_value(row['z_world_mean_m'])} m, "
        f"SD={format_value(row['z_world_sd_m'])} m, 2.5/50/97.5 percentiles="
        f"{format_value(row['z_world_q025_m'])}/{format_value(row['z_world_q50_m'])}/"
        f"{format_value(row['z_world_q975_m'])} m.\n"
        "- Normalized height uses floor=0 and highest upper reference=1: "
        f"weighted mean={format_value(row['height_norm_mean'])}, SD={format_value(row['height_norm_sd'])}, "
        f"2.5/50/97.5 percentiles={format_value(row['height_norm_q025'])}/"
        f"{format_value(row['height_norm_q50'])}/{format_value(row['height_norm_q975'])}.\n"
        "- Weighted support fractions by normalized height: "
        f"below 0={format_value(row['fraction_below_reference'])}; 0-0.10="
        f"{format_value(row['fraction_bottom_00_10'])}; 0.10-0.30="
        f"{format_value(row['fraction_lower_10_30'])}; 0.30-0.70="
        f"{format_value(row['fraction_middle_30_70'])}; 0.70-0.90="
        f"{format_value(row['fraction_upper_70_90'])}; 0.90-1.00="
        f"{format_value(row['fraction_top_90_100'])}; above 1="
        f"{format_value(row['fraction_above_reference'])}."
    )


def joint_prompt(row, evidence, class_order):
    evidence_text = "\n".join(
        f"- {field.replace('_', ' ')}: {evidence[field]}" for field in EVIDENCE_FIELDS
    )
    return (
        "Classify the RED-outlined point-cloud surface support. Reinspect the attached target-centered RECTILINEAR "
        "panel or montage, then use the fallible Stage-1 observations and neutral LiDAR measurements below. All "
        "panels refer to the same target. Correct the Stage-1 text when the image disagrees. Geometry describes "
        "measured orientation, extent, and elevation but does not name the object. No previous semantic class, "
        "route-compatible label list, or review decision is provided. Identify architectural or visible-object "
        "role, not material or unseen substrate.\n\nStage-1 observable image evidence:\n"
        f"{evidence_text}\n\nNeutral metric LiDAR evidence:\n{metric_geometry_text(row)}\n\nOptions:\n"
        f"{options_text(class_order)}\n\nAnswer with only the option letter."
    )


def absolute_horizontal_lock(row):
    if str(row["geometry_role_evidence"]) != "horizontal_plane":
        return None
    if float(row["height_norm_q975"]) <= 0.10:
        return "floor"
    if float(row["height_norm_q025"]) >= 0.65:
        return "ceiling"
    return None


def visual_geometry_compatible(row, label):
    lock = absolute_horizontal_lock(row)
    if lock:
        return label == lock
    role = str(row["geometry_role_evidence"])
    if role in {"vertical_plane", "oblique_plane"}:
        return label not in {"floor", "ceiling", "unknown"}
    return label != "unknown"


def soft_geometry_weights(row, penalty):
    weights = np.ones(len(semantic_v02.SEMANTIC_CLASSES), dtype=np.float64)
    role = str(row["geometry_role_evidence"])
    lock = absolute_horizontal_lock(row)
    if lock:
        weights *= penalty
        weights[semantic_v02.SEMANTIC_CLASSES.index(lock)] = 1.0
    elif role in {"vertical_plane", "oblique_plane"}:
        weights[semantic_v02.SEMANTIC_CLASSES.index("floor")] = penalty
        weights[semantic_v02.SEMANTIC_CLASSES.index("ceiling")] = penalty
    elif role == "horizontal_plane":
        for label in ("wall", "column", "door", "window"):
            weights[semantic_v02.SEMANTIC_CLASSES.index(label)] = penalty
    return weights


def normalize_concrete(probabilities):
    result = probabilities.copy()
    result[semantic_v02.SEMANTIC_CLASSES.index("unknown")] = 0.0
    if result.sum() <= 0:
        result[:] = 0.0
        result[semantic_v02.SEMANTIC_CLASSES.index("other")] = 1.0
    else:
        result /= result.sum()
    return result


def bounded_decision(row, visual, joint, args):
    lock = absolute_horizontal_lock(row)
    visual_index = int(np.argmax(visual))
    visual_label = semantic_v02.SEMANTIC_CLASSES[visual_index]
    if lock:
        decision = np.zeros_like(visual)
        decision[semantic_v02.SEMANTIC_CLASSES.index(lock)] = 1.0
        return decision, lock, "quantitative_horizontal_height_lock"
    if (
        visual_label != "unknown"
        and float(visual[visual_index]) >= args.visual_anchor_threshold
        and visual_geometry_compatible(row, visual_label)
    ):
        return normalize_concrete(visual), visual_label, "high_confidence_compatible_rectilinear_anchor"
    geometry = soft_geometry_weights(row, args.geometry_soft_penalty)
    log_fused = (
        (1.0 - args.joint_weight) * np.log(np.clip(visual, EPSILON, 1.0))
        + args.joint_weight * np.log(np.clip(joint, EPSILON, 1.0))
        + np.log(geometry)
    )
    fused = np.exp(log_fused - log_fused.max())
    fused /= fused.sum()
    decision = normalize_concrete(fused)
    index = int(np.argmax(decision))
    return decision, semantic_v02.SEMANTIC_CLASSES[index], "bounded_rectilinear_joint_metric_geometry"


def top_summary(prefix, probabilities):
    first, second = semantic_v02.top_two(probabilities)
    return {
        f"{prefix}_top1": semantic_v02.SEMANTIC_CLASSES[first],
        f"{prefix}_top1_probability": round(float(probabilities[first]), 8),
        f"{prefix}_top2": semantic_v02.SEMANTIC_CLASSES[second],
        f"{prefix}_top1_margin": round(float(probabilities[first] - probabilities[second]), 8),
        f"{prefix}_entropy_normalized": round(semantic_v02.normalized_entropy(probabilities), 8),
        **semantic_v02.probability_fields(f"{prefix}_p", probabilities),
    }


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing output directory: {output_dir}")
    projection = pd.read_csv(Path(args.projection_dir) / "rectilinear_metric_evidence.csv")
    baseline = pd.read_csv(Path(args.panorama_baseline_dir) / "neutral_geometry_joint_scores.csv")
    key = "provisional_geometric_element_id"
    old_columns = [
        key, "conditioned_option_order", "current_final_provisional", "geometry_role_evidence",
        "baseline_image_only_top1", "baseline_image_only_top1_probability",
        "detail_context_overlay_path", "full_panorama_overlay_path", "area_sum_m2",
        "n_atomic_patches", "n_member_local_surfaces", "n_stations",
    ] + [f"baseline_p_{label}" for label in semantic_v02.SEMANTIC_CLASSES]
    rows = projection.merge(baseline[old_columns], on=key, how="left", validate="one_to_one")
    if len(rows) != 16 or rows["conditioned_option_order"].isna().any():
        raise AssertionError("Expected the complete 16-record rectilinear smoke input")

    processor, model, option_token_ids = load_model(args.model_id)
    results = []
    try:
        for ordinal, (_, row) in enumerate(rows.sort_values(key).iterrows(), start=1):
            image_path = Path(row["rectilinear_overlay_path"])
            if not image_path.is_file():
                raise FileNotFoundError(image_path)
            image = resize_image(image_path, args.max_image_px)
            class_order = str(row["conditioned_option_order"]).split("|")
            if set(class_order) != set(semantic_v02.SEMANTIC_CLASSES):
                raise AssertionError("Option order does not contain the full taxonomy")
            rectilinear = score_image_prompt(
                processor, model, option_token_ids, image,
                rectilinear_image_prompt(class_order), class_order,
            )
            evidence_raw = generate_evidence(processor, model, image)
            evidence = parse_evidence(evidence_raw)
            joint = score_image_prompt(
                processor, model, option_token_ids, image,
                joint_prompt(row, evidence, class_order), class_order,
            )
            decision, provisional, source = bounded_decision(row, rectilinear, joint, args)
            result = row.to_dict()
            result.update({
                "stage1_evidence_raw": evidence_raw,
                **evidence,
                "metric_geometry_card": metric_geometry_text(row),
                "qwen_option_order": "|".join(class_order),
                **top_summary("rectilinear_image", rectilinear),
                **top_summary("rectilinear_joint", joint),
                **top_summary("final_decision", decision),
                "provisional_semantic_class": provisional,
                "provisional_decision_source": source,
                "visual_anchor_threshold": args.visual_anchor_threshold,
                "joint_weight": args.joint_weight,
                "geometry_soft_penalty": args.geometry_soft_penalty,
                "unknown_excluded_from_final": True,
                "rectilinear_changed_from_panorama_image": provisional != row["baseline_image_only_top1"],
                "provisional_changed_from_prior_smoke_final": provisional != row["current_final_provisional"],
                "experiment_status": "uncalibrated_no_ground_truth",
            })
            results.append(result)
            log(
                f"Rectilinear {ordinal}/16: geom-{int(row[key]):04d} "
                f"pano={row['baseline_image_only_top1']} image={result['rectilinear_image_top1']} "
                f"joint={result['rectilinear_joint_top1']} final={provisional} ({source})"
            )
    finally:
        del model, processor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    result = pd.DataFrame(results)
    if result["provisional_semantic_class"].eq("unknown").any():
        raise AssertionError("Final provisional class must be concrete")
    probability_prefixes = ("baseline", "rectilinear_image", "rectilinear_joint", "final_decision")
    for prefix in probability_prefixes:
        columns = [f"{prefix}_p_{label}" for label in semantic_v02.SEMANTIC_CLASSES]
        if not np.allclose(result[columns].sum(axis=1), 1.0, atol=1e-5):
            raise AssertionError(f"{prefix} probabilities do not sum to one")
    output_dir.mkdir(parents=True)
    result.to_csv(output_dir / "rectilinear_metric_joint_scores.csv", index=False, encoding="utf-8")
    manifest = {
        "record_version": "Step05c_RectilinearMetricJointScoring01",
        "n_records": int(len(result)),
        "model_id": args.model_id,
        "max_image_px": args.max_image_px,
        "n_valid_stage1_evidence_json": int(result["stage1_evidence_parse_status"].eq("ok").sum()),
        "visual_anchor_threshold": args.visual_anchor_threshold,
        "joint_weight": args.joint_weight,
        "geometry_soft_penalty": args.geometry_soft_penalty,
        "n_rectilinear_image_top1_changed_from_panorama": int(
            result["rectilinear_image_top1"].ne(result["baseline_image_only_top1"]).sum()
        ),
        "n_joint_top1_changed_from_rectilinear_image": int(
            result["rectilinear_joint_top1"].ne(result["rectilinear_image_top1"]).sum()
        ),
        "n_final_changed_from_prior_smoke_final": int(result["provisional_changed_from_prior_smoke_final"].sum()),
        "provisional_class_counts": {
            str(k): int(v) for k, v in result["provisional_semantic_class"].value_counts().sort_index().items()
        },
        "decision_source_counts": {
            str(k): int(v) for k, v in result["provisional_decision_source"].value_counts().sort_index().items()
        },
        "scope": (
            "Sixteen station-5 smoke records. Stage 1 uses target-centered rectilinear imagery; Stage 2 sees the "
            "same imagery, observable evidence, and neutral metric geometry. No previous class or compatible-label "
            "list is input."
        ),
        "interpretation_limit": "No independent semantic GT; all Qwen option probabilities are uncalibrated.",
    }
    (output_dir / "rectilinear_metric_joint_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
