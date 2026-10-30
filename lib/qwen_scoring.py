"""Qwen3-VL prompts, scoring and evidence helpers for s08.

Consolidated 2026-10-30; every definition is verbatim from its legacy module,
with only the module-alias prefixes removed (e.g. `semantic_source.X` -> `X`):
  * OpenScan2BIM_Step05c_RectilinearMetricJointScoring03.py: class list and
    option texts, option scoring, probability fields.
  * OpenScan2BIM_Step05a_ElementSemanticEvidence02.py: option letters, seed.
  * OpenScan2BIM_Step05c_RectilinearMetricJointScoring01.py: model loading,
    image resizing, the blind Stage-1 evidence prompt, generation and parsing.
  * OpenScan2BIM_Step05c_RectilinearMetricJointScoring05.py: per-record option
    order and the Stage-1 image-only prompt. Its own `SEMANTIC_CLASSES` and
    `options_text` were aliases/identical copies of the Scoring03 ones above.
  * OpenScan2BIM_Step05c_RectilinearMetricJointScoring06.py: the Stage-2 joint
    prompt with the metric-geometry card.
"""

import hashlib
import json
import random
import re
import time

import numpy as np
import pandas as pd
import torch
from PIL import Image

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


def options_text(class_order):
    return "\n".join(
        f"{letter}. {CLASS_TEXT[label]}"
        for letter, label in zip(OPTION_LETTERS, class_order)
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


# ── from OpenScan2BIM_Step05a_ElementSemanticEvidence02.py ──


SEED = 20260916


OPTION_LETTERS = tuple("ABCDEFGHI")


def stable_seed(value):
    digest = hashlib.blake2b(value.encode("utf-8"), digest_size=4).digest()
    return SEED ^ int.from_bytes(digest, "little")


# ── from OpenScan2BIM_Step05c_RectilinearMetricJointScoring01.py ──


EVIDENCE_FIELDS = (
    "target_shape_evidence",
    "target_boundary_attachment_evidence",
    "target_context_evidence",
    "occlusion_mixing_evidence",
    "evidence_limitations",
)


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
    for letter in OPTION_LETTERS:
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


# ── from OpenScan2BIM_Step05c_RectilinearMetricJointScoring05.py ──


def class_order_for(record_id):
    order = list(SEMANTIC_CLASSES)
    random.Random(stable_seed(f"{record_id}:eight-class-rectilinear")).shuffle(order)
    return order


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


def json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    if pd.isna(value):
        return None
    raise TypeError(type(value).__name__)


# ── from OpenScan2BIM_Step05c_RectilinearMetricJointScoring06.py ──


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
        f"- {field.replace('_', ' ')}: {evidence[field]}" for field in EVIDENCE_FIELDS
    )
    return (
        "Classify the RED-outlined point-cloud surface support. Reinspect the attached target-centered "
        "RECTILINEAR panel or montage, then use the fallible Stage-1 observations and neutral LiDAR measurements "
        "below. All panels refer to the same target. Correct the Stage-1 text when the image disagrees. Geometry "
        "describes measured orientation, extent, elevation, observed area, and completeness but does not name the "
        "object. Identify the surface's architectural or visible-object role, not material or unseen substrate.\n\n"
        "Stage-1 observable image evidence:\n"
        f"{evidence_text}\n\nQuantitative metric geometry:\n{metric_geometry_text(row)}\n\nOptions:\n"
        f"{options_text(class_order)}\n\nAnswer with only the option letter."
    )


# ── option-order averaging (pipeline, 2026-10-30, run R01) ──

def cyclic_option_orders():
    """The cyclic shifts of SEMANTIC_CLASSES: every class takes every option letter exactly once."""
    base = list(SEMANTIC_CLASSES)
    return [base[k:] + base[:k] for k in range(len(base))]


def score_over_orders(processor, model, option_token_ids, image, make_prompt, orders):
    """Class probabilities of one prompt for each option order, and their mean over the orders."""
    per_order = np.array([
        score_prompt(processor, model, option_token_ids, image, make_prompt(order), order) for order in orders
    ])
    return per_order.mean(axis=0), per_order


def order_agreement(per_order, mean):
    """Share of option orders whose top-1 class equals the top-1 of the mean probabilities."""
    return float((per_order.argmax(axis=1) == mean.argmax()).mean())
