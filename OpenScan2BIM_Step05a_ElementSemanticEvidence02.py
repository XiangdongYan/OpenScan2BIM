"""Element-anchored semantic evidence over the station-5/6 geometry record.

This module creates uncalibrated semantic evidence only. It never changes a
geometric association, creates a finish region, assigns material, or authors IFC.
"""

import argparse
import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image


ELEMENT_RECORD_DIR = "outputs67_step04b_element_records02_station5_6"
PROJECTION_DIR = "outputs69_step04c_panorama_projection01_station5_6"
OUT_DIR = "outputs79_step05a_element_semantic02_station5_6"
MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"
SEED = 20260916

SEMANTIC_CLASSES = (
    "wall", "floor", "ceiling", "column", "door", "window",
    "furniture", "other", "unknown",
)
OPTION_LETTERS = tuple("ABCDEFGHI")
CLASS_TEXT = {
    "wall": "a wall or planar architectural partition",
    "floor": "a floor or walking surface",
    "ceiling": "a ceiling or overhead soffit surface",
    "column": "a freestanding structural column or column face",
    "door": "a door or doorway",
    "window": "a window or glazed opening",
    "furniture": "furniture, movable equipment, or an interior object",
    "other": "another fixed building object not described above",
    "unknown": "unknown, mixed, or not visually decidable",
}
FIXTURE_EXCEPTION = {
    3: "attached_fixture_candidate_from_step04e_geom0003_pilot",
}


def log(message):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description="OpenScan2BIM Step05a element semantic evidence")
    parser.add_argument("--element_record_dir", default=ELEMENT_RECORD_DIR)
    parser.add_argument("--projection_dir", default=PROJECTION_DIR)
    parser.add_argument("--output_dir", default=OUT_DIR)
    parser.add_argument("--model_id", default=MODEL_ID)
    parser.add_argument("--max_image_px", type=int, default=896)
    parser.add_argument("--limit_elements", type=int, default=0,
                        help="Process the first N elements only; 0 means every element.")
    return parser.parse_args()


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_seed(value):
    digest = hashlib.blake2b(value.encode("utf-8"), digest_size=4).digest()
    return SEED ^ int.from_bytes(digest, "little")


def source_tokens(value):
    return {part.strip().lower() for part in str(value).split("|") if part.strip() and part != "nan"}


def geometry_route(element):
    """Return a deterministic geometry routing decision, not a final semantic label."""
    role = str(element["geometry_role_evidence"])
    states = source_tokens(element["member_geometry_states"])
    elevations = source_tokens(element["member_elevation_bands"])
    contexts = source_tokens(element["member_local_occupancy_contexts"])

    if role == "horizontal_plane":
        if "floor" in states or "floor_reference" in elevations or "lower" in contexts:
            return "geometry_first_horizontal", "floor", SEMANTIC_CLASSES
        if "ceiling_level" in states or "upper" in elevations or "near_upper_reference" in contexts:
            return "geometry_first_horizontal", "ceiling", SEMANTIC_CLASSES
        return "horizontal_geometry_ambiguous", "unknown", SEMANTIC_CLASSES

    if role == "vertical_plane":
        allowed = tuple(label for label in SEMANTIC_CLASSES if label not in {"floor", "ceiling"})
        return "vertical_geometry_constrained_vision", "unknown", allowed

    return "vision_first", "unknown", SEMANTIC_CLASSES


def normalized_entropy(probabilities):
    valid = probabilities[probabilities > 0]
    entropy = -float(np.sum(valid * np.log(valid)))
    return entropy / np.log(len(probabilities))


def top_two(probabilities):
    order = np.argsort(-probabilities)
    return int(order[0]), int(order[1])


class QwenSemanticScorer:
    """Constrained-choice Qwen scoring with option logits retained as raw evidence."""

    def __init__(self, model_id, max_image_px):
        from transformers import AutoProcessor
        try:
            from transformers import AutoModelForImageTextToText as ModelClass
        except ImportError:
            from transformers import Qwen3VLForConditionalGeneration as ModelClass

        log(f"Loading {model_id}")
        self.processor = AutoProcessor.from_pretrained(model_id, local_files_only=True)
        self.model = ModelClass.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="cuda" if torch.cuda.is_available() else "cpu",
            local_files_only=True,
        ).eval()
        self.max_image_px = max_image_px
        self.option_token_ids = []
        for letter in OPTION_LETTERS:
            ids = self.processor.tokenizer(letter, add_special_tokens=False).input_ids
            if len(ids) != 1:
                raise RuntimeError(f"Option {letter!r} is not a single tokenizer token: {ids}")
            self.option_token_ids.append(ids[0])

    @staticmethod
    def prompt(class_order):
        options = "\n".join(
            f"{letter}. {CLASS_TEXT[label]}"
            for letter, label in zip(OPTION_LETTERS, class_order)
        )
        return (
            "You are inspecting a real building interior crop from a terrestrial laser scanner panorama. "
            "The RED outline traces the exact point-cloud surface support under evaluation. The rest of the "
            "image provides context only. Identify the architectural or visible-object role of the outlined "
            "support, not its material and not an unseen structural substrate. If the outline mixes multiple "
            "objects or the role cannot be determined visually, choose unknown.\n\n"
            f"{options}\n\nAnswer with only the option letter."
        )

    def score(self, image_path, seed):
        class_order = list(SEMANTIC_CLASSES)
        random.Random(seed).shuffle(class_order)
        image = Image.open(image_path).convert("RGB")
        if max(image.size) > self.max_image_px:
            scale = self.max_image_px / max(image.size)
            image = image.resize(
                (max(1, round(image.width * scale)), max(1, round(image.height * scale))),
                Image.LANCZOS,
            )
        message = [{"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": self.prompt(class_order)},
        ]}]
        chat = self.processor.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[chat], images=[image], return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            output = self.model(**inputs)
        logits = output.logits[0, -1, self.option_token_ids].float().cpu().numpy()
        option_probabilities = np.exp(logits - logits.max())
        option_probabilities /= option_probabilities.sum()
        probabilities = np.zeros(len(SEMANTIC_CLASSES), dtype=np.float64)
        for probability, label in zip(option_probabilities, class_order):
            probabilities[SEMANTIC_CLASSES.index(label)] = probability
        return probabilities, class_order

    def close(self):
        del self.model, self.processor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def constrain_probabilities(probabilities, allowed):
    mask = np.array([label in allowed for label in SEMANTIC_CLASSES], dtype=bool)
    constrained = np.where(mask, probabilities, 0.0)
    total = float(constrained.sum())
    if total <= 0:
        return np.full_like(probabilities, 1.0 / len(probabilities))
    return constrained / total


def view_weight(row):
    """Area x point-quality x observed-fraction; this is evidence pooling, not calibration."""
    area = max(float(row["area_m2"]), 0.01)
    quality = max(float(row["mean_point_quality"]), 0.0)
    observed_fraction = max(float(row["projected_point_fraction_in_region"]), 0.0)
    return np.sqrt(area) * quality * observed_fraction


def probability_fields(prefix, probabilities):
    return {f"{prefix}_{label}": round(float(probability), 8)
            for label, probability in zip(SEMANTIC_CLASSES, probabilities)}


def make_view_evidence(elements, members, regions, scorer):
    member_fields = members[[
        "member_record_id", "area_m2", "mean_w", "geom_state", "planar_orientation",
        "local_occupancy_context", "elevation_band", "z_ref_m", "zspan_m", "resid_rms_m",
    ]]
    evidence = regions.merge(member_fields, on="member_record_id", how="left", validate="one_to_one")
    if evidence["area_m2"].isna().any():
        raise ValueError("A projected view evidence record has no Step04b local-surface member record")

    element_lookup = elements.set_index("provisional_geometric_element_id").to_dict("index")
    rows = []
    for index, region in evidence.sort_values(
            ["provisional_geometric_element_id", "station_id", "surface_id"]).iterrows():
        element_id = int(region["provisional_geometric_element_id"])
        route, geometry_candidate, allowed = geometry_route(element_lookup[element_id])
        audit_image = Path(region["audit_overlay_path"])
        if not audit_image.is_file():
            raise FileNotFoundError(audit_image)
        probabilities, class_order = scorer.score(audit_image, stable_seed(region["evidence_region_id"]))
        constrained = constrain_probabilities(probabilities, allowed)
        top_index, second_index = top_two(probabilities)
        constrained_top_index, constrained_second_index = top_two(constrained)
        row = {
            "evidence_region_id": region["evidence_region_id"],
            "provisional_geometric_element_id": element_id,
            "element_record_id": f"geom-{element_id:04d}",
            "station_id": region["station_id"],
            "surface_id": int(region["surface_id"]),
            "member_record_id": region["member_record_id"],
            "audit_overlay_path": region["audit_overlay_path"],
            "native_crop_path": region["native_crop_path"],
            "projection_status": region["projection_status"],
            "area_m2": round(float(region["area_m2"]), 6),
            "mean_point_quality": round(float(region["mean_point_quality"]), 6),
            "projected_point_fraction_in_region": round(float(region["projected_point_fraction_in_region"]), 6),
            "semantic_evidence_weight": round(view_weight(region), 8),
            "geometry_route": route,
            "geometry_semantic_candidate": geometry_candidate,
            "allowed_classes": "|".join(allowed),
            "qwen_option_order": "|".join(class_order),
            "vision_top1_raw": SEMANTIC_CLASSES[top_index],
            "vision_top1_raw_probability": round(float(probabilities[top_index]), 8),
            "vision_top2_raw": SEMANTIC_CLASSES[second_index],
            "vision_top1_raw_margin": round(float(probabilities[top_index] - probabilities[second_index]), 8),
            "vision_entropy_normalized": round(normalized_entropy(probabilities), 8),
            "vision_top1_geometry_constrained": SEMANTIC_CLASSES[constrained_top_index],
            "vision_top1_geometry_constrained_probability": round(float(constrained[constrained_top_index]), 8),
            "vision_top2_geometry_constrained": SEMANTIC_CLASSES[constrained_second_index],
            "vision_top1_geometry_constrained_margin": round(
                float(constrained[constrained_top_index] - constrained[constrained_second_index]), 8),
            "semantic_uq_status": "raw_option_logits_only_uncalibrated",
        }
        row.update(probability_fields("vision_raw_p", probabilities))
        row.update(probability_fields("vision_geometry_constrained_p", constrained))
        rows.append(row)
        if (index + 1) % 20 == 0:
            log(f"Scored {index + 1}/{len(evidence)} projected evidence regions")
    return pd.DataFrame(rows)


def make_element_records(elements, view_evidence):
    rows = []
    for _, element in elements.sort_values("provisional_geometric_element_id").iterrows():
        element_id = int(element["provisional_geometric_element_id"])
        evidence = view_evidence[
            view_evidence["provisional_geometric_element_id"] == element_id
        ].copy()
        route, geometry_candidate, _ = geometry_route(element)
        weights = evidence["semantic_evidence_weight"].to_numpy(dtype=np.float64)
        if len(evidence) == 0 or weights.sum() <= 0:
            pooled_raw = np.full(len(SEMANTIC_CLASSES), 1.0 / len(SEMANTIC_CLASSES))
            pooled_constrained = pooled_raw.copy()
            weights = np.ones(len(evidence), dtype=np.float64)
        else:
            weights /= weights.sum()
            pooled_raw = np.sum(np.vstack([
                evidence[f"vision_raw_p_{label}"].to_numpy(dtype=np.float64)
                for label in SEMANTIC_CLASSES
            ]).T * weights[:, None], axis=0)
            pooled_constrained = np.sum(np.vstack([
                evidence[f"vision_geometry_constrained_p_{label}"].to_numpy(dtype=np.float64)
                for label in SEMANTIC_CLASSES
            ]).T * weights[:, None], axis=0)
        raw_top_index, raw_second_index = top_two(pooled_raw)
        constrained_top_index, constrained_second_index = top_two(pooled_constrained)
        visual_top = SEMANTIC_CLASSES[constrained_top_index]
        semantic_label = geometry_candidate if geometry_candidate != "unknown" else visual_top
        reasons = []
        if element_id in FIXTURE_EXCEPTION:
            reasons.append(FIXTURE_EXCEPTION[element_id])
        if geometry_candidate != "unknown" and raw_top_index != SEMANTIC_CLASSES.index(geometry_candidate):
            reasons.append("geometry_visual_top1_disagreement")
        if geometry_candidate == "unknown" and visual_top == "unknown":
            reasons.append("visual_role_unresolved")
        status = "provisional_unvalidated" if not reasons else "review_recommended"
        row = {
            "element_record_id": f"geom-{element_id:04d}",
            "provisional_geometric_element_id": element_id,
            "n_view_evidence_regions": int(len(evidence)),
            "n_stations": int(element["n_stations"]),
            "geometry_role_evidence": element["geometry_role_evidence"],
            "member_geometry_states": element["member_geometry_states"],
            "member_planar_orientations": element["member_planar_orientations"],
            "member_elevation_bands": element["member_elevation_bands"],
            "geometry_route": route,
            "geometry_semantic_candidate": geometry_candidate,
            "semantic_class_provisional": semantic_label,
            "semantic_decision_mode": (
                "geometry_first" if geometry_candidate != "unknown" else "vision_first"
            ),
            "semantic_evidence_status": status,
            "review_reason": "|".join(reasons) if reasons else "none",
            "visual_top1_raw": SEMANTIC_CLASSES[raw_top_index],
            "visual_top1_raw_probability": round(float(pooled_raw[raw_top_index]), 8),
            "visual_top2_raw": SEMANTIC_CLASSES[raw_second_index],
            "visual_top1_raw_margin": round(float(pooled_raw[raw_top_index] - pooled_raw[raw_second_index]), 8),
            "visual_top1_geometry_constrained": visual_top,
            "visual_top1_geometry_constrained_probability": round(float(pooled_constrained[constrained_top_index]), 8),
            "visual_top2_geometry_constrained": SEMANTIC_CLASSES[constrained_second_index],
            "visual_top1_geometry_constrained_margin": round(
                float(pooled_constrained[constrained_top_index] - pooled_constrained[constrained_second_index]), 8),
            "visual_entropy_normalized": round(normalized_entropy(pooled_constrained), 8),
            "view_top1_agreement_fraction": round(float(
                (evidence["vision_top1_geometry_constrained"] == visual_top).mean()), 8),
            "semantic_uq_status": "raw_logits_and_view_disagreement_uncalibrated",
            "visible_finish_region_status": "not_created",
            "visible_finish_label": "unassigned",
            "material_evidence_status": "not_attached",
            "material_uq_status": "not_computed",
        }
        row.update(probability_fields("pooled_vision_raw_p", pooled_raw))
        row.update(probability_fields("pooled_vision_geometry_constrained_p", pooled_constrained))
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    args = parse_args()
    record_dir = Path(args.element_record_dir)
    projection_dir = Path(args.projection_dir)
    output_dir = Path(args.output_dir)
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing output directory: {output_dir}")

    element_path = record_dir / "provisional_geometric_element_records.csv"
    member_path = record_dir / "local_surface_member_records.csv"
    region_path = projection_dir / "projected_view_evidence_regions.csv"
    for path in (element_path, member_path, region_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    elements = pd.read_csv(element_path)
    members = pd.read_csv(member_path)
    regions = pd.read_csv(region_path)
    if args.limit_elements:
        selected_ids = sorted(elements["provisional_geometric_element_id"].astype(int).tolist())[:args.limit_elements]
        elements = elements[elements["provisional_geometric_element_id"].isin(selected_ids)].copy()
        members = members[members["provisional_geometric_element_id"].isin(selected_ids)].copy()
        regions = regions[regions["provisional_geometric_element_id"].isin(selected_ids)].copy()
    if len(elements) == 0 or len(regions) == 0:
        raise ValueError("No element/view evidence records selected")

    output_dir.mkdir(parents=True)
    scorer = None
    try:
        scorer = QwenSemanticScorer(args.model_id, args.max_image_px)
        view_evidence = make_view_evidence(elements, members, regions, scorer)
    finally:
        if scorer is not None:
            scorer.close()
    element_records = make_element_records(elements, view_evidence)
    review_queue = element_records[element_records["semantic_evidence_status"] == "review_recommended"].copy()
    review_queue = review_queue.merge(
        view_evidence[["provisional_geometric_element_id", "station_id", "surface_id", "audit_overlay_path"]],
        on="provisional_geometric_element_id", how="left",
    )

    view_evidence.to_csv(output_dir / "view_semantic_evidence.csv", index=False, encoding="utf-8")
    element_records.to_csv(output_dir / "element_semantic_records.csv", index=False, encoding="utf-8")
    review_queue.to_csv(output_dir / "semantic_review_queue.csv", index=False, encoding="utf-8")
    manifest = {
        "record_version": "Step05a_ElementSemanticEvidence02",
        "scope": "station 5 and station 6 provisional geometric elements",
        "input_sha256": {
            element_path.name: sha256_file(element_path),
            member_path.name: sha256_file(member_path),
            region_path.name: sha256_file(region_path),
        },
        "model_id": args.model_id,
        "semantic_classes": list(SEMANTIC_CLASSES),
        "geometry_routes": {
            "geometry_first_horizontal": "floor/ceiling candidate from horizontal role plus elevation/context",
            "vertical_geometry_constrained_vision": "floor and ceiling excluded before visual pooling",
            "vision_first": "no structural geometry class forced",
        },
        "view_weight": "sqrt(local_surface_area_m2) * mean_point_quality * projected_point_fraction",
        "uncertainty_status": "raw logits, entropy, and view agreement only; no calibration or conformal claim",
        "not_performed": [
            "geometric association modification",
            "opening/object mask extraction",
            "visible-finish region construction",
            "material classification",
            "probability calibration",
            "conformal prediction",
            "IFC authoring",
        ],
        "n_elements": int(len(element_records)),
        "n_view_evidence_regions": int(len(view_evidence)),
        "n_review_queue_rows": int(len(review_queue)),
    }
    (output_dir / "semantic_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
