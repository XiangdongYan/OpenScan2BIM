"""Seven-class region-referenced Final policy (Framework v0.1, Layer 2.3).

`other` is removed and its probability is added to furniture; every station uses
the region reference (z_f, upper references, z_u = z_f + c). Each change proposed
in Framework Section 6 is a separate switch, so it can be tested on dev one at a
time. With every switch at its "v09" setting the rules equal v09:

  ceiling_rule        band  : support meets an upper reference +/- 0.25 m, story coverage <= 0.40
                      zone  : median z >= z_u
  wall_support        v09   : >= 5 % support in the lower 70 % of the station height
                      zone  : z025 < z_u
  wall_min_area_m2    2.0 (v09) or None
  column_rule         v09   : floor- and upper-reference anchored, story coverage >= 0.85, area <= 5 m2
                      zone  : floor anchored, z975 >= z_u, face width <= 2.0 m
  door_rule           v09   : vertical, area >= 2 m2
                      envelope: vertical, |z025 - z_f| <= 0.15 m, span <= 2.6 m, width <= 2.2 m
  window_min_area_m2  2.0 (v09) or None
  furniture_max_area_m2  2.0 (v09) or None
  furniture_zone      False (v09) or True: median z < z_u and not a floor-level horizontal
  routes              True (v09): column envelope -> column; compact floor-supported -> furniture
  prob_source         joint (Stage 2) or image (Stage 1)
"""

import numpy as np


CLASSES = ("wall", "floor", "ceiling", "column", "door", "window", "furniture")
PROB_PREFIX = {"joint": "joint_p", "image": "rectilinear_image_p"}

V09_CONFIG = {
    "prob_source": "joint", "ceiling_rule": "band", "wall_support": "v09", "wall_min_area_m2": 2.0,
    "column_rule": "v09", "door_rule": "v09", "window_min_area_m2": 2.0, "furniture_max_area_m2": 2.0,
    "furniture_zone": False, "routes": True,
}
# Selected on dev by the point-level ablation run_final_policy_ablation02.py (outputs249): variant F2.
DEFAULT_CONFIG = {**V09_CONFIG, "wall_support": "zone", "column_rule": "zone", "door_rule": "envelope",
                  "furniture_zone": True}

REFERENCE_TOLERANCE_M = 0.25
MAX_CEILING_STORY_COVERAGE = 0.40
MIN_WALL_VERTICAL_SPAN_M = 0.30
MIN_WALL_LOWER_SUPPORT = 0.05
MIN_COLUMN_STORY_COVERAGE = 0.85
MAX_COLUMN_AREA_M2 = 5.00
MAX_COLUMN_FACE_WIDTH_M = 2.00
DOOR_BOTTOM_TOLERANCE_M = 0.15
MAX_DOOR_VERTICAL_SPAN_M = 2.60
MAX_DOOR_WIDTH_M = 2.20
FURNITURE_ROUTE_MAX_AREA_M2 = 2.00
PROB_FLOOR = 1e-9


def probabilities(row, source):
    prefix = PROB_PREFIX[source]
    p = {label: float(row[f"{prefix}_{label}"]) for label in CLASSES}
    p["furniture"] += float(row[f"{prefix}_other"])
    return np.array([p[label] for label in CLASSES])


def geometry(row):
    width = row["horizontal_in_plane_width_m"]
    return {
        "role": str(row["geometry_role_evidence"]),
        "z025": float(row["z_world_q025_m"]), "z50": float(row["z_world_q50_m"]), "z975": float(row["z_world_q975_m"]),
        "span": float(row["vertical_span_m"]),
        "width": float(width) if width == width else None,
        "area": float(row["element_observed_union_area_m2"]),
        "story_coverage": float(row["story_coverage"]),
        "lower_support": sum(float(row[c]) for c in ("fraction_bottom_00_10", "fraction_lower_10_30", "fraction_middle_30_70")),
    }


def floor_anchored(g, refs):
    return abs(g["z025"] - refs["z_f"]) <= REFERENCE_TOLERANCE_M


def upper_anchored(g, refs):
    return any(abs(g["z975"] - z) <= REFERENCE_TOLERANCE_M for z in refs["upper"])


def floor_level_horizontal(g, refs):
    return g["role"] == "horizontal_plane" and abs(g["z50"] - refs["z_f"]) <= REFERENCE_TOLERANCE_M


def column_envelope(g, refs, cfg):
    if g["role"] not in {"vertical_plane", "oblique_plane"} or not floor_anchored(g, refs):
        return False
    if cfg["column_rule"] == "zone":
        return g["z975"] >= refs["z_u"] and g["width"] is not None and g["width"] <= MAX_COLUMN_FACE_WIDTH_M
    return (upper_anchored(g, refs) and g["story_coverage"] >= MIN_COLUMN_STORY_COVERAGE
            and g["area"] <= MAX_COLUMN_AREA_M2)


def furniture_envelope(g, refs):
    return (g["role"] in {"vertical_plane", "oblique_plane"} and floor_anchored(g, refs)
            and not upper_anchored(g, refs) and g["span"] >= MIN_WALL_VERTICAL_SPAN_M
            and g["area"] <= FURNITURE_ROUTE_MAX_AREA_M2)


def compatible(label, g, refs, cfg):
    if label == "floor":
        return floor_level_horizontal(g, refs)
    if label == "ceiling":
        if cfg["ceiling_rule"] == "zone":
            return g["z50"] >= refs["z_u"]
        near_upper = any(g["z025"] - REFERENCE_TOLERANCE_M <= z <= g["z975"] + REFERENCE_TOLERANCE_M for z in refs["upper"])
        return near_upper and g["story_coverage"] <= MAX_CEILING_STORY_COVERAGE
    if label == "wall":
        support = (g["z025"] < refs["z_u"]) if cfg["wall_support"] == "zone" else (g["lower_support"] >= MIN_WALL_LOWER_SUPPORT)
        return (g["role"] == "vertical_plane" and g["span"] >= MIN_WALL_VERTICAL_SPAN_M and support
                and (cfg["wall_min_area_m2"] is None or g["area"] >= cfg["wall_min_area_m2"])
                and not (cfg["routes"] and column_envelope(g, refs, cfg)))
    if label == "column":
        return column_envelope(g, refs, cfg)
    if label == "door":
        if g["role"] != "vertical_plane":
            return False
        if cfg["door_rule"] == "envelope":
            return (abs(g["z025"] - refs["z_f"]) <= DOOR_BOTTOM_TOLERANCE_M and g["span"] <= MAX_DOOR_VERTICAL_SPAN_M
                    and g["width"] is not None and g["width"] <= MAX_DOOR_WIDTH_M)
        return g["area"] >= 2.0
    if label == "window":
        return (g["role"] in {"vertical_plane", "oblique_plane"}
                and (cfg["window_min_area_m2"] is None or g["area"] >= cfg["window_min_area_m2"]))
    if label == "furniture":
        if cfg["furniture_max_area_m2"] is not None and g["area"] > cfg["furniture_max_area_m2"]:
            return False
        return not cfg["furniture_zone"] or (g["z50"] < refs["z_u"] and not floor_level_horizontal(g, refs))
    raise ValueError(label)


def constrain_row(row, refs, cfg):
    """Return the station-image Final probabilities and the decision trace."""
    g = geometry(row)
    p = probabilities(row, cfg["prob_source"])
    mask = np.array([compatible(label, g, refs, cfg) for label in CLASSES])
    if cfg["routes"] and column_envelope(g, refs, cfg):
        final, status = np.eye(len(CLASSES))[CLASSES.index("column")], "column_route"
    elif cfg["routes"] and furniture_envelope(g, refs):
        final, status = np.eye(len(CLASSES))[CLASSES.index("furniture")], "furniture_route"
    elif not mask.any():
        default = "ceiling" if g["z50"] >= refs["z_u"] else "furniture"
        final, status = np.eye(len(CLASSES))[CLASSES.index(default)], "zone_default_no_compatible_class"
    else:
        final = (p + PROB_FLOOR) * mask
        final /= final.sum()
        status = "masked_probability"
    label = CLASSES[int(np.argmax(final))]
    top = CLASSES[int(np.argmax(p))]
    return {
        "input_top1": top,
        "incompatible_classes": "|".join(c for c, ok in zip(CLASSES, mask) if not ok),
        "decision_route": status,
        "geometry_constraint_status": "unchanged" if label == top else "reassigned",
        "final_semantic_class": label,
        **{f"final_p_{c}": float(v) for c, v in zip(CLASSES, final)},
    }


def policy_manifest(cfg):
    return {"classes": list(CLASSES), "other_merged_into": "furniture", "config": cfg,
            "reference_tolerance_m": REFERENCE_TOLERANCE_M,
            "rule_definitions": __doc__.split("v09:", 1)[1].strip(),
            "no_compatible_class": "zone default: ceiling if median z >= z_u else furniture"}
