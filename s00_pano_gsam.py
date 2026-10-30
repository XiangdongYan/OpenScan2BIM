"""
s00 -- Stage 0: E57 -> panorama + Grounded-SAM masks + labeled PLY, one station per call.

Pipeline version of OpenScan2BIM_Step01_Segmentation09.py (2026-10-30). Not run
for the dev stations: data/stage0_pano_gsam_level02lab/ is the frozen output of
this code (2026-07-10) for 2ndLab-1..7, with yaw_k 4, conf_threshold 0.27,
image_height 4096, min_area 500, overlap 0.6, vote 0.3, persistence 0.5. The
script default conf_threshold is 0.29, so --conf_threshold is now required;
new stations must use the dev value 0.27. --input_e57 and --output_dir are
required as well; segmentation code unchanged. Legacy history follows.

OpenScan2BIM — Step 01: Spherical Panoramic Projection & Multi-Pass Instance Segmentation
==========================================================================================
v09 (2026-07-10, M4) — changes vs OpenScan2BIM_Step01_Segmentation08.py:
  * YAW-ENSEMBLE detection + persistence score.  An equirectangular
    panorama can be yaw-rotated at zero cost (np.roll along the width),
    which changes where the seam cuts the scene and how objects sit in
    GroundingDINO's receptive field.  Each pass now runs the detector on
    K yaw-rotated copies of the residual image (default K=4: 0°, 90°,
    180°, 270°), unrolls the masks back, clusters them across yaws by
    pixel IoU, and keeps clusters seen in ≥ PERSISTENCE_MIN of the yaws.
    persistence = (# yaws detecting the object) / K is stored per
    instance (metadata CSV column "Persistence" + manifest) and is the
    detector-level epistemic-uncertainty input for Step 04 evidence
    weighting and Step 05 calibration (deep-ensembles/TTA reasoning).
    Motivation (EXP-A): detections at τ=0.29 proved configuration-
    sensitive (10 vs 40 instances on the same scan); single-yaw hits
    that no other yaw reproduces are exactly the unstable ones.
    --yaw_k 1 reproduces v08 behaviour (ablation baseline).
  * Fused cluster mask = majority vote over the yaws that saw the
    object (≥ ceil(n_yaws/2)), which also repairs seam-split masks.
  * NOTE: runtime scales ≈ ×K in the detection stage.

v08 (2026-07-09) — changes vs OpenScan2BIM_Step01_Segmentation07.py:
  * FIX R/B channel swap in the saved panorama: create_panorama() builds
    pano_image in RGB order (colorRed/Green/Blue stacked), but v07 and all
    earlier versions passed it to cv2.imwrite() — which expects BGR — so
    every <scan>_pano.jpg on disk had red and blue swapped (wooden doors
    displayed blue; user-confirmed on outputs08, file channels B=89 vs
    R=61 on a GT-Wood door).  Because GroundingDINO/SAM and Step-02 CLIP
    material classification re-read that jpg and apply BGR2RGB, they all
    saw swapped colors.  CLIP scene classification used the in-memory RGB
    array and was unaffected.  v08 converts RGB→BGR at the single write
    boundary; every downstream consumer then becomes correct unchanged.
    CAVEAT: pano jpgs produced by v07 and earlier (outputs02–10) remain
    swapped on disk — when reading those legacy files with cv2.imread, the
    array is already RGB, so skip the usual BGR2RGB conversion.  Check the
    run manifest's "pano_color_order" field to tell the eras apart.

v07 (2026-07-08) — changes vs OpenScan2BIM_Step01_Segmentation06.py:
  * Reproducibility: fixed RNG seeds (Python/NumPy/Torch/CUDA) + cuDNN
    deterministic mode, so the EXP-A stability study can separate true
    run-to-run noise from configuration sensitivity.
  * Run manifest: every run writes <scan>_run_manifest.json capturing CLI
    args, module constants, the prompt schedule, package/GPU versions and
    result counts — results stay traceable across parameter sweeps.

Manuscript reference:
    Sec. 4.1  Spherical Panoramic Projection
    Sec. 4.2  Multi-Pass Open-Vocabulary Instance Segmentation

Pipeline (this script):
    E57 point cloud
        ↓  Sec. 4.1
    Spherical panoramic projection  →  RGB panorama + depth map + pixel-to-point LUT
        ↓  Sec. 4.2.1
    CLIP-based scene type classification  →  scene type label
        ↓  Sec. 4.2.2  /  Algorithm 1
    Multi-scale Grounded-SAM segmentation  →  merged instance mask (uint16 TIFF)
        ↓  Sec. 4.2.3
    3D back-projection  →  per-point instance IDs
        ↓
    Outputs: labeled PLY  +  metadata CSV  (consumed by Step 02)

Multi-scale strategy (NEW — Sec. 4.2.2):
    Three image scales are processed per pass:
      Scale 1 (1.0×): original resolution — captures large structures
      Scale 2 (1.5×): upscaled — captures medium elements (doors, windows)
      Scale 3 (2.0×): upscaled — captures small elements (pipes, vents)
    Masks from all scales are rescaled back to original resolution, then
    fused via IoU-based instance matching + weighted pixel vote.

Outputs written to --output_dir:
    <scan>_pano.jpg                          RGB spherical panorama
    <scan>_pano_depth_vis.png                depth-map visualisation (jet colormap)
    <scan>_pano_depth.tif                    float32 depth map (optional, --save_depth_tif)
    <scan>_pass<k>_residual.jpg              residual image fed into pass k
    <scan>_pass<k>_scale<s>_all.jpg          all instances for scale s of pass k (stacked)
    <scan>_pass<k>_fused_all.jpg             all fused instances for pass k (stacked)
    <scan>_pano_mask.tif                     uint16 merged instance-label image
    <scan>_pano_labeled.ply                  XYZ-RGB + instance_id + semantic_class
    <scan>_pano_metadata.csv                 per-instance metadata (input for Step 02)
    <scan>_pano_segmented.jpg                final segmentation overview on full panorama

Usage:
    python OpenScan2BIM_Step01_Segmentation.py \\
        --input_e57   ./assets/scan_001.e57 \\
        --output_dir  ./outputs/scan_001 \\
        --image_height 4096

Dependencies:
    pye57, open_clip, segment_anything (SAM), GroundingDINO,
    tifffile, plyfile, opencv-python, torch, numpy, matplotlib
"""

# ---------------------------------------------------------------------------
# Standard library
# ---------------------------------------------------------------------------
import os
import sys
# ── Offline mode: must be set before ANY HuggingFace library is imported ──
os.environ["HF_HUB_OFFLINE"]      = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"]  = "1"
# --------------------------------------------------------------------------
import csv
import json
import time
import random
import argparse
from math import pi
from concurrent.futures import ThreadPoolExecutor, as_completed

# ---------------------------------------------------------------------------
# Third-party
# ---------------------------------------------------------------------------
import cv2
import numpy as np
import torch
import tifffile
import open_clip
import pye57
import matplotlib.pyplot as plt
from PIL import Image
from plyfile import PlyElement, PlyData

# ---------------------------------------------------------------------------
# Grounded-SAM  (GroundingDINO + Segment Anything)
# ---------------------------------------------------------------------------
# The upstream Grounded-SAM packages live in the workspace root (pipeline/..).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from segment_anything.segment_anything import sam_model_registry, SamPredictor
from GroundingDINO.groundingdino.models import build_model
from GroundingDINO.groundingdino.util.utils import (
    get_phrases_from_posmap,
    clean_state_dict,
)
from GroundingDINO.groundingdino.datasets import transforms as T
from GroundingDINO.groundingdino.util.slconfig import SLConfig


# ===========================================================================
# Hyperparameters  —  manuscript Sec. 4.1 / 4.2
# (All tentative values; update after ablation experiments in Sec. 5.3)
# ===========================================================================

DEFAULT_IMAGE_HEIGHT = 4096   # panorama height H; width W = 2H
                               # Manuscript: 8192×4096. Use 2048 for fast tests.

CONF_THRESHOLD   = 0.29       # τ — detection confidence threshold
PHRASE_THRESHOLD = 0.25       # token-level threshold for phrase extraction
MIN_MASK_AREA    = 500        # A_min pixels — discard smaller candidates
PASS_ID_OFFSET   = 10_000     # Δ_k = k × PASS_ID_OFFSET (global ID uniqueness)

# ── Multi-scale inference parameters (NEW) ───────────────────────────────────
# Three scales are applied per pass. Masks from each scale are rescaled back
# to original resolution and fused before writing to merged_mask.
MULTISCALE_FACTORS = [1.0]   # image scale multipliers (high → low resolution)
                                         # Rationale: GroundingDINO internally resizes
                                         # input to 800px, so upscaling beyond 1.0× gives
                                         # diminishing returns. Downscaling instead changes
                                         # how much of the scene fits within DINO's
                                         # receptive field:
                                         # 1.0× (4096px) → full detail, medium context
                                         # 0.7× (2867px) → slightly wider context
                                         # 0.5× (2048px) → widest context, global structures

# Per-scale weights for the pixel-level weighted vote (must sum to 1.0).
# Full-resolution scale gets highest weight since it has the best spatial detail.
MULTISCALE_WEIGHTS = [0.90, 0.30, 0.20]

# Pixel vote threshold: a pixel is accepted into the fused foreground if its
# weighted vote total meets this value.
# 0.30 → any single scale is enough (union-like, maximum recall)
# 0.65 → at least two scales must agree (intersection-like, higher precision)
VOTE_THRESHOLD = 0.30

# IoU threshold for matching instances across scales before union-merging.
# Kept low because the same object seen at different scales often has
# moderate IoU due to resolution and boundary differences.
CROSS_SCALE_IOU_THRESHOLD = 0.15

# Overlap threshold: a fused mask is discarded if this fraction of its pixels
# already belongs to a previously accepted instance in merged_mask.
# Updated immediately after each accepted instance within the same pass,
# so intra-pass collisions are also caught correctly.
OVERLAP_THRESHOLD = 0.60

# Minimum residual unlabeled pixels to justify another pass
MIN_RESIDUAL_PX = 100

# ── Yaw-ensemble parameters (v09, M4) ────────────────────────────────────────
# The residual panorama is yaw-rotated K times (equal steps of W/K pixels,
# a pure np.roll — no interpolation) and the detector runs once per yaw.
YAW_ENSEMBLE_K  = 4     # ensemble size; 1 = v08 behaviour (no ensemble)
PERSISTENCE_MIN = 0.5   # keep clusters detected in ≥ this fraction of yaws
                        # (K=4 → at least 2 yaws must agree)
YAW_MATCH_IOU   = 0.50  # pixel-IoU threshold for matching detections of the
                        # same object across yaws (after unrolling)


# ===========================================================================
# Scene-type prompt table  —  manuscript Table 1 / Sec. 4.2.1
# ===========================================================================
# Prompts use Grounding DINO period-separated format.
# The list is ordered: prompt[0] → pass 1, prompt[1] → pass 2, …
# The multi-pass loop advances through this list until:
#   (a) max fused confidence < τ  (no confident detections remain), OR
#   (b) fewer than MIN_RESIDUAL_PX unlabeled pixels remain, OR
#   (c) all prompts in the list have been exhausted.

SCENE_TYPE_PROMPTS = {
    # ── Indoor  (institutional building interiors, default) ───────────────
    "Indoor": [
        "ceiling . wall . floor . door . window .",        # pass 1 — dominant elements
        "wall .",  # pass 2 — structural details
        "",      # pass 3 — MEP elements
    ],
    # ── Outdoor  (building facades, entrances) ────────────────────────────
    "Outdoor": [
        "wall . window . door . roof . column .",          # pass 1
        "balcony . canopy . overhang . signage .",         # pass 2
        "vegetation . pavement . ground . curb . fence .", # pass 3
    ],
    # ── Foundation  (basement, underground mechanical rooms) ──────────────
    "Foundation": [
        "wall . floor . column . beam . slab .",                  # pass 1
        "pipe . duct . conduit . footing . structural support .", # pass 2
    ],
    # ── Ground  (open ground-level courtyards, plazas) ───────────────────
    "Ground": [
        "ground . wall . pavement . curb . drain .",  # pass 1
        "vegetation . equipment . signage . ramp .",  # pass 2
    ],
}

# CLIP text descriptions for scene type classification (Sec. 4.2.1)
SCENE_TYPE_DESCRIPTIONS = {
    "Indoor":     "an indoor room interior of a building",
    "Outdoor":    "an outdoor building exterior or facade",
    "Foundation": "a building foundation, basement, or underground space",
    "Ground":     "an open ground-level area or courtyard",
}


# ===========================================================================
# Utilities
# ===========================================================================

def timestamp() -> str:
    """Return a formatted timestamp string for log messages."""
    return time.strftime("%Y-%m-%d %H:%M:%S")


def log(msg: str) -> None:
    """Print a timestamped log message to stdout."""
    print(f"[{timestamp()}] {msg}")


# ===========================================================================
# Sec. 4.1 — Point cloud loading
# ===========================================================================

def load_point_cloud(e57_path: str) -> tuple:
    """
    Load the first scan from an E57 file.

    Returns
    -------
    data : dict
        Raw scan fields (cartesianX/Y/Z, colorRed/Green/Blue, …).
    sensor_position : np.ndarray, shape (3,)
        Scanner origin in world coordinates (E57 header translation).
    """
    e57             = pye57.E57(e57_path)
    data            = e57.read_scan(0, intensity=True, colors=True,
                                    ignore_missing_fields=True)
    sensor_position = np.array(e57.get_header(0).translation)
    log(f"Loaded E57: {e57_path}  |  scanner origin: {sensor_position}")
    return data, sensor_position


# ===========================================================================
# Sec. 4.1 — Spherical projection  (Eq. 1–2)
# ===========================================================================

def _spherical_to_pixel(
    rel_xyz:   np.ndarray,   # shape (3, N)
    distances: np.ndarray,   # shape (N,)
    height:    int,
    width:     int,
) -> tuple:
    """
    Map 3-D relative coordinates to spherical panorama pixel coordinates.

    Eq. (1):   θ = arctan2(y, x)           azimuth  ∈ [−π, π)
               φ = arcsin(z / r)            elevation ∈ [−π/2, π/2]

    Eq. (2):   u = floor( (θ + π) / 2π · W )
               v = floor( (π/2 − φ) / π · H )

    Returns
    -------
    px, py : np.ndarray dtype int32  —  pixel column (u) and row (v).
    """
    x, y, z = rel_xyz
    theta = np.arctan2(y, x)
    phi   = np.arcsin(np.clip(z / distances, -1.0, 1.0))

    px = ((theta + pi) / (2.0 * pi) * width ).astype(np.int32)
    py = ((pi / 2.0 - phi)   / pi   * height).astype(np.int32)

    px = np.clip(px, 0, width  - 1)
    py = np.clip(py, 0, height - 1)
    return px, py


def create_panorama(
    data:           dict,
    sensor_pos:     np.ndarray,
    height:         int,
    output_dir:     str,
    base_name:      str,
    save_depth_tif: bool = False,
) -> tuple:
    """
    Project the point cloud onto a spherical panoramic image.

    Depth-buffer: when multiple points project to the same pixel, only the
    closest return (minimum distance) is retained (Sec. 4.1).

    Returns
    -------
    pano_image      : (H, W, 3) uint8   RGB panorama
    world_points    : (N, 3) float64    world-frame 3-D coordinates
    point_colors    : (N, 3) uint8      per-point RGB
    depth_map       : (H, W) float32    metres; np.inf where empty
    pixel_to_point  : (H, W) int64      pixel → point index; −1 if empty
    """
    width = height * 2

    xyz       = np.stack([data["cartesianX"], data["cartesianY"], data["cartesianZ"]])
    rel_xyz   = xyz - sensor_pos[:, None]
    distances = np.linalg.norm(rel_xyz, axis=0)

    valid     = distances > 0
    rel_xyz   = rel_xyz[:, valid]
    distances = distances[valid]
    colors    = np.stack([
        data["colorRed"][valid],
        data["colorGreen"][valid],
        data["colorBlue"][valid],
    ], axis=-1).astype(np.uint8)
    world_pts = rel_xyz.T + sensor_pos

    px, py = _spherical_to_pixel(rel_xyz, distances, height, width)

    pano_image     = np.zeros((height, width, 3), dtype=np.uint8)
    depth_map      = np.full( (height, width),    np.inf, dtype=np.float32)
    pixel_to_point = np.full( (height, width),    -1,     dtype=np.int64)

    n_pts      = len(px)
    n_threads  = max(1, os.cpu_count() or 1)
    chunk_size = max(1, n_pts // n_threads)

    def _rasterize(start: int, end: int) -> None:
        for i in range(start, end):
            u, v = px[i], py[i]
            if distances[i] < depth_map[v, u]:
                depth_map[v, u]      = distances[i]
                pano_image[v, u]     = colors[i]
                pixel_to_point[v, u] = i

    with ThreadPoolExecutor(max_workers=n_threads) as pool:
        futs = [
            pool.submit(_rasterize, s, min(s + chunk_size, n_pts))
            for s in range(0, n_pts, chunk_size)
        ]
        for f in as_completed(futs):
            f.result()

    vis_path = os.path.join(output_dir, f"{base_name}_pano_depth_vis.png")
    fig, ax  = plt.subplots(figsize=(14, 7))
    im       = ax.imshow(depth_map.copy(), cmap="jet")
    plt.colorbar(im, ax=ax, label="Distance (m)")
    ax.set_title("Depth map — scanner distance (m)")
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(vis_path, dpi=150)
    plt.close()
    log(f"Saved depth visualisation → {vis_path}")

    if save_depth_tif:
        tif_path = os.path.join(output_dir, f"{base_name}_pano_depth.tif")
        tifffile.imwrite(tif_path, depth_map.copy().astype(np.float32))
        log(f"Saved depth TIFF → {tif_path}")

    return pano_image, world_pts, colors, depth_map, pixel_to_point


# ===========================================================================
# Sec. 4.2.1 — CLIP scene type classification
# ===========================================================================

def classify_scene_type(
    pano_image:      np.ndarray,
    clip_model,
    clip_preprocess,
    clip_tokenizer,
    device:          str,
) -> tuple:
    """
    Zero-shot scene type classification using CLIP (Sec. 4.2.1).

    Returns
    -------
    scene_type  : str    — one of {"Indoor", "Outdoor", "Foundation", "Ground"}
    confidence  : float  — softmax score ∈ (0, 1)
    """
    image_input  = clip_preprocess(Image.fromarray(pano_image)).unsqueeze(0).to(device)
    descriptions = list(SCENE_TYPE_DESCRIPTIONS.values())
    labels       = list(SCENE_TYPE_DESCRIPTIONS.keys())
    text_tokens  = clip_tokenizer(descriptions).to(device)

    with torch.no_grad():
        img_feats  = clip_model.encode_image(image_input)
        txt_feats  = clip_model.encode_text(text_tokens)
        img_feats /= img_feats.norm(dim=-1, keepdim=True)
        txt_feats /= txt_feats.norm(dim=-1, keepdim=True)
        sim        = (100.0 * img_feats @ txt_feats.T).softmax(dim=-1)
        best_idx   = int(sim.argmax().item())

    scene_type = labels[best_idx]
    confidence = float(sim[0, best_idx].item())
    log(f"Scene type: {scene_type}  (confidence {confidence:.2%})")
    return scene_type, confidence


# ===========================================================================
# Sec. 4.2 — Grounded-SAM model loading
# ===========================================================================

def load_grounded_sam(
    device:          str,
    dino_config:     str,
    dino_checkpoint: str,
    sam_checkpoint:  str,
    sam_version:     str = "vit_h",
) -> tuple:
    """Load GroundingDINO and SAM models onto `device`."""
    args = SLConfig.fromfile(dino_config)
    args.device = device
    dino_model  = build_model(args)
    ckpt        = torch.load(dino_checkpoint, map_location=device)
    dino_model.load_state_dict(clean_state_dict(ckpt["model"]), strict=False)
    dino_model.eval().to(device)
    log(f"GroundingDINO loaded  ←  {dino_checkpoint}")

    sam           = sam_model_registry[sam_version](checkpoint=sam_checkpoint)
    sam.to(device)
    sam_predictor = SamPredictor(sam)
    log(f"SAM ({sam_version}) loaded  ←  {sam_checkpoint}")

    return dino_model, sam_predictor


# ===========================================================================
# Sec. 4.2.2 — Single-scale Grounded-SAM inference  (internal helper)
# ===========================================================================

def _run_single_scale(
    image_bgr:        np.ndarray,  # BGR image at the current scale
    orig_H:           int,         # original panorama height (for mask rescaling)
    orig_W:           int,         # original panorama width
    dino_model,
    sam_predictor:    SamPredictor,
    device:           str,
    text_prompt:      str,
    conf_threshold:   float,
    phrase_threshold: float,
    min_area:         int,
    scale_label:      str,         # e.g. "1.0×" — used only for log messages
) -> tuple:
    """
    Run one GroundingDINO + SAM forward pass on `image_bgr`.

    All returned masks are rescaled to (orig_H, orig_W) so outputs from
    different scales can be directly compared and fused.

    Returns
    -------
    masks_list   : list of (orig_H, orig_W) bool arrays
    phrases_list : list of str labels
    scores_list  : list of float confidences
    """
    scale_H, scale_W = image_bgr.shape[:2]

    # GroundingDINO transform
    image_pil     = Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
    transform     = T.Compose([
        T.RandomResize([800], max_size=1333),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    img_tensor, _ = transform(image_pil, None)

    prompt = text_prompt.lower().strip()
    if not prompt.endswith("."):
        prompt += "."

    # GroundingDINO forward pass
    with torch.no_grad():
        outputs = dino_model(img_tensor[None].to(device), captions=[prompt])

    logits = outputs["pred_logits"].sigmoid()[0].cpu()
    boxes  = outputs["pred_boxes"][0].cpu()

    scores   = logits.max(dim=1)[0]
    keep     = scores > conf_threshold
    logits_f = logits[keep]
    boxes_f  = boxes[keep]
    scores_f = scores[keep]

    if boxes_f.shape[0] == 0:
        return [], [], []

    tokenizer = dino_model.tokenizer
    tokenized = tokenizer(prompt)
    phrases   = [
        get_phrases_from_posmap(l > phrase_threshold, tokenized, tokenizer)
        for l in logits_f
    ]

    # cx cy w h (normalised) → xyxy (scaled-image pixels)
    boxes_xyxy        = boxes_f.clone() * torch.tensor(
                            [scale_W, scale_H, scale_W, scale_H])
    boxes_xyxy[:, :2] -= boxes_xyxy[:, 2:] / 2
    boxes_xyxy[:, 2:] += boxes_xyxy[:, :2]

    # SAM mask prediction
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    sam_predictor.set_image(image_rgb)
    xformed = sam_predictor.transform.apply_boxes_torch(
        boxes_xyxy, image_rgb.shape[:2]
    ).to(device)
    masks_tensor, _, _ = sam_predictor.predict_torch(
        point_coords=None, point_labels=None,
        boxes=xformed, multimask_output=False,
    )   # (K, 1, scale_H, scale_W)

    # Rescale masks → original resolution, apply area filter
    masks_list, phrases_list, scores_list = [], [], []
    for idx, mask_t in enumerate(masks_tensor):
        mask_u8 = (mask_t.cpu().numpy()[0] > 0).astype(np.uint8) * 255
        if scale_H != orig_H or scale_W != orig_W:
            mask_u8 = cv2.resize(mask_u8, (orig_W, orig_H),
                                 interpolation=cv2.INTER_NEAREST)
        mask_bool = mask_u8 > 127

        area = int(mask_bool.sum())
        if area < min_area:
            continue

        label = phrases[idx]
        conf  = float(scores_f[idx].item())
        masks_list.append(mask_bool)
        phrases_list.append(label)
        scores_list.append(conf)

        log(f"      [{scale_label}] '{label}' | area={area:,} px | conf={conf:.3f}")

    return masks_list, phrases_list, scores_list


# ===========================================================================
# Sec. 4.2.2 — Multi-scale mask fusion  (NEW)
# ===========================================================================

def _iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    """Pixel-level IoU between two boolean masks."""
    inter = int((mask_a & mask_b).sum())
    union = int((mask_a | mask_b).sum())
    return inter / union if union > 0 else 0.0


def _fuse_multiscale_masks(
    all_masks:   list,   # [scale_idx] → list of (H, W) bool
    all_phrases: list,   # [scale_idx] → list of str
    all_scores:  list,   # [scale_idx] → list of float
    orig_H:      int,
    orig_W:      int,
) -> tuple:
    """
    Fuse instance masks from multiple image scales.

    Step 1 — Weighted pixel vote:
        vote_map[v,u] = Σ_s  w_s · 1[pixel (v,u) is foreground in scale s]
        Accept pixel if vote_map[v,u] ≥ VOTE_THRESHOLD.
        This removes scale-specific noise while retaining pixels seen by
        at least one (or more, depending on threshold) scale.

    Step 2 — Cross-scale IoU union:
        Each candidate mask (from any scale) is clipped to the voted foreground.
        Candidates are sorted by confidence (highest first).
        If a candidate overlaps an already-accepted instance by IoU ≥
        CROSS_SCALE_IOU_THRESHOLD, they are merged via pixel union (same object,
        different scale view). Otherwise it is added as a new instance.
        The merged instance keeps the highest-confidence label.

    Returns
    -------
    fused_masks   : list of (H, W) bool
    fused_phrases : list of str
    fused_scores  : list of float
    """
    n_scales = len(all_masks)
    if not any(all_masks):
        return [], [], []

    # ── Step 1: Weighted pixel vote ───────────────────────────────────────
    vote_map = np.zeros((orig_H, orig_W), dtype=np.float32)
    for s_idx, masks_s in enumerate(all_masks):
        if not masks_s:
            continue
        w = (MULTISCALE_WEIGHTS[s_idx]
             if s_idx < len(MULTISCALE_WEIGHTS) else 1.0 / n_scales)
        fg_s = np.zeros((orig_H, orig_W), dtype=bool)
        for m in masks_s:
            fg_s |= m
        vote_map += fg_s.astype(np.float32) * w

    voted_fg = vote_map >= VOTE_THRESHOLD   # (H, W) bool

    # ── Step 2: Collect candidates, clip to voted foreground ─────────────
    candidates = []
    for s_idx in range(n_scales):
        for m, ph, sc in zip(all_masks[s_idx], all_phrases[s_idx],
                              all_scores[s_idx]):
            m_clipped = m & voted_fg
            if m_clipped.sum() < 1:
                continue
            candidates.append((m_clipped, ph, sc))

    # Sort by confidence descending so best detections anchor clusters
    candidates.sort(key=lambda x: x[2], reverse=True)

    # ── Step 3: IoU-based cross-scale union ───────────────────────────────
    fused_masks:   list = []
    fused_phrases: list = []
    fused_scores:  list = []

    for cand_mask, cand_phrase, cand_score in candidates:
        matched = False
        for i, exist_mask in enumerate(fused_masks):
            if _iou(cand_mask, exist_mask) >= CROSS_SCALE_IOU_THRESHOLD:
                # Same physical object seen at a different scale → union
                fused_masks[i] = exist_mask | cand_mask
                if cand_score > fused_scores[i]:
                    fused_phrases[i] = cand_phrase
                    fused_scores[i]  = cand_score
                matched = True
                break
        if not matched:
            fused_masks.append(cand_mask.copy())
            fused_phrases.append(cand_phrase)
            fused_scores.append(cand_score)

    raw_count = sum(len(m) for m in all_masks)
    log(f"    Fusion: {raw_count} raw → {len(fused_masks)} fused instances")
    return fused_masks, fused_phrases, fused_scores


# ===========================================================================
# Yaw-ensemble aggregation  (v09, M4)
# ===========================================================================

def _aggregate_yaw_ensemble(
    detections:      list,   # (mask HxW bool, phrase, score, yaw_idx)
    yaw_k:           int,
    persistence_min: float,
    min_area:        int,
) -> tuple:
    """
    Cluster unrolled detections across yaws and score persistence.

    Greedy clustering anchored on the highest-confidence detections
    (deterministic: score-sorted): a detection joins an existing cluster
    when its pixel IoU with the cluster's union mask ≥ YAW_MATCH_IOU.
    Per cluster:
      persistence = (# distinct yaws contributing) / yaw_k   — the TTA/
      deep-ensembles agreement proxy for detector epistemic uncertainty;
      fused mask  = pixels seen by ≥ ceil(n_yaws/2) of the contributing
      yaws (per-yaw union first, so fragments within one yaw count once);
      label/score = from the highest-confidence member.
    Clusters with persistence < persistence_min are dropped (single-yaw
    hits are exactly the configuration-sensitive detections EXP-A
    documented) — unless yaw_k == 1, which reproduces v08.

    Returns (masks, phrases, scores, persistences), score-sorted.
    """
    if not detections:
        return [], [], [], []
    detections = sorted(detections, key=lambda d: d[2], reverse=True)
    clusters: list = []          # each: {"union", "members": [(mask, ph, sc, yaw)]}
    for mask, phrase, score, yaw in detections:
        matched = False
        for cl in clusters:
            if _iou(mask, cl["union"]) >= YAW_MATCH_IOU:
                cl["union"] = cl["union"] | mask
                cl["members"].append((mask, phrase, score, yaw))
                matched = True
                break
        if not matched:
            clusters.append({"union": mask.copy(),
                             "members": [(mask, phrase, score, yaw)]})

    out_masks, out_phrases, out_scores, out_pers = [], [], [], []
    n_dropped = 0
    for cl in clusters:
        yaws = sorted({m[3] for m in cl["members"]})
        pers = len(yaws) / float(yaw_k)
        if pers < persistence_min:
            n_dropped += 1
            continue
        # per-yaw union, then majority vote over contributing yaws
        vote = np.zeros(cl["union"].shape, dtype=np.int16)
        for y in yaws:
            fg = np.zeros(cl["union"].shape, dtype=bool)
            for m, _, _, my in cl["members"]:
                if my == y:
                    fg |= m
            vote += fg.astype(np.int16)
        need = max(1, int(np.ceil(len(yaws) / 2.0)))
        fused = vote >= need
        if int(fused.sum()) < min_area:
            n_dropped += 1
            continue
        best = max(cl["members"], key=lambda m: m[2])
        out_masks.append(fused)
        out_phrases.append(best[1])
        out_scores.append(best[2])
        out_pers.append(pers)

    order = sorted(range(len(out_masks)), key=lambda i: out_scores[i],
                   reverse=True)
    log(f"    Yaw ensemble: {len(detections)} detections → "
        f"{len(clusters)} clusters → {len(order)} kept "
        f"(persistence ≥ {persistence_min}), {n_dropped} dropped")
    return ([out_masks[i] for i in order], [out_phrases[i] for i in order],
            [out_scores[i] for i in order], [out_pers[i] for i in order])


# ===========================================================================
# Sec. 4.2.2 — Multi-pass segmentation loop  (Algorithm 1, extended)
# ===========================================================================

def segment_multipass(
    pano_image_path: str,
    dino_model,
    sam_predictor:   SamPredictor,
    device:          str,
    scene_type:      str,
    output_dir:      str,
    base_name:       str,
    conf_threshold:  float = CONF_THRESHOLD,
    min_area:        int   = MIN_MASK_AREA,
    phrase_threshold:float = PHRASE_THRESHOLD,
    yaw_k:           int   = YAW_ENSEMBLE_K,
    persistence_min: float = PERSISTENCE_MIN,
) -> tuple:
    """
    Iterative multi-pass segmentation with multi-scale inference and
    (v09) a yaw ensemble: each pass runs the detector on yaw_k rolled
    copies of the residual panorama, unrolls the masks, clusters them
    across yaws and keeps clusters with persistence ≥ persistence_min.

    At each pass k:
      1. Build residual image I^(k): zero already-labeled pixels (Eq. 3).
      2. For each scale factor in MULTISCALE_FACTORS:
           a. Resize I^(k) to that scale.
           b. Run GroundingDINO + SAM → raw masks at scale resolution.
           c. Rescale masks back to original (orig_H, orig_W).
      3. Fuse masks across scales: weighted pixel vote + IoU union.
      4. Filter fused masks using OVERLAP_THRESHOLD against merged_mask.
         — merged_mask is updated after EACH accepted instance so that
           intra-pass collisions are correctly detected.
      5. Accept survivors; assign global IDs = local_counter + k×PASS_ID_OFFSET.
      6. Terminate if max fused confidence < τ.

    Returns
    -------
    merged_mask          : (H, W) uint16  global instance labels (0 = background)
    instance_to_label    : {global_id → str}
    instance_areas       : {global_id → int pixels}
    instance_confidences : {global_id → float}
    instance_persistence : {global_id → float}  (v09)
    """
    prompt_list    = SCENE_TYPE_PROMPTS.get(scene_type, SCENE_TYPE_PROMPTS["Indoor"])
    base_img       = cv2.imread(pano_image_path)
    orig_H, orig_W = base_img.shape[:2]

    merged_mask          = np.zeros((orig_H, orig_W), dtype=np.uint16)
    instance_to_label    = {}
    instance_areas       = {}
    instance_confidences = {}
    instance_persistence = {}
    yaw_shifts = [(orig_W * y) // max(yaw_k, 1) for y in range(max(yaw_k, 1))]

    for pass_idx, prompt in enumerate(prompt_list, start=1):

        # ── Residual check ────────────────────────────────────────────────
        n_unlabeled = int((merged_mask == 0).sum())
        if n_unlabeled < MIN_RESIDUAL_PX:
            log(f"Pass {pass_idx}: {n_unlabeled} unlabeled pixels < "
                f"MIN_RESIDUAL_PX={MIN_RESIDUAL_PX} — stopping.")
            break

        log(f"{'─'*60}")
        log(f"Pass {pass_idx}/{len(prompt_list)}  |  prompt: '{prompt}'  |  "
            f"unlabeled: {n_unlabeled:,} px")

        # ── Build residual image I^(k)  (Eq. 3) ──────────────────────────
        residual_mask_u8 = (merged_mask == 0).astype(np.uint8) * 255
        residual_img     = cv2.bitwise_and(base_img, base_img,
                                           mask=residual_mask_u8)
        residual_path = os.path.join(
            output_dir, f"{base_name}_pass{pass_idx}_residual.jpg"
        )
        cv2.imwrite(residual_path, residual_img)

        # ── Yaw ensemble × multi-scale GroundingDINO + SAM inference ──────
        # (v09) The residual panorama is rolled by each yaw shift; masks
        # come back unrolled so all yaws share the original pixel frame.
        yaw_detections: list = []          # (mask, phrase, score, yaw_idx)
        for y_idx, shift in enumerate(yaw_shifts):
            if len(yaw_shifts) > 1:
                log(f"  Yaw {y_idx+1}/{len(yaw_shifts)}  "
                    f"(roll {shift} px = {360 * shift // orig_W}°) …")
            yaw_img = residual_img if shift == 0 else \
                np.roll(residual_img, -shift, axis=1)

            all_masks:   list = []
            all_phrases: list = []
            all_scores:  list = []
            for s_idx, scale in enumerate(MULTISCALE_FACTORS):
                label = f"{scale:.1f}×"
                log(f"  Scale {s_idx+1}/{len(MULTISCALE_FACTORS)}  ({label}) …")

                if scale == 1.0:
                    scaled_img = yaw_img
                else:
                    new_W      = int(orig_W * scale)
                    new_H      = int(orig_H * scale)
                    scaled_img = cv2.resize(yaw_img, (new_W, new_H),
                                            interpolation=cv2.INTER_LINEAR)

                masks_s, phrases_s, scores_s = _run_single_scale(
                    image_bgr        = scaled_img,
                    orig_H           = orig_H,
                    orig_W           = orig_W,
                    dino_model       = dino_model,
                    sam_predictor    = sam_predictor,
                    device           = device,
                    text_prompt      = prompt,
                    conf_threshold   = conf_threshold,
                    phrase_threshold = phrase_threshold,
                    min_area         = min_area,
                    scale_label      = label,
                )

                # Save per-scale visualisation for the un-rolled yaw only
                # (other yaws would quadruple the QC images)
                if masks_s and shift == 0:
                    _save_scale_vis(
                        output_dir = output_dir,
                        base_name  = base_name,
                        pass_idx   = pass_idx,
                        scale_idx  = s_idx + 1,
                        masks      = masks_s,
                        phrases    = phrases_s,
                        scores     = scores_s,
                        image_bgr  = residual_img,
                    )

                all_masks.append(masks_s)
                all_phrases.append(phrases_s)
                all_scores.append(scores_s)

            # Fuse masks across scales WITHIN this yaw (v08 semantics) …
            f_masks, f_phrases, f_scores = _fuse_multiscale_masks(
                all_masks   = all_masks,
                all_phrases = all_phrases,
                all_scores  = all_scores,
                orig_H      = orig_H,
                orig_W      = orig_W,
            )
            # … then unroll back to the original pixel frame
            for m, ph, sc in zip(f_masks, f_phrases, f_scores):
                m_unrolled = m if shift == 0 else np.roll(m, shift, axis=1)
                yaw_detections.append((m_unrolled, ph, sc, y_idx))

        # ── Aggregate across yaws: persistence-gated clusters ─────────────
        if len(yaw_shifts) > 1:
            fused_masks, fused_phrases, fused_scores, fused_pers = \
                _aggregate_yaw_ensemble(yaw_detections, len(yaw_shifts),
                                        persistence_min, min_area)
        else:                                   # yaw_k=1 → v08 behaviour
            fused_masks   = [d[0] for d in yaw_detections]
            fused_phrases = [d[1] for d in yaw_detections]
            fused_scores  = [d[2] for d in yaw_detections]
            fused_pers    = [1.0] * len(yaw_detections)

        # ── Termination check ─────────────────────────────────────────────
        max_conf = max(fused_scores, default=0.0)
        if max_conf < conf_threshold:
            log(f"Pass {pass_idx}: max fused conf={max_conf:.3f} < "
                f"τ={conf_threshold} — stopping.")
            break

        # ── Overlap filtering + merge into global mask ────────────────────
        # Sort by confidence (highest first) so the best-quality detection
        # claims pixels before weaker duplicates.
        order         = sorted(range(len(fused_masks)),
                               key=lambda i: fused_scores[i], reverse=True)
        id_offset     = pass_idx * PASS_ID_OFFSET
        local_counter = 0
        n_accepted    = 0

        for fi in order:
            fused_mask   = fused_masks[fi]
            fused_phrase = fused_phrases[fi]
            fused_score  = fused_scores[fi]
            fused_p      = fused_pers[fi]

            new_px = int(fused_mask.sum())
            if new_px == 0:
                continue

            # Overlap ratio against ALL previously written pixels.
            # Because we update merged_mask immediately after each accepted
            # instance, this check is valid for both inter-pass AND intra-pass
            # collisions — earlier instances in the same pass are already in
            # merged_mask when later ones are evaluated.
            overlap_px    = int((fused_mask & (merged_mask > 0)).sum())
            overlap_ratio = overlap_px / new_px

            label_str = (list(fused_phrase)[0]
                         if isinstance(fused_phrase, set) else fused_phrase)

            if overlap_ratio > OVERLAP_THRESHOLD:
                log(f"    Discarded '{label_str}' — "
                    f"overlap {overlap_ratio:.1%} > {OVERLAP_THRESHOLD:.0%}")
                continue

            # Assign global ID and write ONLY the free (unlabeled) pixels.
            # Writing only free pixels prevents overwriting higher-ranked
            # instances already written in this loop iteration.
            local_counter += 1
            global_id      = local_counter + id_offset
            free_px        = fused_mask & (merged_mask == 0)
            merged_mask[free_px] = global_id

            instance_to_label[global_id]    = label_str
            instance_areas[global_id]       = int(free_px.sum())
            instance_confidences[global_id] = fused_score
            instance_persistence[global_id] = fused_p
            n_accepted += 1

            log(f"    Accepted  ID={global_id:6d} | '{label_str}' | "
                f"area={int(free_px.sum()):,} px | conf={fused_score:.3f} | "
                f"persistence={fused_p:.2f} | overlap={overlap_ratio:.1%}")

        # Save fused-pass visualisation (one file per instance + combined)
        if fused_masks:
            _save_fused_vis(
                output_dir  = output_dir,
                base_name   = base_name,
                pass_idx    = pass_idx,
                fused_masks = fused_masks,
                phrases     = fused_phrases,
                scores      = fused_scores,
                image_bgr   = residual_img,
            )

        log(f"Pass {pass_idx}: {n_accepted}/{len(fused_masks)} instances "
            f"accepted  (max_conf={max_conf:.3f})")

    # ── Save merged instance mask ─────────────────────────────────────────
    mask_path = os.path.join(output_dir, f"{base_name}_pano_mask.tif")
    tifffile.imwrite(mask_path, merged_mask.astype(np.uint16))
    log(f"Saved merged instance mask → {mask_path}")

    return (merged_mask, instance_to_label, instance_areas,
            instance_confidences, instance_persistence)


# ===========================================================================
# Sec. 4.2.3 — Back-projection: 2D mask → 3D point labels
# ===========================================================================

def back_project_labels(
    merged_mask:  np.ndarray,
    world_points: np.ndarray,
    sensor_pos:   np.ndarray,
    height:       int,
) -> np.ndarray:
    """
    Assign 2D instance labels to 3D points (Sec. 4.2.3).

    Returns
    -------
    instance_ids : (N,) uint16 — 0 = background / unlabeled
    """
    width   = height * 2
    rel_xyz = world_points - sensor_pos
    dists   = np.linalg.norm(rel_xyz, axis=1)
    px, py  = _spherical_to_pixel(rel_xyz.T, dists, height, width)

    instance_ids = np.zeros(len(world_points), dtype=np.uint16)
    for i in range(len(px)):
        u, v = px[i], py[i]
        if 0 <= u < width and 0 <= v < height:
            instance_ids[i] = merged_mask[v, u]

    return instance_ids


# ===========================================================================
# Output writers
# ===========================================================================

def write_labeled_ply(
    output_path:       str,
    world_points:      np.ndarray,
    point_colors:      np.ndarray,
    instance_ids:      np.ndarray,
    instance_to_label: dict,
) -> None:
    """Write PLY: x y z red green blue instance_id semantic_class."""
    def _clean(v) -> str:
        raw = list(v)[0] if isinstance(v, set) else v
        return str(raw).strip().lower().rstrip(".")

    # Fixed semantic class mapping — must match Step 02 SEM_ID_MAP and Step 03
    # EVAL_CLASS_IDS.  Dynamic alphabetical ordering (the old approach) produced
    # scene-dependent integers that were inconsistent across scans.
    SEM_ID_MAP = {
        "ceiling": 1,
        "wall":    2,
        "door":    3,
        "window":  4,
        "floor":   5,
    }
    semantic_ids = np.array([
        SEM_ID_MAP.get(
            _clean(instance_to_label[iid]) if iid in instance_to_label else "", 0
        )
        for iid in instance_ids
    ], dtype=np.uint8)

    verts = np.array(
        [(x, y, z, r, g, b, iid, sid)
         for (x, y, z), (r, g, b), iid, sid
         in zip(world_points, point_colors, instance_ids, semantic_ids)],
        dtype=[
            ("x", "f4"), ("y", "f4"), ("z", "f4"),
            ("red", "u1"), ("green", "u1"), ("blue", "u1"),
            ("instance_id", "u2"), ("semantic_class", "u1"),
        ],
    )
    PlyData([PlyElement.describe(verts, "vertex")], text=False).write(output_path)
    log(f"Saved labeled PLY → {output_path}")


def write_metadata_csv(
    output_path:          str,
    scene_type:           str,
    scene_confidence:     float,
    instance_to_label:    dict,
    instance_confidences: dict,
    instance_areas:       dict,
    instance_persistence: dict = None,
) -> None:
    """Write per-instance metadata CSV (consumed by Step 02).  v09 appends
    a Persistence column (yaw-ensemble agreement, 0–1; 1.0 when yaw_k=1) —
    trailing column keeps the file readable by existing Step-02 versions."""
    header = ["Scene Type", "Confidence",
              "Instance ID", "Label", "Instance Confidence", "Area",
              "Persistence"]
    instance_persistence = instance_persistence or {}
    rows = []
    for iid, label in instance_to_label.items():
        lbl_text = list(label)[0] if isinstance(label, set) else label
        rows.append([
            scene_type, round(scene_confidence, 4),
            iid, lbl_text,
            round(instance_confidences.get(iid, 0.0), 4),
            instance_areas.get(iid, 0),
            round(instance_persistence.get(iid, 1.0), 3),
        ])
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(sorted(rows, key=lambda r: r[2]))
    log(f"Saved metadata CSV → {output_path}")


# ===========================================================================
# Visualisation helpers
# ===========================================================================

# Colour palette: 20 visually distinct colours (RGBA, alpha=0.50).
# Using a fixed palette avoids random colour clashes between runs.
_PALETTE = [
    (0.122, 0.467, 0.706, 0.50), (1.000, 0.498, 0.055, 0.50),
    (0.173, 0.627, 0.173, 0.50), (0.839, 0.153, 0.157, 0.50),
    (0.580, 0.404, 0.741, 0.50), (0.549, 0.337, 0.294, 0.50),
    (0.890, 0.467, 0.761, 0.50), (0.620, 0.855, 0.898, 0.50),
    (0.737, 0.741, 0.133, 0.50), (0.090, 0.745, 0.812, 0.50),
    (0.682, 0.780, 0.910, 0.50), (1.000, 0.733, 0.471, 0.50),
    (0.596, 0.875, 0.541, 0.50), (1.000, 0.596, 0.588, 0.50),
    (0.773, 0.690, 0.835, 0.50), (0.769, 0.612, 0.580, 0.50),
    (0.969, 0.714, 0.824, 0.50), (0.780, 0.780, 0.780, 0.50),
    (0.859, 0.859, 0.553, 0.50), (0.498, 0.498, 0.498, 0.50),
]


def _draw_single_mask(
    ax,
    mask:   np.ndarray,   # (H, W) bool
    label:  str,
    score:  float,
    colour: tuple,        # (R, G, B, A)
) -> None:
    """
    Draw one mask overlay + bounding box + label annotation onto `ax`.

    The bounding box is derived from the mask's tight pixel extent, so it
    always fits the actual detected region regardless of the scale at which
    the mask was produced.
    """
    h, w = mask.shape

    # Translucent filled overlay
    overlay = mask.reshape(h, w, 1) * np.array(colour).reshape(1, 1, -1)
    ax.imshow(overlay)

    # Tight bounding box from mask extents
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    bw, bh  = x1 - x0, y1 - y0

    box_colour = colour[:3]   # RGB without alpha, for the rectangle edge
    ax.add_patch(plt.Rectangle(
        (x0, y0), bw, bh,
        edgecolor=box_colour, facecolor="none", linewidth=1.5,
    ))

    # Label text just above the top-left corner of the bounding box
    ax.text(
        x0, max(y0 - 3, 0),
        f"{label}  {score:.2f}",
        fontsize=6, color="white",
        bbox=dict(facecolor="black", alpha=0.60, pad=1, edgecolor="none"),
    )


def _save_single_mask_image(
    output_path: str,
    mask:        np.ndarray,   # (H, W) bool
    label:       str,
    score:       float,
    image_bgr:   np.ndarray,
    colour:      tuple,
    title:       str = "",
) -> None:
    """
    Save one individual mask as a separate JPEG — one file per detected instance.

    Each file shows:
      • The background panorama (or residual image)
      • A single translucent coloured mask overlay
      • A bounding box around the mask extent
      • A label + confidence annotation
    """
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    fig, ax   = plt.subplots(figsize=(14, 7))
    ax.imshow(image_rgb)
    _draw_single_mask(ax, mask, label, score, colour)
    if title:
        ax.set_title(title, fontsize=8)
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight", pad_inches=0.0)
    plt.close()
    log(f"      Saved mask image → {output_path}")


def _save_scale_vis(
    output_dir:  str,
    base_name:   str,
    pass_idx:    int,
    scale_idx:   int,            # 1-indexed scale number
    masks:       list,           # list of (H, W) bool
    phrases:     list,
    scores:      list,
    image_bgr:   np.ndarray,     # full-resolution residual image for background
) -> None:
    """
    Save one combined JPEG with ALL masks for this scale stacked together.

    Output: <base>_pass<k>_scale<s>_all.jpg
    All instances within the same scale are overlaid on a single image,
    each with its own colour, bounding box, and label annotation.
    Separate files are NOT produced per instance at the scale level —
    only at the final fused level (see _save_fused_vis).
    """
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    fig, ax   = plt.subplots(figsize=(14, 7))
    ax.imshow(image_rgb)
    for inst_idx, (mask, phrase, score) in enumerate(
            zip(masks, phrases, scores), start=1):
        label  = (list(phrase)[0] if isinstance(phrase, set) else phrase)
        colour = _PALETTE[(inst_idx - 1) % len(_PALETTE)]
        _draw_single_mask(ax, mask, label, score, colour)
    ax.set_title(
        f"Pass {pass_idx} | Scale {scale_idx} ({len(masks)} instances detected)",
        fontsize=8,
    )
    ax.axis("off")
    plt.tight_layout()
    out_path = os.path.join(
        output_dir, f"{base_name}_pass{pass_idx}_scale{scale_idx}_all.jpg"
    )
    plt.savefig(out_path, dpi=150, bbox_inches="tight", pad_inches=0.0)
    plt.close()
    log(f"    Scale {scale_idx} overview → {out_path}")


def _save_fused_vis(
    output_dir:  str,
    base_name:   str,
    pass_idx:    int,
    fused_masks: list,
    phrases:     list,
    scores:      list,
    image_bgr:   np.ndarray,
) -> None:
    """
    Save one combined JPEG with all fused instances for this pass stacked together.

    Output: <base>_pass<k>_fused_all.jpg
    Individual per-instance files are not produced at the fused level;
    the final overview (_pano_segmented.jpg) serves that purpose.
    """
    # ── Combined fused overview ───────────────────────────────────────────
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    fig, ax   = plt.subplots(figsize=(14, 7))
    ax.imshow(image_rgb)
    for inst_idx, (mask, phrase, score) in enumerate(
            zip(fused_masks, phrases, scores), start=1):
        label  = (list(phrase)[0] if isinstance(phrase, set) else phrase)
        colour = _PALETTE[(inst_idx - 1) % len(_PALETTE)]
        _draw_single_mask(ax, mask, label, score, colour)
    ax.set_title(
        f"Pass {pass_idx} | Fused ({len(fused_masks)} instances accepted)",
        fontsize=8,
    )
    ax.axis("off")
    plt.tight_layout()
    combined_path = os.path.join(
        output_dir, f"{base_name}_pass{pass_idx}_fused_all.jpg"
    )
    plt.savefig(combined_path, dpi=150, bbox_inches="tight", pad_inches=0.0)
    plt.close()
    log(f"    Pass {pass_idx} fused overview → {combined_path}")


def save_final_segmentation_jpg(
    output_path:       str,
    merged_mask:       np.ndarray,   # (H, W) uint16
    pano_bgr:          np.ndarray,   # (H, W, 3) uint8 original panorama
    instance_to_label: dict,
    instance_confs:    dict,
) -> None:
    """
    Save a single summary JPEG showing ALL accepted instances overlaid on the
    original (non-residual) panoramic image, with bounding boxes and labels.

    This is the final segmentation overview: <base>_pano_segmented.jpg
    """
    image_rgb = cv2.cvtColor(pano_bgr, cv2.COLOR_BGR2RGB)
    fig, ax   = plt.subplots(figsize=(18, 9))
    ax.imshow(image_rgb)

    unique_ids = [iid for iid in np.unique(merged_mask) if iid != 0]
    for colour_idx, iid in enumerate(unique_ids):
        mask   = (merged_mask == iid)
        label  = instance_to_label.get(iid, "unknown")
        label  = list(label)[0] if isinstance(label, set) else label
        score  = instance_confs.get(iid, 0.0)
        colour = _PALETTE[colour_idx % len(_PALETTE)]
        _draw_single_mask(ax, mask, label, score, colour)

    ax.set_title(
        f"Final segmentation — {len(unique_ids)} instances", fontsize=9
    )
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight", pad_inches=0.0)
    plt.close()
    log(f"Saved final segmentation JPEG → {output_path}")


# ===========================================================================
# Main entry point
# ===========================================================================

RUN_SEED = 0   # fixed RNG seed — reproducibility baseline for EXP-A stability runs


def set_global_seeds(seed: int = RUN_SEED) -> None:
    """
    Pin all RNG sources and disable cuDNN autotuning so repeated runs on the
    same input take identical kernel paths.  Full torch deterministic mode
    (torch.use_deterministic_algorithms) is intentionally NOT enabled: some
    Grounded-SAM ops lack deterministic implementations and would raise.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="OpenScan2BIM Step 01 — Projection + Multi-Scale Multi-Pass Segmentation"
    )
    p.add_argument("--input_e57",    required=True)
    p.add_argument("--output_dir",   required=True)
    p.add_argument("--yaw_k", type=int, default=YAW_ENSEMBLE_K,
                   help="Yaw-ensemble size K (1 = v08 behaviour, no ensemble).")
    p.add_argument("--persistence_min", type=float, default=PERSISTENCE_MIN,
                   help="Keep clusters detected in ≥ this fraction of yaws.")
    p.add_argument("--image_height", type=int,   default=DEFAULT_IMAGE_HEIGHT)
    p.add_argument("--conf_threshold",   type=float, required=True,
                   help="Detection confidence τ (dev Stage 0 used 0.27; legacy default CONF_THRESHOLD=0.29).")
    p.add_argument("--min_area",     type=int,   default=MIN_MASK_AREA,
                   help="Min mask area A_min (px).")
    p.add_argument("--overlap_threshold", type=float, default=OVERLAP_THRESHOLD,
                   help="Max allowed overlap with existing masks (0–1).")
    p.add_argument("--vote_threshold",    type=float, default=VOTE_THRESHOLD,
                   help="Pixel vote threshold for multi-scale fusion (0–1).")
    p.add_argument("--save_depth_tif", action="store_true")
    p.add_argument("--dino_config",
                   default="GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py")
    p.add_argument("--dino_ckpt",
                   default="grounded_sam_checkpoints/groundingdino_swint_ogc.pth")
    p.add_argument("--sam_ckpt",
                   default="grounded_sam_checkpoints/sam_vit_h_4b8939.pth")
    p.add_argument("--sam_version",  default="vit_h",
                   choices=["vit_h", "vit_l", "vit_b"])
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    set_global_seeds()
    t_start = time.time()

    # Allow CLI to override module-level thresholds
    global OVERLAP_THRESHOLD, VOTE_THRESHOLD
    OVERLAP_THRESHOLD = args.overlap_threshold
    VOTE_THRESHOLD    = args.vote_threshold

    base_name = os.path.splitext(os.path.basename(args.input_e57))[0]
    os.makedirs(args.output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    log(f"{'='*60}")
    log(f"OpenScan2BIM — Step 01  (multi-scale multi-pass)")
    log(f"Scan        : {base_name}")
    log(f"Device      : {device}")
    log(f"Scales      : {MULTISCALE_FACTORS}  weights={MULTISCALE_WEIGHTS}")
    log(f"τ={args.conf_threshold}  A_min={args.min_area}  "
        f"overlap_thresh={OVERLAP_THRESHOLD}  vote_thresh={VOTE_THRESHOLD}")
    log(f"{'='*60}")

    # 1 — Spherical projection
    data, sensor_pos = load_point_cloud(args.input_e57)
    pano_image, world_pts, pt_colors, _, _ = create_panorama(
        data=data, sensor_pos=sensor_pos, height=args.image_height,
        output_dir=args.output_dir, base_name=base_name,
        save_depth_tif=args.save_depth_tif,
    )
    pano_path = os.path.join(args.output_dir, f"{base_name}_pano.jpg")
    # pano_image is RGB; cv2.imwrite expects BGR (v08 fix — v07 and earlier
    # wrote the RGB array directly, swapping R/B in every saved panorama)
    cv2.imwrite(pano_path, cv2.cvtColor(pano_image, cv2.COLOR_RGB2BGR))
    log(f"Saved panorama → {pano_path}")

    # 2 — Scene type classification
    clip_model, clip_preprocess, _ = open_clip.create_model_and_transforms(
        "ViT-B-16", pretrained="laion2b_s34b_b88k", device=device,
    )
    clip_tokenizer = open_clip.get_tokenizer("ViT-B-16")
    scene_type, scene_conf = classify_scene_type(
        pano_image, clip_model, clip_preprocess, clip_tokenizer, device
    )

    # 3 — Multi-scale multi-pass segmentation
    dino_model, sam_predictor = load_grounded_sam(
        device=device, dino_config=args.dino_config,
        dino_checkpoint=args.dino_ckpt, sam_checkpoint=args.sam_ckpt,
        sam_version=args.sam_version,
    )
    merged_mask, inst_labels, inst_areas, inst_confs, inst_pers = \
        segment_multipass(
            pano_image_path=pano_path, dino_model=dino_model,
            sam_predictor=sam_predictor, device=device,
            scene_type=scene_type, output_dir=args.output_dir,
            base_name=base_name, conf_threshold=args.conf_threshold,
            min_area=args.min_area,
            yaw_k=args.yaw_k, persistence_min=args.persistence_min,
        )

    # 4 — Back-projection
    instance_ids = back_project_labels(
        merged_mask=merged_mask, world_points=world_pts,
        sensor_pos=sensor_pos, height=args.image_height,
    )

    # 5 — Write outputs
    ply_path  = os.path.join(args.output_dir, f"{base_name}_pano_labeled.ply")
    csv_path  = os.path.join(args.output_dir, f"{base_name}_pano_metadata.csv")
    seg_path  = os.path.join(args.output_dir, f"{base_name}_pano_segmented.jpg")
    mask_path = os.path.join(args.output_dir, f"{base_name}_pano_mask.tif")

    # Labeled point cloud PLY
    write_labeled_ply(ply_path, world_pts, pt_colors, instance_ids, inst_labels)

    # Metadata CSV
    write_metadata_csv(csv_path, scene_type, scene_conf,
                       inst_labels, inst_confs, inst_areas, inst_pers)

    # Final segmentation overview JPEG: all instances on original panorama
    pano_bgr = cv2.imread(pano_path)
    save_final_segmentation_jpg(seg_path, merged_mask, pano_bgr,
                                inst_labels, inst_confs)

    # Instance label mask TIFF (uint16, already written by segment_multipass;
    # log its location here for completeness)
    log(f"Instance mask TIFF → {mask_path}")

    # 6 — Run manifest: full configuration + results, for traceability across
    #     parameter sweeps and repeated stability runs (EXP-A).
    label_values = list(inst_labels.values())
    manifest = {
        "script":    os.path.basename(__file__),
        "version":   "09",
        "pano_color_order": "BGR-standard (v08 fix; v07-and-earlier pano jpgs "
                            "have R/B swapped on disk)",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "elapsed_s": round(time.time() - t_start, 1),
        "seed":      RUN_SEED,
        "args":      vars(args),
        "constants": {
            "YAW_ENSEMBLE_K":     args.yaw_k,
            "PERSISTENCE_MIN":    args.persistence_min,
            "YAW_MATCH_IOU":      YAW_MATCH_IOU,
            "MULTISCALE_FACTORS": list(MULTISCALE_FACTORS),
            "MULTISCALE_WEIGHTS": list(MULTISCALE_WEIGHTS),
            "PASS_ID_OFFSET":     PASS_ID_OFFSET,
            "MIN_RESIDUAL_PX":    MIN_RESIDUAL_PX,
            "OVERLAP_THRESHOLD":  OVERLAP_THRESHOLD,
            "VOTE_THRESHOLD":     VOTE_THRESHOLD,
            "prompt_schedule":    SCENE_TYPE_PROMPTS.get(scene_type,
                                                         SCENE_TYPE_PROMPTS["Indoor"]),
        },
        "environment": {
            "torch":       torch.__version__,
            "numpy":       np.__version__,
            "opencv":      cv2.__version__,
            "cuda_device": torch.cuda.get_device_name(0)
                           if torch.cuda.is_available() else "cpu",
        },
        "results": {
            "scene_type":       scene_type,
            "scene_confidence": round(float(scene_conf), 4),
            "n_instances":      len(inst_labels),
            "label_histogram":  {lbl: label_values.count(lbl)
                                 for lbl in sorted(set(label_values))},
            "persistence": {
                "mean": round(float(np.mean(list(inst_pers.values()))), 3)
                        if inst_pers else None,
                "min":  round(float(min(inst_pers.values())), 3)
                        if inst_pers else None,
                "histogram": {f"{v:.2f}": list(inst_pers.values()).count(v)
                              for v in sorted(set(inst_pers.values()))},
            },
        },
    }
    manifest_path = os.path.join(args.output_dir, f"{base_name}_run_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, ensure_ascii=False)
    log(f"Run manifest → {manifest_path}")

    log(f"{'='*60}")
    log(f"Step 01 complete — {base_name}")
    log(f"  Scene type     : {scene_type}  ({scene_conf:.2%})")
    log(f"  Total instances: {len(inst_labels)}")
    log(f"  Panorama       : {pano_path}")
    if args.save_depth_tif:
        log(f"  Depth TIFF     : {os.path.join(args.output_dir, base_name + '_pano_depth.tif')}")
    log(f"  Mask TIFF      : {mask_path}")
    log(f"  PLY            : {ply_path}")
    log(f"  CSV            : {csv_path}")
    log(f"  Segmented JPEG : {seg_path}")
    log(f"{'='*60}")


if __name__ == "__main__":
    main()
