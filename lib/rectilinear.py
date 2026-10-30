"""Rectilinear view and metric-geometry helpers shared by s06b and s07.

Consolidated 2026-10-30 from two legacy modules, code otherwise verbatim:
  * OpenScan2BIM_Step04d_RectilinearElementProjection02.py: view selection,
    rendering, montages, quality-weighted voxels; its `metric_geometry` is
    kept as `base_metric_geometry`.
  * OpenScan2BIM_Step04d_RectilinearElementProjection03.py: `planar_shape` and
    `metric_geometry`, which wraps the v02 geometry with unambiguous shape
    ratios. This is the version Step04d v05 (now s06b) calls.

Original v02 description: each model input is reprojected from the
native-camera equirectangular panorama into one to four world-up rectilinear
panels; element support points are reprojected from 3D to draw the red
contour; quality-weighted 2 cm support voxels provide orientation-aware
dimensions and normalized-height statistics in the registered world frame.
Audit tiles are row-major in greedy residual-coverage order.
"""

import html
import json
from math import asin, atan2, ceil, cos, degrees, pi, radians
from pathlib import Path

import cv2
import numpy as np

TILE_PX = 768
FOV_DEG = 100.0
COVER_RADIUS_DEG = 46.0
MAX_VIEWS = 4
TARGET_COVERAGE = 0.98
VOXEL_M = 0.02
MAX_RAW_GEOMETRY_SAMPLES = 1_500_000
MAX_CLUSTER_RAYS = 20_000
MAX_MASK_VOXELS = 300_000


def parse_patch_ids(values):
    return sorted({
        int(token)
        for value in values
        for token in str(value).split("|")
        if token and token != "nan"
    })


def weighted_quantile(values, quantiles, weights):
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    order = np.argsort(values)
    values, weights = values[order], weights[order]
    cumulative = np.cumsum(weights)
    if len(values) == 0 or cumulative[-1] <= 0:
        return np.full(len(np.atleast_1d(quantiles)), np.nan)
    positions = (cumulative - 0.5 * weights) / cumulative[-1]
    return np.interp(np.atleast_1d(quantiles), positions, values)


def quality_weighted_voxels(points, quality):
    keys = np.floor(points / VOXEL_M).astype(np.int32)
    _, inverse = np.unique(keys, axis=0, return_inverse=True)
    count = np.bincount(inverse)
    quality_sum = np.bincount(inverse, weights=quality)
    weights = quality_sum / np.maximum(count, 1)
    weights = np.maximum(weights, 1.0 / 254.0)
    xyz = np.column_stack([
        np.bincount(inverse, weights=quality * points[:, axis]) / np.maximum(quality_sum, 1e-9)
        for axis in range(3)
    ])
    invalid = (quality_sum <= 1e-12) | ~np.isfinite(xyz).all(axis=1)
    if invalid.any():
        fallback = np.column_stack([
            np.bincount(inverse, weights=points[:, axis]) / np.maximum(count, 1)
            for axis in range(3)
        ])
        xyz[invalid] = fallback[invalid]
    return xyz, weights


def robust_extent(values, weights):
    q02, q98 = weighted_quantile(values, [0.02, 0.98], weights)
    return float(max(q98 - q02, 0.0))


def base_metric_geometry(row, points, weights, z_floor, z_top):
    normal = np.array([
        float(row["component_plane_normal_x"]),
        float(row["component_plane_normal_y"]),
        float(row["component_plane_normal_z"]),
    ])
    normal /= max(np.linalg.norm(normal), 1e-12)
    if normal[2] < 0:
        normal *= -1
    centered = points - np.average(points, axis=0, weights=weights)
    role = str(row["geometry_role_evidence"])
    result = {
        "world_frame_definition": "registered_scan_XY_horizontal_Z_up",
        "plane_normal_x": normal[0],
        "plane_normal_y": normal[1],
        "plane_normal_z": normal[2],
        "plane_tilt_from_horizontal_deg": float(np.degrees(np.arccos(np.clip(abs(normal[2]), 0.0, 1.0)))),
        "plane_normal_azimuth_mod180_deg": (
            float(np.degrees(np.arctan2(normal[1], normal[0])) % 180.0)
            if np.hypot(normal[0], normal[1]) > 1e-6 else np.nan
        ),
        "dimension_schema": "unresolved",
        "observed_length_m": np.nan,
        "observed_width_m": np.nan,
        "observed_height_m": np.nan,
        "observed_major_m": np.nan,
        "observed_minor_m": np.nan,
        "observed_aspect_ratio": np.nan,
    }
    if role == "vertical_plane":
        world_up = np.array([0.0, 0.0, 1.0])
        local_vertical = world_up - normal * np.dot(world_up, normal)
        local_vertical /= max(np.linalg.norm(local_vertical), 1e-12)
        local_horizontal = np.cross(world_up, normal)
        local_horizontal /= max(np.linalg.norm(local_horizontal), 1e-12)
        width = robust_extent(centered @ local_horizontal, weights)
        height = robust_extent(centered @ local_vertical, weights)
        result.update({
            "dimension_schema": "vertical_width_height",
            "observed_width_m": width,
            "observed_height_m": height,
            "observed_aspect_ratio": width / max(height, 1e-9),
        })
    else:
        if role == "horizontal_plane":
            coordinates = centered[:, :2]
            basis3 = None
            schema = "horizontal_length_width"
        else:
            reference = np.array([0.0, 0.0, 1.0])
            if abs(np.dot(reference, normal)) > 0.95:
                reference = np.array([1.0, 0.0, 0.0])
            first = np.cross(reference, normal)
            first /= max(np.linalg.norm(first), 1e-12)
            second = np.cross(normal, first)
            second /= max(np.linalg.norm(second), 1e-12)
            basis3 = np.column_stack((first, second))
            coordinates = centered @ basis3
            schema = "oblique_major_minor"
        covariance = np.cov(coordinates, rowvar=False, aweights=weights)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        eigenvectors = eigenvectors[:, np.argsort(eigenvalues)[::-1]]
        projected = coordinates @ eigenvectors
        major = robust_extent(projected[:, 0], weights)
        minor = robust_extent(projected[:, 1], weights)
        result.update({
            "dimension_schema": schema,
            "observed_length_m": major if role == "horizontal_plane" else np.nan,
            "observed_width_m": minor if role == "horizontal_plane" else np.nan,
            "observed_major_m": major if role != "horizontal_plane" else np.nan,
            "observed_minor_m": minor if role != "horizontal_plane" else np.nan,
            "observed_aspect_ratio": major / max(minor, 1e-9),
        })
        if role == "horizontal_plane":
            major_world = eigenvectors[:, 0]
            result["horizontal_major_axis_azimuth_mod180_deg"] = float(
                np.degrees(np.arctan2(major_world[1], major_world[0])) % 180.0
            )

    z = points[:, 2]
    mean = float(np.average(z, weights=weights))
    sd = float(np.sqrt(np.average((z - mean) ** 2, weights=weights)))
    q025, q50, q975 = weighted_quantile(z, [0.025, 0.5, 0.975], weights)
    height_range = max(z_top - z_floor, 1e-9)
    normalized = (z - z_floor) / height_range
    nmean = float(np.average(normalized, weights=weights))
    nsd = float(np.sqrt(np.average((normalized - nmean) ** 2, weights=weights)))
    nq025, nq50, nq975 = weighted_quantile(normalized, [0.025, 0.5, 0.975], weights)
    total = float(weights.sum())
    fraction = lambda lo, hi: float(weights[(normalized >= lo) & (normalized < hi)].sum() / total)
    result.update({
        "elevation_reference_status": "station_floor_to_highest_upper_reference",
        "station_floor_reference_z_m": z_floor,
        "station_top_reference_z_m": z_top,
        "z_world_mean_m": mean,
        "z_world_sd_m": sd,
        "z_world_q025_m": q025,
        "z_world_q50_m": q50,
        "z_world_q975_m": q975,
        "height_norm_mean": nmean,
        "height_norm_sd": nsd,
        "height_norm_q025": nq025,
        "height_norm_q50": nq50,
        "height_norm_q975": nq975,
        "fraction_below_reference": float(weights[normalized < 0.0].sum() / total),
        "fraction_bottom_00_10": fraction(0.0, 0.1),
        "fraction_lower_10_30": fraction(0.1, 0.3),
        "fraction_middle_30_70": fraction(0.3, 0.7),
        "fraction_upper_70_90": fraction(0.7, 0.9),
        "fraction_top_90_100": fraction(0.9, 1.0000001),
        "fraction_above_reference": float(weights[normalized > 1.0].sum() / total),
    })
    return result


def view_basis(forward):
    forward = forward / max(np.linalg.norm(forward), 1e-12)
    world_up = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(forward, world_up)) > 0.96:
        world_up = np.array([0.0, 1.0, 0.0])
    right = np.cross(world_up, forward)
    right /= max(np.linalg.norm(right), 1e-12)
    up = np.cross(forward, right)
    up /= max(np.linalg.norm(up), 1e-12)
    return forward, right, up


def choose_view_centers(rays):
    if len(rays) > MAX_CLUSTER_RAYS:
        rays = rays[np.linspace(0, len(rays) - 1, MAX_CLUSTER_RAYS, dtype=np.int64)]
    uncovered = np.ones(len(rays), dtype=bool)
    centers = []
    sequence = []
    threshold = cos(radians(COVER_RADIUS_DEG))
    while uncovered.mean() > 1.0 - TARGET_COVERAGE and len(centers) < MAX_VIEWS:
        active = rays[uncovered]
        candidate_indices = np.linspace(0, len(active) - 1, min(128, len(active)), dtype=np.int64)
        candidates = active[candidate_indices]
        mean = active.mean(axis=0)
        if np.linalg.norm(mean) > 1e-6:
            candidates = np.vstack((mean / np.linalg.norm(mean), candidates))
        counts = np.array([(active @ candidate >= threshold).sum() for candidate in candidates])
        center = candidates[int(np.argmax(counts))]
        center /= np.linalg.norm(center)
        centers.append(center)
        newly_covered = uncovered & (rays @ center >= threshold)
        uncovered &= ~newly_covered
        sequence.append({
            "view_rank": len(centers),
            "yaw_deg": round(degrees(atan2(center[1], center[0])), 3),
            "pitch_deg": round(degrees(asin(float(np.clip(center[2], -1.0, 1.0)))), 3),
            "incremental_selection_coverage": round(float(newly_covered.mean()), 6),
            "cumulative_selection_coverage": round(1.0 - float(uncovered.mean()), 6),
        })
    coverage = 1.0 - float(uncovered.mean())
    return centers, coverage, sequence


def render_rectilinear(pano, center, rays, tile_px=TILE_PX):
    forward, right, up = view_basis(center)
    tan_half = np.tan(np.radians(FOV_DEG) / 2.0)
    axis = (np.arange(tile_px, dtype=np.float64) + 0.5) / tile_px * 2.0 - 1.0
    xx, yy = np.meshgrid(axis, -axis)
    directions = (
        forward[None, None, :]
        + xx[:, :, None] * tan_half * right[None, None, :]
        + yy[:, :, None] * tan_half * up[None, None, :]
    )
    directions /= np.linalg.norm(directions, axis=2, keepdims=True)
    theta = np.arctan2(directions[:, :, 1], directions[:, :, 0])
    phi = np.arcsin(np.clip(directions[:, :, 2], -1.0, 1.0))
    map_x = ((theta + pi) / (2 * pi) * pano.shape[1]).astype(np.float32)
    map_y = ((pi / 2 - phi) / pi * pano.shape[0]).astype(np.float32)
    clean = cv2.remap(pano, map_x, map_y, cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_WRAP)

    depth = rays @ forward
    x = (rays @ right) / np.maximum(depth, 1e-12)
    y = (rays @ up) / np.maximum(depth, 1e-12)
    visible = (depth > 0) & (np.abs(x) <= tan_half) & (np.abs(y) <= tan_half)
    u = np.clip(((x[visible] / tan_half + 1.0) * 0.5 * tile_px).astype(np.int32), 0, tile_px - 1)
    v = np.clip(((1.0 - y[visible] / tan_half) * 0.5 * tile_px).astype(np.int32), 0, tile_px - 1)
    mask = np.zeros((tile_px, tile_px), dtype=np.uint8)
    mask[v, u] = 255
    display = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((13, 13), dtype=np.uint8))
    display = cv2.dilate(display, np.ones((5, 5), dtype=np.uint8))
    overlay = clean.copy()
    contours, _ = cv2.findContours(display, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (255, 0, 0), 5)
    return clean, overlay, visible


def montage(images):
    if len(images) == 1:
        return images[0]
    columns = 2
    rows = int(ceil(len(images) / columns))
    canvas = np.full((rows * TILE_PX, columns * TILE_PX, 3), 32, dtype=np.uint8)
    for index, image in enumerate(images):
        row, column = divmod(index, columns)
        canvas[row * TILE_PX:(row + 1) * TILE_PX, column * TILE_PX:(column + 1) * TILE_PX] = image
    return canvas


def ordered_audit_montage(images, sequence):
    canvas = montage(images).copy()
    columns = 1 if len(images) == 1 else 2
    for index, meta in enumerate(sequence):
        row, column = divmod(index, columns)
        x0, y0 = column * TILE_PX, row * TILE_PX
        label = (
            f"View {meta['view_rank']}/{len(sequence)}  coverage rank  "
            f"yaw={meta['yaw_deg']:+.1f} deg  pitch={meta['pitch_deg']:+.1f} deg  "
            f"new={100.0 * meta['incremental_selection_coverage']:.1f}%"
        )
        cv2.rectangle(canvas, (x0, y0), (x0 + TILE_PX - 1, y0 + 42), (24, 24, 24), -1)
        cv2.putText(canvas, label, (x0 + 12, y0 + 28), cv2.FONT_HERSHEY_SIMPLEX,
                    0.58, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.rectangle(canvas, (x0, y0), (x0 + TILE_PX - 1, y0 + TILE_PX - 1),
                      (255, 214, 10), 3)
    return canvas


def write_index(output_dir, result):
    cards = []
    for row in result.to_dict("records"):
        image_name = Path(row["rectilinear_ordered_audit_path"]).name
        sequence = json.loads(row["rectilinear_view_sequence_json"])
        sequence_text = "\n".join(
            f"View {item['view_rank']}: yaw={item['yaw_deg']:+.3f} deg; "
            f"pitch={item['pitch_deg']:+.3f} deg; incremental="
            f"{100.0 * item['incremental_selection_coverage']:.2f}%; cumulative="
            f"{100.0 * item['cumulative_selection_coverage']:.2f}%"
            for item in sequence
        )
        cards.append(
            f"<article><h2>{html.escape(row['element_record_id'])} / "
            f"{html.escape(row['station_id'])}</h2>"
            f"<img src=\"rectilinear_ordered_audit/{html.escape(image_name)}\">"
            f"<pre>{html.escape(sequence_text)}</pre></article>"
        )
    document = """<!doctype html><html><head><meta charset="utf-8">
<title>Ordered rectilinear montage audit</title><style>
body{font-family:Arial,sans-serif;margin:24px;background:#f3f4f6;color:#111827}
h1{margin-bottom:6px}p{max-width:980px}main{display:grid;grid-template-columns:repeat(auto-fit,minmax(520px,1fr));gap:18px}
article{background:white;border:1px solid #d1d5db;padding:14px}h2{font-size:17px;margin:0 0 10px}
img{width:100%;height:auto;display:block;background:#181818}pre{white-space:pre-wrap;font-size:12px;line-height:1.45}
</style></head><body><h1>Ordered rectilinear montage audit</h1>
<p>Tiles are row-major in greedy residual-coverage order. View 1 covers the largest target-ray subset; each later view covers the largest remaining subset. Labels are audit-only and are not part of the Qwen model image.</p><main>"""
    document += "".join(cards) + "</main></body></html>"
    (output_dir / "index.html").write_text(document, encoding="utf-8")


def write_rgb(path, image):
    if not cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 92]):
        raise OSError(path)


# ── from OpenScan2BIM_Step04d_RectilinearElementProjection03.py ──

def planar_shape(centered, weights, basis):
    coordinates = centered @ basis
    covariance = np.cov(coordinates, rowvar=False, aweights=weights)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    projected = coordinates @ eigenvectors[:, np.argsort(eigenvalues)[::-1]]
    major = robust_extent(projected[:, 0], weights)
    minor = robust_extent(projected[:, 1], weights)
    return major, minor


def metric_geometry(row, points, weights, z_floor, z_top):
    result = base_metric_geometry(row, points, weights, z_floor, z_top)
    normal = np.array([
        float(row["component_plane_normal_x"]),
        float(row["component_plane_normal_y"]),
        float(row["component_plane_normal_z"]),
    ])
    normal /= max(np.linalg.norm(normal), 1e-12)
    centered = points - np.average(points, axis=0, weights=weights)
    role = str(row["geometry_role_evidence"])
    vertical_span = float(result["z_world_q975_m"] - result["z_world_q025_m"])
    story_height = max(float(z_top - z_floor), 1e-9)

    if role == "horizontal_plane":
        major = float(result["observed_length_m"])
        minor = float(result["observed_width_m"])
        result.update({
            "dimension_schema": "horizontal_planar_major_minor",
            "vertical_span_m": vertical_span,
            "horizontal_in_plane_width_m": np.nan,
            "vertical_slenderness": np.nan,
            "story_coverage": vertical_span / story_height,
            "planar_major_extent_m": major,
            "planar_minor_extent_m": minor,
            "planar_elongation": major / max(minor, 1e-9),
        })
    else:
        world_up = np.array([0.0, 0.0, 1.0])
        horizontal = np.cross(world_up, normal)
        horizontal /= max(np.linalg.norm(horizontal), 1e-12)
        up_in_plane = np.cross(normal, horizontal)
        up_in_plane /= max(np.linalg.norm(up_in_plane), 1e-12)
        width = robust_extent(centered @ horizontal, weights)
        major, minor = planar_shape(centered, weights, np.column_stack((horizontal, up_in_plane)))
        result.update({
            "dimension_schema": "nonhorizontal_vertical_span_width",
            "vertical_span_m": vertical_span,
            "horizontal_in_plane_width_m": width,
            "vertical_slenderness": vertical_span / max(width, 1e-9),
            "story_coverage": vertical_span / story_height,
            "planar_major_extent_m": major,
            "planar_minor_extent_m": minor,
            "planar_elongation": major / max(minor, 1e-9),
        })

    for stale in (
        "observed_length_m", "observed_width_m", "observed_height_m",
        "observed_major_m", "observed_minor_m", "observed_aspect_ratio",
    ):
        result.pop(stale, None)
    return result
