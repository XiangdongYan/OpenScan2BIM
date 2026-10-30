"""Recompute rectilinear element geometry with unambiguous shape ratios."""

import numpy as np

import OpenScan2BIM_Step04d_RectilinearElementProjection02 as v02


OUT_DIR = "outputs162_step04d_rectilinear_metric03_smoke_station5_000_016"
BASE_METRIC_GEOMETRY = v02.metric_geometry


def planar_shape(centered, weights, basis):
    coordinates = centered @ basis
    covariance = np.cov(coordinates, rowvar=False, aweights=weights)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    projected = coordinates @ eigenvectors[:, np.argsort(eigenvalues)[::-1]]
    major = v02.robust_extent(projected[:, 0], weights)
    minor = v02.robust_extent(projected[:, 1], weights)
    return major, minor


def metric_geometry(row, points, weights, z_floor, z_top):
    result = BASE_METRIC_GEOMETRY(row, points, weights, z_floor, z_top)
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
        width = v02.robust_extent(centered @ horizontal, weights)
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


def main():
    v02.OUT_DIR = OUT_DIR
    v02.metric_geometry = metric_geometry
    v02.main()


if __name__ == "__main__":
    main()
