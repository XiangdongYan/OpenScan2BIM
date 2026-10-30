"""
OpenScan2BIM -- Step 01c: Occlusion-Aware Station-Local Surface Reassembly
============================================================================
v16 (2026-09-23) -- evaluate the door-boundary veto only after normal,
delta-perpendicular, and in-plane-gap proposal gates pass. This keeps the
door audit limited to pairs that geometry would otherwise consider.

v15 (2026-09-23) -- retain the v14 geometry baseline and add one narrowly
scoped semantic-boundary rule: an atomic patch backed by a Grounded-SAM door
instance cannot merge with a non-door patch. Door evidence remains a candidate
attribute, not a final semantic class. No window or other thing class is used.

v14 (2026-09-23) -- retain the v12 local-surface baseline and reject remote,
tiny satellite patches before they can enlarge a dominant surface footprint;
the remote threshold is fixed at 0.75 m after the Station-7 v13 sensitivity run.

WHY THIS EXISTS
----------------
Step01b v04 emits *atomic* 3D-connected planar fragments only.  It performs
no cross-gap coplanar merge.  Step01c is therefore the sole owner of
station-local association: this sensitivity run uses the projected-gap design
of v04 and a 0.20 m component-RMS screening threshold. Per-member plane
offset remains a diagnostic rather than a second hard veto. This removes the
inherited 3D bbox-gap criterion: for sparse TLS observations, meaningful
fragmentation occurs along a plane, not normal to it. It instead separates
(a) a permissive, fixed delta_perp candidate gate from (b) a
projected-footprint, in-plane bridge test. A broad candidate gate is never
itself a merge decision: ray evidence and global plane validation remain
mandatory.

Deliberately NOT done in this pass (kept for a later cross-station layer,
per the framework agreed with the user + Codex):
  - cross-station association (still Step04's job, now consuming these
    surfaces instead of raw patches once validated)
  - final class_final / vision-evidence combination
  - column association (cylindrical columns still come from Step01b's
    separate residue cylinder fit; rectangular/box columns are NOT
    addressed by this pass either -- their 4 faces have ~90 deg normal
    differences and fail the coplanarity gate by design, same limitation
    flagged earlier for Step04)

METHOD
------
1. Candidate pairs: same-station patches with compatible normals and mutual
   centroid-to-plane distance delta_perp <= 0.25 m.  This is deliberately a
   broad *proposal* gate, not an acceptance tolerance.
2. In-plane bridge: construct a common candidate-plane basis, project each
   patch's observed footprint into it, and compute g_parallel as the shortest
   distance between sampled projected footprints.  Pairs with g_parallel >
   6 m are not proposed in this pilot.  The bridge samples lie on the common
   candidate plane, not on a 3D bbox-to-bbox line.
3. Occlusion test: sample the in-plane bridge at density proportional to its
   length.  For each sample,
   ray-cast from the station's sensor and compare to the station's own
   min-range spherical grid (reused from Step04):
     grid_range < sample_range - OCC_MARGIN   -> occluded  (something IS
         in front -- supports "real gap caused by a foreground object")
     |grid_range - sample_range| <= SUPPORT_TOL -> consistent (the surface
         was actually scanned right there -- even stronger support)
     grid_range > sample_range + SUPPORT_TOL  -> contradicted (the beam at
         that exact direction traveled PAST the hypothesised surface to
         something farther -- i.e. this location is empty air, not a
         surface at all -- evidence AGAINST merging)
     no return at that pixel                  -> unknown (weak, uninformative)
   Accept the edge only if contradicted evidence is negligible AND
   (occluded + consistent) covers most of the informative samples.
4. Incremental component-plane agglomeration over ray-supported edges. Edges
   are considered from tighter to looser geometric fit. The fixed 0.25 m
   delta_perp threshold is retained only as an atomic-pair proposal gate.
   Before accepting an edge, each current component is represented by the
   exact PCA plane fitted from all of its raw member points. The two component
   planes must remain normal-compatible and close in the normal direction;
   their prospective union must also pass the 0.20 m RMS screen. This avoids
   both a level-chain merge and a false split caused by one distant but
   otherwise compatible member pair.
5. Global validation per surviving component (>=2 patches): independently
   refit ONE plane to ALL member points (not a re-average of per-patch
   planes) and reject a rare component whose RMS residual is too large. This
   is a verification of the incremental moment calculation, not the ordinary
   association mechanism.
6. Attach factorized geometry evidence to each surviving local surface.
Keep the local above/below occupancy separately from a station-relative
elevation band inferred from broad horizontal surfaces. The retained legacy
geom_state is a traceability field only, not a final BIM class. Final
   wall/floor/ceiling/column/door/window/furniture/other classification
   remains an element-stage Step04 task.

This is explicitly a PILOT on one station for human review before deciding
whether/how to calibrate thresholds and generalize to the other 6 stations
or wire into Step04.

This parameter setting was retained after comparative inspection on 2ndLab-6.
v12 retains v11's geometry association exactly, but replaces the v11
`area < 0.50 m2 -> horizontal_context=unknown` reliability guard. That guard
only hid unstable local occupancy values. v12 retains those values as
`local_occupancy_context` and separately estimates station-relative vertical
reference bands from broad horizontal local surfaces. `elevation_band` says
only whether a horizontal observation is near the floor reference, near an
upper reference level, or at an intermediate elevation. It is not a final
floor/ceiling class and never changes association.

Outputs (default `outputs53_step01c_local_surfaces12_elevation_bands/`):
  <station>_surfaces.csv            one row per surviving surface
  <station>_elevation_reference_bands.csv  station-local reference bands
  <station>_patch_to_surface.csv    (patch_id -> surface_id)
  <station>_merge_log.csv           every in-plane bridge candidate + verdict
                                     (accepted / rejected_occlusion /
                                     rejected_residual) with evidence
  <station>_pair_gate_audit.csv     every patch pair, including its first
                                     failed gate (normal / delta_perp /
                                     g_parallel), for recall diagnostics
"""
import argparse
import os, csv, time
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
from plyfile import PlyData

from OpenScan2BIM_Step01b_Primitives04 import classify_patch, factorized_planar_attributes
from OpenScan2BIM_Step04_Fusion03 import spherical_pixels, min_range_grid, FS_GRID_H

PANO_DIR = "./outputs18"
PRIM_DIR = "./outputs210_step01b2_door_preservation01_station5"
OUT_DIR = "./outputs212_step01c_local_surfaces16_door_station5"
STATION = "2ndLab-5"

# ── geometry proposal gate ───────────────────────────────────────────────────
THETA_SURF_DEG = 5.0
BAND_SURF = 0.250          # m -- current local-surface proposal tolerance
G_PARALLEL_MAX = 6.00      # m -- maximum projected-footprint gap in this pilot

# A tiny isolated return several metres away should not enlarge the footprint
# of a dominant patch merely because the two fitted planes are compatible.
# Rejected satellites remain auditable singleton local surfaces.
REMOTE_SATELLITE_GAP_MIN_M = 0.75
REMOTE_SATELLITE_MAX_AREA_M2 = 0.05
REMOTE_SATELLITE_MAX_AREA_RATIO = 0.05

# ── occlusion ray-cast test ──
N_REP = 400                 # downsample cap per patch for nearest-pair search
N_SAMPLES_MIN = 9           # bridge samples for short gaps (endpoints excluded)
BRIDGE_SAMPLE_SPACING = 0.20  # m -- denser samples for long in-plane bridges
N_SAMPLES_MAX = 61          # cap to keep one pair test bounded
OCC_MARGIN = 0.10           # m -- grid nearer than sample by this -> occluded
SUPPORT_TOL = 0.15          # m -- |grid - sample| within this -> plane-consistent
CONTRA_MAX_FRAC = 0.15      # max allowed contradicted / total_samples
MERGE_MIN_EVIDENCE_FRAC = 0.60   # (occluded+consistent) / informative >= this

# ── post-merge global-refit validation ──
RESID_MAX = 0.200           # m -- current component-plane RMS screen

# ── station-relative vertical reference bands (diagnostic only) ─────────────
# These are deliberately relative to the registered station/world Z frame,
# not fixed absolute building elevations. They are neither merge gates nor
# final construction classes.
ELEVATION_SUPPORT_MIN_AREA_M2 = 1.00
ELEVATION_CLUSTER_TOL = 0.15       # m -- broad horizontal-band clustering
ELEVATION_ASSIGN_TOL = 0.25        # m -- proximity to a station reference
ELEVATION_MIN_CLEARANCE = 1.50     # m above floor to qualify as upper reference


def log(msg):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def attach_station_elevation_bands(surf_rows):
    """Attach relative vertical context without re-labelling a BIM element.

    A station may contain several ceiling or soffit levels, so there is one
    lower floor reference and zero or more upper reference bands. A small
    horizontal surface close to an upper band is only *near* that reference:
    it remains a separate local surface unless the Step01c geometry and
    free-space evidence accepted its association.
    """
    support = [r for r in surf_rows
               if r["planar_orientation"] == "horizontal"
               and r["area_m2"] >= ELEVATION_SUPPORT_MIN_AREA_M2]
    support.sort(key=lambda r: r["z_ref_m"])
    bands = []
    for row in support:
        z = float(row["z_ref_m"])
        if not bands or abs(z - bands[-1]["z_ref_m"]) > ELEVATION_CLUSTER_TOL:
            bands.append({"surface_ids": [row["surface_id"]],
                          "total_area_m2": float(row["area_m2"]),
                          "z_ref_m": z})
            continue
        band = bands[-1]
        old_area = band["total_area_m2"]
        new_area = old_area + float(row["area_m2"])
        band["z_ref_m"] = ((band["z_ref_m"] * old_area) +
                            (z * float(row["area_m2"]))) / new_area
        band["total_area_m2"] = new_area
        band["surface_ids"].append(row["surface_id"])

    floor_band = min(bands, key=lambda b: b["z_ref_m"]) if bands else None
    upper_bands = ([] if floor_band is None else
                   [b for b in bands
                    if b["z_ref_m"] >= floor_band["z_ref_m"] + ELEVATION_MIN_CLEARANCE])

    for row in surf_rows:
        row["station_floor_reference_z_m"] = "" if floor_band is None else round(floor_band["z_ref_m"], 3)
        row["elevation_band"] = "NA"
        row["elevation_reference_z_m"] = ""
        row["elevation_reference_delta_m"] = ""
        if row["planar_orientation"] != "horizontal" or floor_band is None:
            continue
        z = float(row["z_ref_m"])
        floor_delta = abs(z - floor_band["z_ref_m"])
        closest = floor_band
        role = "floor_reference"
        delta = floor_delta
        if upper_bands:
            upper = min(upper_bands, key=lambda b: abs(z - b["z_ref_m"]))
            upper_delta = abs(z - upper["z_ref_m"])
            if upper_delta < delta:
                closest, role, delta = upper, "near_upper_reference", upper_delta
        row["elevation_reference_z_m"] = round(closest["z_ref_m"], 3)
        row["elevation_reference_delta_m"] = round(delta, 3)
        if delta <= ELEVATION_ASSIGN_TOL:
            row["elevation_band"] = role
        elif z > floor_band["z_ref_m"] + ELEVATION_ASSIGN_TOL:
            row["elevation_band"] = "intermediate"
        else:
            row["elevation_band"] = "unresolved"

    report_rows = []
    for i, band in enumerate(bands, start=1):
        role = "floor_reference" if band is floor_band else (
            "upper_reference" if band in upper_bands else "intermediate_reference")
        report_rows.append({
            "band_id": i,
            "role": role,
            "z_ref_m": round(band["z_ref_m"], 3),
            "total_area_m2": round(band["total_area_m2"], 3),
            "support_surface_ids": "|".join(str(s) for s in band["surface_ids"]),
        })
    return report_rows


def load_patches(station):
    path = os.path.join(PRIM_DIR, f"{station}_patches.csv")
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"No planar patch rows in {path}")
    state_field = "legacy_geom_state" if "legacy_geom_state" in rows[0] else "guess"
    P = len(rows)
    d = {
        "pid": np.array([int(r["patch_id"]) for r in rows], np.int32),
        # Kept under the historic internal key so the candidate audit remains
        # comparable, but sourced from the explicit traceability field.
        "guess": [r[state_field] for r in rows],
        "area": np.array([float(r["area_m2"]) for r in rows], np.float64),
        "normal": np.array([[float(r["nx"]), float(r["ny"]), float(r["nz"])] for r in rows], np.float64),
        "d": np.array([float(r["d"]) for r in rows], np.float64),
        "centroid": np.array([[float(r["cx"]), float(r["cy"]), float(r["cz"])] for r in rows], np.float64),
        "z_p02": np.array([float(r["z_p02"]) for r in rows], np.float64),
        "z_p98": np.array([float(r["z_p98"]) for r in rows], np.float64),
        "frac_above": np.array([float(r["frac_above"]) for r in rows], np.float64),
        "frac_below": np.array([float(r["frac_below"]) for r in rows], np.float64),
        "door_candidate": np.array([int(r.get("door_candidate", 0)) for r in rows], np.uint8),
        "door_evidence_status": [r.get("door_evidence_status", "none") for r in rows],
        "door_instance_ids": [
            {int(value) for value in r.get("door_instance_ids", "").split("|") if value}
            for r in rows
        ],
    }
    assert np.all(d["pid"] == np.arange(1, P + 1)), "patch ids not contiguous"
    return d


class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))

    def find(self, a):
        while self.parent[a] != a:
            self.parent[a] = self.parent[self.parent[a]]
            a = self.parent[a]
        return a

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)


def plane_basis(normal):
    """Return two orthonormal axes spanning the supplied unit normal's plane."""
    seed = np.array([1.0, 0.0, 0.0])
    if abs(normal @ seed) > 0.9:
        seed = np.array([0.0, 1.0, 0.0])
    axis_u = np.cross(normal, seed)
    axis_u /= np.linalg.norm(axis_u)
    axis_v = np.cross(normal, axis_u)
    return axis_u, axis_v


def projected_bridge(points_a, points_b, normal_a, normal_b, center_a, center_b):
    """Find the shortest observed footprint bridge in a shared plane basis.

    The returned endpoints are projected to the average local plane, so ray
    evidence tests a corridor *along* the hypothesised surface rather than a
    line that can drift in the normal direction.
    """
    normal_b_aligned = normal_b if normal_a @ normal_b >= 0 else -normal_b
    normal = normal_a + normal_b_aligned
    normal /= np.linalg.norm(normal)
    origin = (center_a + center_b) * 0.5
    axis_u, axis_v = plane_basis(normal)

    uv_a = np.column_stack([(points_a - origin) @ axis_u,
                            (points_a - origin) @ axis_v])
    uv_b = np.column_stack([(points_b - origin) @ axis_u,
                            (points_b - origin) @ axis_v])
    delta = uv_a[:, None, :] - uv_b[None, :, :]
    distances = np.linalg.norm(delta, axis=-1)
    idx_a, idx_b = np.unravel_index(np.argmin(distances), distances.shape)

    endpoint_a, endpoint_b = points_a[idx_a].astype(np.float64), points_b[idx_b].astype(np.float64)
    offset_a = normal @ (endpoint_a - origin)
    offset_b = normal @ (endpoint_b - origin)
    common_offset = (offset_a + offset_b) * 0.5
    endpoint_a += (common_offset - offset_a) * normal
    endpoint_b += (common_offset - offset_b) * normal
    return float(distances[idx_a, idx_b]), endpoint_a, endpoint_b


def parse_args():
    parser = argparse.ArgumentParser(
        description="OpenScan2BIM Step01c v16 -- door-aware station-local surface reassembly")
    parser.add_argument("--station", default=STATION)
    parser.add_argument("--pano_dir", default=PANO_DIR,
                        help="Step01 labeled-PLY directory containing <station>/")
    parser.add_argument("--primitives_dir", default=PRIM_DIR,
                        help="Step01b v04 atomic-patch output directory")
    parser.add_argument("--output_dir", default=OUT_DIR)
    return parser.parse_args()


def main():
    global PANO_DIR, PRIM_DIR, OUT_DIR, STATION
    args = parse_args()
    PANO_DIR, PRIM_DIR, OUT_DIR, STATION = (
        args.pano_dir, args.primitives_dir, args.output_dir, args.station)
    os.makedirs(OUT_DIR, exist_ok=True)
    station = STATION
    log(f"[{station}] loading patches / points / sensor ...")
    meta = load_patches(station)
    P = len(meta["pid"])

    ply = os.path.join(PANO_DIR, station, f"{station}_pano_labeled.ply")
    data = PlyData.read(ply)["vertex"].data
    pts = np.stack([np.asarray(data["x"]), np.asarray(data["y"]), np.asarray(data["z"])],
                   axis=-1).astype(np.float32)
    del data

    patch_ids = np.load(os.path.join(PRIM_DIR, f"{station}_patch_ids.npy"))
    assert len(patch_ids) == len(pts)

    q = np.load(os.path.join(PRIM_DIR, f"{station}_point_quality.npz"))
    sensor = q["sensor_pos"].astype(np.float64)
    log(f"[{station}] {P} patches, {len(pts):,} points, sensor={sensor}")

    # per-patch member index ranges (1-indexed patch ids; id 0 = unassigned)
    order = np.argsort(patch_ids, kind="stable")
    sorted_ids = patch_ids[order]
    starts = np.searchsorted(sorted_ids, np.arange(1, P + 2))
    member_idx = [order[starts[p - 1]:starts[p]] for p in range(1, P + 1)]

    # Exact sufficient statistics make prospective component-plane tests
    # O(1) per candidate edge, without retaining a copied point cloud per
    # component.  They are deliberately computed from raw member points,
    # rather than from patch-plane averages.
    patch_stats = []
    for idx in member_idx:
        pp = pts[idx].astype(np.float64)
        patch_stats.append((len(pp), pp.sum(axis=0), pp.T @ pp))

    def plane_from_stats(stats):
        count, sum_xyz, sum_outer = stats
        mean = sum_xyz / count
        cov = sum_outer / count - np.outer(mean, mean)
        eigvals, eigvecs = np.linalg.eigh(cov)
        normal = eigvecs[:, 0]
        rms = float(np.sqrt(max(eigvals[0], 0.0)))
        return mean, normal, rms

    def add_stats(left, right):
        return (left[0] + right[0], left[1] + right[1], left[2] + right[2])

    # Per-patch representatives approximate observed 2D footprints. This
    # atomic-patch version deliberately avoids axis-aligned 3D bbox gaps.
    reps = []
    for p in range(P):
        pp = pts[member_idx[p]]
        stride = max(1, len(pp) // N_REP)
        reps.append(pp[::stride][:N_REP])
    log(f"[{station}] projected-footprint representatives computed")

    # station-wide min-range spherical grid (reused mechanism from Step04)
    rel = pts.astype(np.float64) - sensor
    rng = np.maximum(np.linalg.norm(rel, axis=1), 1e-6).astype(np.float32)
    grid = min_range_grid(rel.astype(np.float32), rng, FS_GRID_H)
    log(f"[{station}] min-range grid built ({FS_GRID_H}x{2*FS_GRID_H})")

    def ray_lookup(sample_pts):
        r = sample_pts.astype(np.float64) - sensor
        rr = np.maximum(np.linalg.norm(r, axis=1), 1e-6).astype(np.float32)
        pix = spherical_pixels(r.astype(np.float32), rr, FS_GRID_H)
        return grid[pix], rr

    # ── candidate pairs: broad common-plane proposal -> in-plane bridge ──
    cos_surf = np.cos(np.deg2rad(THETA_SURF_DEG))
    nrm, d, ctr = meta["normal"], meta["d"], meta["centroid"]
    merge_log = []
    pair_gate_audit = []
    candidate_edges = []
    n_geom_candidates = 0

    for i in range(P):
        for j in range(i + 1, P):
            dot = float(nrm[j] @ nrm[i])
            angle_deg = float(np.rad2deg(np.arccos(np.clip(abs(dot), -1.0, 1.0))))
            audit = {
                "patch_a": i + 1, "patch_b": j + 1,
                "door_candidate_a": int(meta["door_candidate"][i]),
                "door_candidate_b": int(meta["door_candidate"][j]),
                "door_instance_ids_a": "|".join(str(value) for value in sorted(meta["door_instance_ids"][i])),
                "door_instance_ids_b": "|".join(str(value) for value in sorted(meta["door_instance_ids"][j])),
                "normal_angle_deg": round(angle_deg, 3),
                "delta_a_to_b_m": "", "delta_b_to_a_m": "",
                "delta_perp_m": "", "g_parallel_m": "", "gate": "",
            }
            if abs(dot) < cos_surf:
                audit["gate"] = "rejected_normal"
                pair_gate_audit.append(audit)
                continue

            off_i = float(abs(ctr[j] @ nrm[i] + d[i]))
            off_j = float(abs(ctr[i] @ nrm[j] + d[j]))
            delta_perp = max(off_i, off_j)
            audit.update({
                "delta_a_to_b_m": round(off_i, 4),
                "delta_b_to_a_m": round(off_j, 4),
                "delta_perp_m": round(delta_perp, 4),
            })
            if delta_perp > BAND_SURF:
                audit["gate"] = "rejected_delta_perp"
                pair_gate_audit.append(audit)
                continue

            g_parallel, p_a, p_b = projected_bridge(
                reps[i], reps[j], nrm[i], nrm[j], ctr[i], ctr[j])
            audit["g_parallel_m"] = round(g_parallel, 4)
            if g_parallel > G_PARALLEL_MAX:
                audit["gate"] = "rejected_g_parallel_max"
                pair_gate_audit.append(audit)
                continue

            door_a = bool(meta["door_candidate"][i])
            door_b = bool(meta["door_candidate"][j])
            same_door_instance = bool(
                meta["door_instance_ids"][i] & meta["door_instance_ids"][j]
            )
            if door_a != door_b or (door_a and door_b and not same_door_instance):
                audit["gate"] = "rejected_door_boundary"
                pair_gate_audit.append(audit)
                continue

            smaller_area = float(min(meta["area"][i], meta["area"][j]))
            larger_area = float(max(meta["area"][i], meta["area"][j]))
            area_ratio = smaller_area / max(larger_area, 1e-12)
            audit["smaller_patch_area_m2"] = round(smaller_area, 4)
            audit["patch_area_ratio"] = round(area_ratio, 4)
            if (
                g_parallel > REMOTE_SATELLITE_GAP_MIN_M
                and smaller_area < REMOTE_SATELLITE_MAX_AREA_M2
                and area_ratio < REMOTE_SATELLITE_MAX_AREA_RATIO
            ):
                audit["gate"] = "rejected_remote_tiny_satellite"
                pair_gate_audit.append(audit)
                merge_log.append({
                    "patch_a": i + 1, "patch_b": j + 1,
                    "guess_a": meta["guess"][i], "guess_b": meta["guess"][j],
                    "delta_perp_m": round(delta_perp, 4),
                    "g_parallel_m": round(g_parallel, 4),
                    "smaller_patch_area_m2": round(smaller_area, 4),
                    "patch_area_ratio": round(area_ratio, 4),
                    "n_bridge_samples": 0,
                    "n_occluded": 0, "n_consistent": 0,
                    "n_contradicted": 0, "n_unknown": 0,
                    "support_fraction": 0.0,
                    "component_reject_reason": "remote_tiny_satellite",
                    "verdict": "rejected_remote_tiny_satellite",
                })
                continue

            n_geom_candidates += 1
            n_samples = int(np.clip(
                np.ceil(g_parallel / BRIDGE_SAMPLE_SPACING), N_SAMPLES_MIN, N_SAMPLES_MAX))
            fr = np.linspace(0.1, 0.9, n_samples)[:, None]
            samples = p_a[None, :] * (1 - fr) + p_b[None, :] * fr
            g_range, s_range = ray_lookup(samples)
            finite = np.isfinite(g_range)
            occluded = finite & (g_range < s_range - OCC_MARGIN)
            consistent = finite & (np.abs(g_range - s_range) <= SUPPORT_TOL) & ~occluded
            contradicted = finite & (g_range > s_range + SUPPORT_TOL)
            unknown = ~finite
            informative = int((occluded | consistent | contradicted).sum())
            n_occ, n_con, n_contra = int(occluded.sum()), int(consistent.sum()), int(contradicted.sum())

            contra_frac = n_contra / n_samples
            if informative == 0:
                verdict = "rejected_no_evidence"
            elif contra_frac > CONTRA_MAX_FRAC:
                verdict = "rejected_contradicted"
            elif (n_occ + n_con) / informative >= MERGE_MIN_EVIDENCE_FRAC:
                verdict = "accepted_geom"
            else:
                verdict = "rejected_occlusion"

            row = {
                "patch_a": i + 1, "patch_b": j + 1,
                "guess_a": meta["guess"][i], "guess_b": meta["guess"][j],
                "delta_perp_m": round(delta_perp, 4),
                "g_parallel_m": round(g_parallel, 4), "n_bridge_samples": n_samples,
                "smaller_patch_area_m2": round(smaller_area, 4),
                "patch_area_ratio": round(area_ratio, 4),
                "n_occluded": n_occ, "n_consistent": n_con,
                "n_contradicted": n_contra, "n_unknown": int(unknown.sum()),
                "support_fraction": round((n_occ + n_con) / max(informative, 1), 4),
                "component_reject_reason": "",
                "verdict": verdict,
            }
            merge_log.append(row)
            audit["gate"] = verdict
            pair_gate_audit.append(audit)
            if verdict == "accepted_geom":
                candidate_edges.append({
                    "i": i, "j": j,
                    "delta_perp": delta_perp,
                    "g_parallel": g_parallel,
                    "support_fraction": (n_occ + n_con) / max(informative, 1),
                    "row": row,
                })

    log(f"[{station}] in-plane bridge candidates: {n_geom_candidates}  "
        f"passed ray evidence: {len(candidate_edges)}")

    # ── constrained agglomeration: prevent level chains without complete-link false splits ──
    # A prior version required EVERY cross-component atomic pair to pass the
    # 25 cm band. It incorrectly separated two fragments of the same ceiling
    # when just one remote pair exceeded the band, even though their combined
    # raw-point plane had 1.75 cm RMS. Here the component plane is the model.
    uf = UnionFind(P)
    component_members = {p: [p] for p in range(P)}
    component_stats = {p: patch_stats[p] for p in range(P)}
    component_rejections = {
        "component_normal": 0,
        "component_delta_perp": 0,
        "component_residual": 0,
    }
    n_component_merges = 0
    n_redundant_edges = 0

    for edge in sorted(candidate_edges,
                       key=lambda e: (e["delta_perp"], e["g_parallel"], -e["support_fraction"])):
        i, j, row = edge["i"], edge["j"], edge["row"]
        left_root, right_root = uf.find(i), uf.find(j)
        if left_root == right_root:
            row["verdict"] = "accepted_redundant"
            n_redundant_edges += 1
            continue

        left_stats = component_stats[left_root]
        right_stats = component_stats[right_root]
        left_mu, left_normal, _ = plane_from_stats(left_stats)
        right_mu, right_normal, _ = plane_from_stats(right_stats)
        component_angle = float(np.rad2deg(np.arccos(np.clip(
            abs(left_normal @ right_normal), -1.0, 1.0))))
        row["component_normal_angle_deg"] = round(component_angle, 3)
        if component_angle > THETA_SURF_DEG:
            row["verdict"] = "rejected_component_normal"
            row["component_reject_reason"] = "component_normal"
            component_rejections["component_normal"] += 1
            continue

        left_d = -float(left_normal @ left_mu)
        right_d = -float(right_normal @ right_mu)
        component_delta = max(
            abs(float(right_mu @ left_normal + left_d)),
            abs(float(left_mu @ right_normal + right_d)),
        )
        row["component_plane_delta_perp_m"] = round(component_delta, 4)
        if component_delta > BAND_SURF:
            row["verdict"] = "rejected_component_delta_perp"
            row["component_reject_reason"] = "component_delta_perp"
            component_rejections["component_delta_perp"] += 1
            continue

        prospective_stats = add_stats(left_stats, right_stats)
        _, _, prospective_rms = plane_from_stats(prospective_stats)
        row["component_prospective_rms_m"] = round(prospective_rms, 4)
        if prospective_rms > RESID_MAX:
            row["verdict"] = "rejected_component_residual"
            row["component_reject_reason"] = "component_residual"
            component_rejections["component_residual"] += 1
            continue

        keep, drop = (left_root, right_root) if left_root < right_root else (right_root, left_root)
        uf.parent[drop] = keep
        component_members[keep].extend(component_members.pop(drop))
        component_stats[keep] = prospective_stats
        component_stats.pop(drop)
        row["verdict"] = "accepted_component"
        n_component_merges += 1

    log(f"[{station}] component-plane agglomeration: {n_component_merges} component joins, "
        f"{n_redundant_edges} redundant edges, "
        f"component rejections={component_rejections}")

    # ── components -> global refit residual validation ──
    root = np.array([uf.find(p) for p in range(P)])
    comp_members: dict = {}
    for p, r in enumerate(root):
        comp_members.setdefault(int(r), []).append(p)

    final_root = np.arange(P)   # after residual veto, singles fall back to themselves
    resid_rejections = []
    for r, members in comp_members.items():
        if len(members) < 2:
            continue
        idx = np.concatenate([member_idx[p] for p in members])
        pp = pts[idx].astype(np.float64)
        mu = pp.mean(axis=0)
        cov = (pp - mu).T @ (pp - mu) / len(pp)
        w, v = np.linalg.eigh(cov)
        resid = float(np.sqrt(max(w[0], 0.0)))
        normal_global = v[:, 0]
        max_member_offset = float(np.abs((ctr[members] - mu) @ normal_global).max())
        if resid > RESID_MAX:
            resid_rejections.append((members, resid, max_member_offset))
            for p in members:
                final_root[p] = p           # veto: revert to separate patches
            for row in merge_log:
                a, b = row["patch_a"] - 1, row["patch_b"] - 1
                if row["verdict"] in ("accepted_component", "accepted_redundant") and a in members and b in members:
                    row["verdict"] = "rejected_global_plane"
                    row["resid_m"] = round(resid, 4)
                    row["max_member_offset_m"] = round(max_member_offset, 4)
        else:
            for p in members:
                final_root[p] = min(members)
    log(f"[{station}] components with >=2 patches: {sum(1 for m in comp_members.values() if len(m)>=2)}  "
        f"vetoed by global-plane checks: {len(resid_rejections)}")

    # ── build final surfaces ──
    surf_of = {}
    next_id = 0
    surf_members: dict = {}
    for p in range(P):
        r = final_root[p]
        if r not in surf_of:
            surf_of[r] = next_id
            next_id += 1
        surf_members.setdefault(surf_of[r], []).append(p)

    surf_rows = []
    patch_map_rows = []
    for sid, members in sorted(surf_members.items()):
        idx = np.concatenate([member_idx[p] for p in members])
        pp = pts[idx].astype(np.float64)
        mu = pp.mean(axis=0)
        cov = (pp - mu).T @ (pp - mu) / len(pp)
        w, v = np.linalg.eigh(cov)
        nrm_s = v[:, 0]
        resid_s = float(np.sqrt(max(w[0], 0.0)))
        max_member_offset_s = float(np.abs((ctr[members] - mu) @ nrm_s).max())

        area = float(meta["area"][members].sum())
        z02 = float(min(meta["z_p02"][p] for p in members))
        z98 = float(max(meta["z_p98"][p] for p in members))
        areas_w = meta["area"][members]
        wsum = areas_w.sum()
        above = float((meta["frac_above"][members] * areas_w).sum() / wsum) if wsum > 0 else 0.0
        below = float((meta["frac_below"][members] * areas_w).sum() / wsum) if wsum > 0 else 0.0

        guess, wall_tilt, slab_tilt = classify_patch(nrm_s, np.array([z02, z98]), area, above, below)
        factorized = factorized_planar_attributes(nrm_s, guess)
        factorized.pop("observation_extent")
        factorized["observation_unit"] = "local_surface"
        # These fields report measured geometry, not a construction class.
        # "oblique" avoids implying that a small/noisy normal proves a real
        # inclined building surface. Keep local occupancy separate from the
        # station-relative elevation context attached after all surfaces exist.
        if factorized["planar_orientation"] == "inclined":
            factorized["planar_orientation"] = "oblique"
        if factorized["horizontal_context"] == "mid":
            factorized["horizontal_context"] = "ambiguous"
        local_occupancy_context = factorized.pop("horizontal_context")
        factorized["local_occupancy_context"] = local_occupancy_context

        pre_guesses = [meta["guess"][p] for p in members]
        from collections import Counter
        pre_dominant = Counter(pre_guesses).most_common(1)[0][0]
        door_instances = sorted(set().union(*(meta["door_instance_ids"][p] for p in members)))
        door_statuses = sorted({meta["door_evidence_status"][p] for p in members
                                if meta["door_evidence_status"][p] != "none"})

        surf_rows.append({
            "surface_id": sid + 1,
            "n_patches_merged": len(members),
            "member_patch_ids": "|".join(str(p + 1) for p in members),
            "pre_dominant_guess": pre_dominant,
            "pre_dominant_legacy_state": pre_dominant,
            "geom_state": guess,
            "door_candidate": int(any(meta["door_candidate"][p] for p in members)),
            "door_instance_ids": "|".join(str(value) for value in door_instances),
            "door_evidence_status": "|".join(door_statuses) if door_statuses else "none",
            **factorized,
            "area_m2": round(area, 3),
            "z_ref_m": round(float(mu[2]), 3),
            "zspan_m": round(z98 - z02, 3),
            "resid_rms_m": round(resid_s, 4),
            "max_member_offset_m": round(max_member_offset_s, 4),
            "wall_tilt_deg": round(wall_tilt, 2),
            "slab_tilt_deg": round(slab_tilt, 2),
            "frac_above": round(above, 3), "frac_below": round(below, 3),
        })
        for p in members:
            patch_map_rows.append({"patch_id": p + 1, "surface_id": sid + 1})

    elevation_band_rows = attach_station_elevation_bands(surf_rows)

    n_merged_surfaces = sum(1 for r in surf_rows if r["n_patches_merged"] >= 2)
    log(f"[{station}] {P} patches -> {len(surf_rows)} surfaces "
        f"({n_merged_surfaces} are real merges of >=2 patches)")

    with open(os.path.join(OUT_DIR, f"{station}_surfaces.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(surf_rows[0].keys()))
        w.writeheader(); w.writerows(surf_rows)
    with open(os.path.join(OUT_DIR, f"{station}_patch_to_surface.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["patch_id", "surface_id"])
        w.writeheader(); w.writerows(patch_map_rows)
    with open(os.path.join(OUT_DIR, f"{station}_elevation_reference_bands.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["band_id", "role", "z_ref_m", "total_area_m2", "support_surface_ids"])
        w.writeheader(); w.writerows(elevation_band_rows)
    with open(os.path.join(OUT_DIR, f"{station}_merge_log.csv"), "w", newline="", encoding="utf-8") as f:
        # Component-stage diagnostics exist only on some candidate rows; use
        # the ordered union of keys rather than silently dropping them.
        fn = []
        for row in merge_log:
            fn.extend(key for key in row if key not in fn)
        w = csv.DictWriter(f, fieldnames=fn or ["patch_a"])
        w.writeheader(); w.writerows(merge_log)
    with open(os.path.join(OUT_DIR, f"{station}_pair_gate_audit.csv"), "w", newline="", encoding="utf-8") as f:
        audit_fields = []
        for row in pair_gate_audit:
            audit_fields.extend(key for key in row if key not in audit_fields)
        w = csv.DictWriter(f, fieldnames=audit_fields)
        w.writeheader(); w.writerows(pair_gate_audit)

    log(f"wrote {station}_surfaces.csv / _patch_to_surface.csv / _elevation_reference_bands.csv / _merge_log.csv / _pair_gate_audit.csv to {OUT_DIR}")

    # ── summary printout: which merges changed geom_state vs pre-fusion dominant guess ──
    print()
    print("=== merges where the surface's geom_state differs from the pre-merge dominant guess ===")
    for r in surf_rows:
        if r["n_patches_merged"] >= 2 and r["geom_state"] != r["pre_dominant_guess"]:
            print(f"  surface {r['surface_id']}: patches {r['member_patch_ids']}  "
                  f"pre={r['pre_dominant_guess']} -> geom_state={r['geom_state']}  "
                  f"area={r['area_m2']}m2 zspan={r['zspan_m']}m resid={r['resid_rms_m']}m")

    print()
    print("=== all real merges (>=2 patches), regardless of class change ===")
    for r in sorted(surf_rows, key=lambda x: -x["n_patches_merged"]):
        if r["n_patches_merged"] >= 2:
            print(f"  surface {r['surface_id']}: {r['n_patches_merged']} patches "
                  f"[{r['member_patch_ids']}]  pre={r['pre_dominant_guess']} geom_state={r['geom_state']} "
                  f"area={r['area_m2']}m2 zspan={r['zspan_m']}m resid={r['resid_rms_m']}m")

    print()
    from collections import Counter
    print("verdict counts across all tested candidate pairs:")
    for k, v in Counter(row["verdict"] for row in merge_log).most_common():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
