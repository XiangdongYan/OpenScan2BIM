"""s10 -- per-point semantic labels for every TLS point (Framework v0.1, Layer 3).

Pipeline version of OpenScan2BIM_Step06_PointLabels01.py (2026-10-30): every
input path and the propagation distance are required arguments (the legacy
default --propagation_m 0 differed from the selected 0.05 m); rules unchanged.

3.1 element -> points: each member point of an element takes the element's
    final class (`other` is counted as furniture, seven classes).
3.2 upper-zone split: wall, door, column, and furniture points at or above the
    region's z_u become ceiling (overhead service zone); windows are exempt.
3.3 residual completion for points in no patch:
      z >= z_u                              -> ceiling
      z <= z_f + floor_band (incl. below)   -> floor
      otherwise: class of the nearest element-labelled 1 cm voxel within
      --propagation_m (0 disables), else furniture.

Outputs per station, aligned with the Stage-0 PLY (data/stage0_*) / E57 point order:
  <station>_pred_class.npy    uint8 codes 1-7 (ceiling, wall, door, window, floor, column, furniture)
  <station>_element_id.npy    int16 provisional element id, -1 = none
  <station>_label_source.npy  uint8, see label_source_codes in point_label_summary.json
  element_classes.csv         Final class code per element, before Layer-3 relabelling
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from plyfile import PlyData
from scipy.spatial import cKDTree

CLASS_CODES = {"ceiling": 1, "wall": 2, "door": 3, "window": 4, "floor": 5, "column": 6, "furniture": 7, "other": 7}
SPLIT_CLASSES = [2, 3, 6, 7]          # wall, door, column, furniture
SOURCES = {1: "element", 2: "upper_split", 3: "residual_upper", 4: "residual_floor",
           5: "residual_propagated", 6: "residual_default"}
VOXEL_M = 0.01


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--region_refs", required=True)
    p.add_argument("--atomic_dir", required=True)
    p.add_argument("--member_csv", required=True)
    p.add_argument("--final_csv", required=True)
    p.add_argument("--class_column", default="final_semantic_class")
    p.add_argument("--pano_dir", required=True)
    p.add_argument("--floor_band_m", type=float, default=0.05)
    p.add_argument("--propagation_m", type=float, required=True)
    p.add_argument("--output_dir", required=True)
    return p.parse_args(argv)


def element_ids(station, patch_ids, members):
    patch_to_element = np.full(int(patch_ids.max()) + 1, -1, np.int16)
    for row in members[members.station_id == station].itertuples():
        for pid in str(row.member_patch_ids).split("|"):
            if patch_to_element[int(pid)] != -1:
                raise ValueError(f"{station}: patch {pid} belongs to two elements")
            patch_to_element[int(pid)] = int(row.provisional_geometric_element_id)
    eid = patch_to_element[patch_ids]
    eid[patch_ids == 0] = -1
    return eid


def propagate(xyz, labelled, label, query, radius):
    """Label of the nearest element-labelled 1 cm voxel within `radius` (0 if none)."""
    vox = np.floor(xyz[labelled] / VOXEL_M).astype(np.int64)
    vox, first = np.unique(vox, axis=0, return_index=True)
    tree = cKDTree((vox + 0.5) * VOXEL_M)
    dist, idx = tree.query(xyz[query], distance_upper_bound=radius, workers=-1)
    out = np.zeros(int(query.sum()), np.uint8)
    hit = np.isfinite(dist)
    out[hit] = label[labelled][first][idx[hit]]
    return out


def main(argv=None):
    args = parse_args(argv)
    out = Path(args.output_dir)
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite existing output directory: {out}")
    refs = json.loads(Path(args.region_refs).read_text(encoding="utf-8"))
    z_f, z_u = refs["floor_reference_z_m"], refs["upper_zone_min_z_m"]
    final = pd.read_csv(args.final_csv)
    element_class = dict(zip(final["provisional_geometric_element_id"].astype(int),
                             final[args.class_column].map(CLASS_CODES)))
    if any(pd.isna(v) for v in element_class.values()):
        raise ValueError("final classes outside the taxonomy")
    lut = np.zeros(max(element_class) + 1, np.uint8)
    for eid, code in element_class.items():
        lut[eid] = code
    members = pd.read_csv(args.member_csv)

    out.mkdir(parents=True)
    pd.DataFrame({"element_id": list(element_class), "class_code": list(element_class.values())}).to_csv(
        out / "element_classes.csv", index=False)   # Final class per element, before any Layer-3 relabelling
    summary = {"inputs": vars(args), "region": refs["region"], "floor_reference_z_m": z_f, "upper_zone_min_z_m": z_u,
               "label_source_codes": {str(k): v for k, v in SOURCES.items()}, "stations": {}}
    for s in refs["stations"]:
        patch_ids = np.load(Path(args.atomic_dir) / f"{s}_patch_ids.npy")
        eid = element_ids(s, patch_ids, members)
        v = PlyData.read(Path(args.pano_dir) / s / f"{s}_pano_labeled.ply", mmap=True)["vertex"].data
        xyz = np.column_stack((v["x"], v["y"], v["z"])).astype(np.float64)
        if len(xyz) != len(eid):
            raise ValueError(f"{s}: patch ids and points differ in length")
        z = xyz[:, 2]
        pred = np.where(eid >= 0, lut[np.maximum(eid, 0)], 0).astype(np.uint8)
        source = np.where(eid >= 0, 1, 0).astype(np.uint8)

        split = (eid >= 0) & (z >= z_u) & np.isin(pred, SPLIT_CLASSES)
        pred[split], source[split] = 1, 2
        residual = eid < 0
        upper = residual & (z >= z_u)
        floor = residual & (z <= z_f + args.floor_band_m)
        middle = residual & ~upper & ~floor
        pred[upper], source[upper] = 1, 3
        pred[floor], source[floor] = 5, 4
        if args.propagation_m > 0 and middle.any():
            lab = propagate(xyz, eid >= 0, pred, middle, args.propagation_m)
            mid_idx = np.flatnonzero(middle)
            hit = lab > 0
            pred[mid_idx[hit]], source[mid_idx[hit]] = lab[hit], 5
            middle[mid_idx[hit]] = False
        pred[middle], source[middle] = 7, 6
        if (pred == 0).any():
            raise RuntimeError(f"{s}: unlabelled points remain")

        np.save(out / f"{s}_pred_class.npy", pred)
        np.save(out / f"{s}_element_id.npy", eid)
        np.save(out / f"{s}_label_source.npy", source)
        counts = np.bincount(source, minlength=7)
        summary["stations"][s] = {"points": int(len(pred)),
                                  "label_source_fraction": {SOURCES[k]: round(float(counts[k] / len(pred)), 4) for k in SOURCES},
                                  "class_fraction": {c: round(float((pred == k).mean()), 4)
                                                     for c, k in CLASS_CODES.items() if c != "other"}}
        log(f"{s}: " + ", ".join(f"{k} {v:.3f}" for k, v in summary["stations"][s]["label_source_fraction"].items()))
    (out / "point_label_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    log(f"wrote {out}")


if __name__ == "__main__":
    main()
