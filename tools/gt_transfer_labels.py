"""Transfer annotated representative labels back to every original E57 point.

Pipeline copy of transfer_semantic_point_gt01.py (2026-10-30), which wrote
gt/dev_station5_6_7/point_labels_v02; arguments are now required.

Input is the annotation cloud from make_semantic_point_gt_cloud01.py after it
has been labelled (e.g. in CloudCompare) with a `class` scalar field holding
the fine GT codes 0-10. CloudCompare may export scalar fields with a
`scalar_` prefix and as floats; both are accepted.

Every original point is mapped through its 1 cm voxel key to the voxel's
representative, so the result is exact and needs no nearest-neighbour search.
Output per station: <station>_gt_fine.npy (uint8, aligned with the outputs18
PLY / E57 point order).
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
from plyfile import PlyData

FINE_CLASSES = {
    0: "ignore", 1: "ceiling", 2: "wall", 3: "door", 4: "window", 5: "floor",
    6: "column", 7: "furniture", 8: "other", 9: "overhead_mep", 10: "beam",
}


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--annotated_ply", required=True)
    p.add_argument("--cloud_dir", required=True)
    p.add_argument("--pano_dir", required=True,
                   help="Stage-0 directory; overrides the pano_dir recorded in the GT manifest (old outputs18)")
    p.add_argument("--class_field", default="class")
    p.add_argument("--output_dir", required=True)
    return p.parse_args(argv)


def field(data, name):
    for candidate in (name, f"scalar_{name}"):
        if candidate in data.dtype.names:
            return np.asarray(data[candidate])
    raise KeyError(f"field '{name}' not found; available: {data.dtype.names}")


def as_int(values, name):
    rounded = np.rint(values.astype(np.float64))
    if np.abs(values - rounded).max() > 1e-3:
        raise ValueError(f"field '{name}' contains non-integer values")
    return rounded.astype(np.int64)


def main(argv=None):
    args = parse_args(argv)
    cloud_dir = Path(args.cloud_dir)
    out = Path(args.output_dir)
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite existing output directory: {out}")
    manifest = json.loads((cloud_dir / "manifest.json").read_text(encoding="utf-8"))
    manifest["pano_dir"] = args.pano_dir
    rep_keys = np.load(cloud_dir / "rep_voxel_keys.npy")
    n_rep = len(rep_keys)

    log(f"reading {args.annotated_ply}")
    data = PlyData.read(args.annotated_ply)["vertex"].data
    rep_id = as_int(field(data, "rep_hi"), "rep_hi") * 65536 + as_int(field(data, "rep_lo"), "rep_lo")
    label = as_int(field(data, args.class_field), args.class_field)

    problems = []
    bad_ids = (rep_id < 0) | (rep_id >= n_rep)
    if bad_ids.any():
        problems.append(f"{int(bad_ids.sum())} points have rep_id outside 0..{n_rep - 1}")
    bad_cls = ~np.isin(label, list(FINE_CLASSES))
    if bad_cls.any():
        problems.append(f"{int(bad_cls.sum())} points have class codes outside 0-10: "
                        f"{sorted(set(label[bad_cls].tolist()))[:10]}")
    if problems:
        raise ValueError("; ".join(problems))
    counts = np.bincount(rep_id, minlength=n_rep)
    missing, duplicated = int((counts == 0).sum()), int((counts > 1).sum())
    if duplicated:
        order = np.argsort(rep_id, kind="stable")
        r, lab = rep_id[order], label[order]
        conflict = int(((r[1:] == r[:-1]) & (lab[1:] != lab[:-1])).sum())
        if conflict:
            raise ValueError(f"{conflict} duplicated representatives carry conflicting labels")
    if missing:
        raise ValueError(f"{missing:,} of {n_rep:,} representatives are missing from the annotated file "
                         "(points were deleted during annotation)")

    rep_label = np.zeros(n_rep, np.uint8)
    rep_label[rep_id] = label.astype(np.uint8)

    out.mkdir(parents=True)
    origin = np.array(manifest["voxel_origin"])
    voxel = manifest["voxel_m"]
    bits = manifest["voxel_key_bits_per_axis"]
    summary = {"annotated_ply": str(args.annotated_ply), "cloud_dir": str(cloud_dir),
               "fine_classes": FINE_CLASSES, "representatives": n_rep,
               "duplicated_representatives_consistent": duplicated,
               "representative_class_counts": {FINE_CLASSES[k]: int(c) for k, c in
                                               enumerate(np.bincount(rep_label, minlength=11))},
               "stations": {}}
    for s in manifest["stations"]:
        v = PlyData.read(Path(manifest["pano_dir"]) / s / f"{s}_pano_labeled.ply", mmap=True)["vertex"].data
        xyz = np.column_stack((v["x"], v["y"], v["z"])).astype(np.float64)
        ijk = np.floor((xyz - origin) / voxel).astype(np.int64)
        del xyz
        keys = (ijk[:, 0] << (2 * bits)) | (ijk[:, 1] << bits) | ijk[:, 2]
        del ijk
        pos = np.searchsorted(rep_keys, keys)
        pos = np.clip(pos, 0, n_rep - 1)
        if not np.array_equal(rep_keys[pos], keys):
            raise RuntimeError(f"{s}: some original points fall outside the annotation voxels")
        gt = rep_label[pos]
        np.save(out / f"{s}_gt_fine.npy", gt)
        hist = np.bincount(gt, minlength=11)
        summary["stations"][s] = {"points": int(len(gt)),
                                  "class_points": {FINE_CLASSES[k]: int(c) for k, c in enumerate(hist)},
                                  "ignore_fraction": round(float(hist[0]) / len(gt), 4)}
        log(f"  {s}: {len(gt):,} points labelled; ignore fraction {hist[0] / len(gt):.3f}")
    (out / "transfer_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    log(f"wrote {out}")


if __name__ == "__main__":
    main()
