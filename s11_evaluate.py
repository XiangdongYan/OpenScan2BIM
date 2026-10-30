"""s11 -- Seven-class point-level evaluation against the fine point GT (Framework v0.1, Section 8).

Pipeline version of evaluate_semantic_point_gt02.py (2026-10-30): GT and cloud
directories are required arguments; metrics unchanged.

Fine GT codes 0-10 collapse to the seven classes (other -> furniture,
overhead_mep/beam -> ceiling); predicted `other` (code 8) counts as furniture.
GT ignore (0) is excluded. A prediction of 0 (no prediction) counts as a miss.

Reported per station and pooled, point-weighted and 1 cm voxel-weighted, on three
subsets: all GT points; points with a prediction; points farther than
--boundary_cm from a different GT label. Layer checks:
  Layer 1  element purity (share of an element's GT points in its majority class)
  Layer 2  element accuracy against the GT majority, overall and on pure elements
  Layer 3  accuracy and GT composition per label source, when the prediction
           directory provides <station>_label_source.npy
"""
import argparse
import csv
import itertools
import json
from pathlib import Path

import numpy as np
from plyfile import PlyData

CLASSES = ["ceiling", "wall", "door", "window", "floor", "column", "furniture"]
GT_LUT = np.array([0, 1, 2, 3, 4, 5, 6, 7, 7, 1, 1], np.uint8)    # fine GT code -> eval code
PRED_LUT = np.array([0, 1, 2, 3, 4, 5, 6, 7, 7], np.uint8)        # predicted code (8 = other) -> eval code
PURE = 0.8
N = len(CLASSES) + 1


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--gt_dir", required=True, help="per-station <station>_gt_fine.npy (gt/.../point_labels_v02)")
    p.add_argument("--pred_dir", required=True, help="<station>_pred_class.npy, optional _element_id / _label_source")
    p.add_argument("--cloud_dir", required=True, help="GT annotation cloud (gt/.../annotation_cloud)")
    p.add_argument("--boundary_cm", type=int, default=2)
    p.add_argument("--pano_dir", required=True,
                   help="Stage-0 directory; overrides the pano_dir recorded in the GT manifest (old outputs18)")
    p.add_argument("--output_dir", required=True)
    return p.parse_args(argv)


def confusion(gt, pred, weight=None):
    return np.bincount(gt.astype(np.int64) * N + pred, weights=weight, minlength=N * N).reshape(N, N).astype(np.float64)


def metrics(cm):
    rows, present = {}, []
    for k, name in enumerate(CLASSES, start=1):
        tp = cm[k, k]
        fn = cm[k, :].sum() - tp
        fp = cm[1:, k].sum() - tp
        support = cm[k, :].sum()
        iou = tp / (tp + fp + fn) if tp + fp + fn > 0 else None
        rows[name] = {"precision": tp / (tp + fp) if tp + fp > 0 else None,
                      "recall": tp / (tp + fn) if tp + fn > 0 else None, "iou": iou, "gt_support": support}
        if support > 0:
            present.append(iou)
    labelled, missed = cm[1:, :].sum(), cm[1:, 0]
    return {
        "per_class": rows,
        "miou": float(np.mean(present)) if present else None,
        "overall_accuracy": float(np.trace(cm[1:, 1:]) / labelled) if labelled else None,
        "prediction_coverage": float(1 - missed.sum() / labelled) if labelled else None,
        "unpredicted_gt_composition": ({c: float(m / missed.sum()) for c, m in zip(CLASSES, missed)}
                                       if missed.sum() else None),
        "confusion_gt_rows_1to7_pred_cols_0to7": cm[1:, :].tolist(),
    }


def station_index(station, manifest, rep_keys):
    """Per-point representative index (= 1 cm voxel) for one station."""
    v = PlyData.read(Path(manifest["pano_dir"]) / station / f"{station}_pano_labeled.ply", mmap=True)["vertex"].data
    xyz = np.column_stack((v["x"], v["y"], v["z"])).astype(np.float64)
    ijk = np.floor((xyz - np.array(manifest["voxel_origin"])) / manifest["voxel_m"]).astype(np.int64)
    del xyz
    b = manifest["voxel_key_bits_per_axis"]
    keys = (ijk[:, 0] << (2 * b)) | (ijk[:, 1] << b) | ijk[:, 2]
    pos = np.clip(np.searchsorted(rep_keys, keys), 0, len(rep_keys) - 1)
    if not np.array_equal(rep_keys[pos], keys):
        raise RuntimeError(f"{station}: points outside the annotation voxels")
    return pos.astype(np.int32)


def boundary_reps(rep_keys, rep_label, bits, radius):
    """Representatives with a differently labelled representative within `radius` voxels."""
    m = (1 << bits) - 1
    ijk = np.column_stack(((rep_keys >> (2 * bits)) & m, (rep_keys >> bits) & m, rep_keys & m))
    flag = np.zeros(len(rep_keys), bool)
    for d in itertools.product(range(-radius, radius + 1), repeat=3):
        if d == (0, 0, 0) or sum(x * x for x in d) > radius * radius:
            continue
        n = ijk + np.array(d)
        ok = ((n >= 0) & (n <= m)).all(axis=1)
        nk = (n[:, 0] << (2 * bits)) | (n[:, 1] << bits) | n[:, 2]
        p = np.clip(np.searchsorted(rep_keys, nk), 0, len(rep_keys) - 1)
        lab = rep_label[p]
        flag |= ok & (rep_keys[p] == nk) & (lab != rep_label) & (lab > 0) & (rep_label > 0)
    return flag


def element_rows(station, gt, pred, element_id, weight, element_class=None):
    """`element_class` maps element id -> Final class code (before Layer-3 relabelling).
    Without it, the element's class is its most frequent point prediction."""
    keep = (gt > 0) & (element_id >= 0)
    ids, inv = np.unique(element_id[keep], return_inverse=True)
    pts = np.zeros((len(ids), N), np.int64)
    np.add.at(pts, (inv, gt[keep]), 1)
    vox = np.zeros((len(ids), N))
    np.add.at(vox, (inv, gt[keep]), weight[keep])
    if element_class is not None:
        pred_of = np.array([element_class[int(e)] for e in ids], np.uint8)
    else:
        votes = np.zeros((len(ids), N), np.int64)
        np.add.at(votes, (inv, pred[keep]), 1)
        pred_of = votes.argmax(axis=1)
    majority = pts.argmax(axis=1)
    return [{"station": station, "element_id": int(e), "gt_points": int(p.sum()), "gt_voxel_weight": round(float(v.sum()), 1),
             "gt_majority": CLASSES[m - 1], "purity": round(float(p.max() / p.sum()), 4),
             "predicted": CLASSES[c - 1] if c else "none", "correct": bool(m == c)}
            for e, p, v, m, c in zip(ids, pts, vox, majority, pred_of)]


def element_summary(rows):
    pur = np.array([r["purity"] for r in rows])
    ok = np.array([r["correct"] for r in rows])
    vw = np.array([r["gt_voxel_weight"] for r in rows])
    pure = pur >= PURE
    return {"station_element_observations": len(rows),
            "layer1_observations_purity_below_0.8": int((~pure).sum()),
            "layer1_voxel_share_in_impure_elements": float(vw[~pure].sum() / vw.sum()),
            "layer1_median_purity": float(np.median(pur)),
            "layer2_accuracy_all_elements": float(ok.mean()),
            "layer2_accuracy_pure_elements": float(ok[pure].mean()) if pure.any() else None,
            "layer2_pure_elements": int(pure.sum())}


def main(argv=None):
    args = parse_args(argv)
    out = Path(args.output_dir)
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite existing output directory: {out}")
    pred_dir = Path(args.pred_dir)
    manifest = json.loads((Path(args.cloud_dir) / "manifest.json").read_text(encoding="utf-8"))
    manifest["pano_dir"] = args.pano_dir
    rep_keys = np.load(Path(args.cloud_dir) / "rep_voxel_keys.npy")
    stations = manifest["stations"]
    pred_summary = next(iter(sorted(pred_dir.glob("*summary*.json"))), None)
    source_names = (json.loads(pred_summary.read_text(encoding="utf-8")).get("label_source_codes")
                    if pred_summary else None)
    element_class = None
    if (pred_dir / "element_classes.csv").exists():
        table = np.loadtxt(pred_dir / "element_classes.csv", delimiter=",", skiprows=1, dtype=np.int64, ndmin=2)
        element_class = {int(e): int(PRED_LUT[c]) for e, c in table}

    gt, pred, pos, vweight, rep_label = {}, {}, {}, {}, np.zeros(len(rep_keys), np.uint8)
    for s in stations:
        gt[s] = GT_LUT[np.load(Path(args.gt_dir) / f"{s}_gt_fine.npy")]
        pred[s] = PRED_LUT[np.load(pred_dir / f"{s}_pred_class.npy")]
        if len(gt[s]) != len(pred[s]):
            raise ValueError(f"{s}: GT and prediction lengths differ")
        pos[s] = station_index(s, manifest, rep_keys)
        vweight[s] = 1.0 / np.bincount(pos[s], minlength=len(rep_keys))[pos[s]]
        rep_label[pos[s]] = gt[s]
    boundary = boundary_reps(rep_keys, rep_label, manifest["voxel_key_bits_per_axis"], args.boundary_cm)

    subsets = ["all", "predicted_only", f"boundary_excluded_{args.boundary_cm}cm"]
    pooled = {(sub, w): np.zeros((N, N)) for sub in subsets for w in ("point", "voxel")}
    report = {"classes": CLASSES, "inputs": vars(args), "boundary_representative_fraction":
              float(boundary[rep_label > 0].mean()), "stations": {}}
    elements, source_cm = [], {}
    for s in stations:
        g, p = gt[s], pred[s]
        masks = {"all": g > 0, "predicted_only": (g > 0) & (p > 0), subsets[2]: (g > 0) & ~boundary[pos[s]]}
        st = {"ignored_points": int((g == 0).sum())}
        for sub, mk in masks.items():
            for w, weight in (("point", None), ("voxel", vweight[s][mk])):
                cm = confusion(g[mk], p[mk], weight)
                pooled[(sub, w)] += cm
                st[f"{sub}/{w}_weighted"] = metrics(cm)
        report["stations"][s] = st
        if (pred_dir / f"{s}_element_id.npy").exists():
            elements += element_rows(s, g, p, np.load(pred_dir / f"{s}_element_id.npy"), vweight[s], element_class)
        if source_names and (pred_dir / f"{s}_label_source.npy").exists():
            src = np.load(pred_dir / f"{s}_label_source.npy")
            for code, name in source_names.items():
                mk = (src == int(code)) & (g > 0)
                source_cm[name] = source_cm.get(name, np.zeros((N, N))) + confusion(g[mk], p[mk], vweight[s][mk])
        print(f"{s}: voxel mIoU {st['all/voxel_weighted']['miou']:.4f}, coverage {st['all/voxel_weighted']['prediction_coverage']:.4f}")
    report["pooled_over_stations"] = {f"{sub}/{w}_weighted": metrics(cm) for (sub, w), cm in pooled.items()}
    report["pooled_note"] = "station sums; surfaces seen by several stations count once per station"

    summary_rows = []
    for scope, block in [*((s, report["stations"][s]) for s in stations), ("pooled", report["pooled_over_stations"])]:
        for key, m in block.items():
            if "/" in key:
                sub, w = key.split("/")
                summary_rows.append({"scope": scope, "subset": sub, "weighting": w, "coverage": m["prediction_coverage"],
                                     "overall_accuracy": m["overall_accuracy"], "miou": m["miou"],
                                     **{f"iou_{c}": m["per_class"][c]["iou"] for c in CLASSES}})
    files = {"metrics_summary.csv": summary_rows}
    if elements:
        report["element_summary"] = element_summary(elements)
        files["element_purity.csv"] = elements
    if source_cm:
        files["label_source_summary.csv"] = [
            {"label_source": name, "gt_voxel_weight": round(float(cm[1:].sum()), 1),
             "accuracy": float(np.trace(cm[1:, 1:]) / cm[1:].sum()) if cm[1:].sum() else None,
             **{f"gt_{c}": float(cm[k].sum() / cm[1:].sum()) if cm[1:].sum() else None for k, c in enumerate(CLASSES, 1)}}
            for name, cm in source_cm.items()]

    out.mkdir(parents=True)
    for fname, rows in files.items():
        with open(out / fname, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    (out / "evaluation_report.json").write_text(json.dumps(report, indent=2, default=float), encoding="utf-8")
    pv = report["pooled_over_stations"]["all/voxel_weighted"]
    print(f"pooled voxel-weighted: mIoU {pv['miou']:.4f}, OA {pv['overall_accuracy']:.4f}, coverage {pv['prediction_coverage']:.4f}")
    if elements:
        print("elements:", json.dumps(report["element_summary"]))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
