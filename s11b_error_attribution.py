"""s11b -- where the point errors come from (dev diagnostic, Framework v0.1 Section 8).

FN of GT class c (voxel-weighted); each missed point falls in exactly one bucket:
  residual      point not in any element (Layer 1.1 coverage; label from Layer 3 rules)
  mixed_elem    point in an element whose GT majority != c (Layer 1 boundary)
  class_elem    point in an element whose GT majority == c but element class != c (Layer 2)
  upper_split   element class == c, point relabelled ceiling above z_u (Layer 3)
FP of predicted class c (GT != c): residual, minority points of a c-majority
element, or points of an element predicted c whose majority is not c.
The element table lists every element touching GT or predicted door/column,
with the Qwen Stage-1 (image) and Stage-2 (joint) top-1 of its station record.

Pipeline version of tools/diagnose_error_attribution01.py (2026-10-30): inputs
are arguments and the tables are written as CSV.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from s11_evaluate import CLASSES, GT_LUT, station_index

CODE = {c: i for i, c in enumerate(CLASSES, start=1)}
NAME = {i: c for c, i in CODE.items()}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--pred_dir", required=True, help="s10 output")
    p.add_argument("--eval_dir", required=True, help="s11 output (element_purity.csv)")
    p.add_argument("--score_csv", required=True, help="s08 station_image_semantic_scores.csv")
    p.add_argument("--gt_dir", required=True)
    p.add_argument("--cloud_dir", required=True)
    p.add_argument("--output_dir", required=True)
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    out = Path(args.output_dir)
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite existing output directory: {out}")
    pred_dir, gt_dir, cloud = Path(args.pred_dir), Path(args.gt_dir), Path(args.cloud_dir)
    manifest = json.loads((cloud / "manifest.json").read_text(encoding="utf-8"))
    rep_keys = np.load(cloud / "rep_voxel_keys.npy")
    pur = pd.read_csv(Path(args.eval_dir) / "element_purity.csv")
    sc = pd.read_csv(args.score_csv)
    sc["img"] = sc["rectilinear_image_top1"].replace({"other": "furniture"})
    sc["joint"] = sc["joint_top1"].replace({"other": "furniture"})
    elem_cls = pd.read_csv(pred_dir / "element_classes.csv").set_index("element_id")["class_code"]

    fn_rows, fp_rows, elem_rows = [], [], []
    for s in manifest["stations"]:
        eid = np.load(pred_dir / f"{s}_element_id.npy").astype(np.int32)
        pred = np.load(pred_dir / f"{s}_pred_class.npy")
        gt = GT_LUT[np.load(gt_dir / f"{s}_gt_fine.npy")]
        pos = station_index(s, manifest, rep_keys)
        w = 1.0 / np.bincount(pos, minlength=len(rep_keys))[pos]
        k = gt > 0
        eid, pred, gt, w = eid[k], pred[k], gt[k], w[k]
        p = pur[pur.station == s].set_index("element_id")
        maj = np.zeros(eid.max() + 2, np.uint8)
        maj[p.index.values] = [CODE[m] for m in p["gt_majority"]]
        ecls = np.zeros(eid.max() + 2, np.uint8)
        known = elem_cls.index.values <= eid.max()
        ecls[elem_cls.index.values[known]] = elem_cls.values[known]
        in_e = eid >= 0
        m = np.where(in_e, maj[np.maximum(eid, 0)], 0)
        ec = np.where(in_e, ecls[np.maximum(eid, 0)], 0)
        for c in range(1, 8):
            g = (gt == c) & (pred != c)
            fn_rows.append({"station": s, "cls": NAME[c], "support": w[gt == c].sum(),
                            "residual": w[g & ~in_e].sum(),
                            "mixed_elem": w[g & in_e & (m != c)].sum(),
                            "class_elem": w[g & in_e & (m == c) & (ec != c)].sum(),
                            "upper_split": w[g & in_e & (m == c) & (ec == c)].sum()})
            f = (pred == c) & (gt != c)
            fp_rows.append({"station": s, "cls": NAME[c],
                            "residual": w[f & ~in_e].sum(),
                            "minority_in_c_elem": w[f & in_e & (m == c)].sum(),
                            "wrong_class_elem": w[f & in_e & (m != c)].sum()})
        for c in (CODE["door"], CODE["column"]):
            sel = (gt == c) | (pred == c)
            for e in np.unique(eid[sel & in_e]):
                me = eid == e
                gw = w[me & (gt == c)].sum()
                pw = w[me & (pred == c)].sum()
                if max(gw, pw) < 50:
                    continue
                q = sc[(sc.station_id == s) & (sc.provisional_geometric_element_id == e)]
                elem_rows.append({"cls": NAME[c], "station": s, "element": int(e), "gt_c_vox": round(gw),
                                  "pred_c_vox": round(pw), "elem_vox": round(w[me].sum()),
                                  "gt_majority": NAME.get(int(maj[e]), "?"),
                                  "purity": p.loc[e, "purity"] if e in p.index else None,
                                  "final": NAME.get(int(ecls[e]), "?"),
                                  "qwen_image_top1": q["img"].iloc[0] if len(q) else "",
                                  "qwen_joint_top1": q["joint"].iloc[0] if len(q) else ""})
        print("done", s, flush=True)

    out.mkdir(parents=True)
    fn_station = pd.DataFrame(fn_rows)
    fn = fn_station.groupby("cls").sum(numeric_only=True)
    fn["fn_total"] = fn[["residual", "mixed_elem", "class_elem", "upper_split"]].sum(axis=1)
    fn_share = fn[["residual", "mixed_elem", "class_elem", "upper_split", "fn_total"]].div(fn["support"], axis=0)
    fp_share = pd.DataFrame(fp_rows).groupby("cls").sum(numeric_only=True).div(fn["support"], axis=0)
    fn_share.round(4).to_csv(out / "fn_attribution_pooled.csv")
    fp_share.round(4).to_csv(out / "fp_attribution_pooled.csv")
    fn_station.round(1).to_csv(out / "fn_attribution_by_station_voxels.csv", index=False)
    elements = pd.DataFrame(elem_rows).sort_values(["cls", "gt_c_vox"], ascending=[True, False])
    elements.to_csv(out / "door_column_elements.csv", index=False)
    pd.set_option("display.width", 250)
    print("\nFN attribution (share of GT class support, voxel-weighted)")
    print(fn_share.round(3))
    print("\nFP attribution (share of GT class support, voxel-weighted)")
    print(fp_share.round(3))


if __name__ == "__main__":
    main()
