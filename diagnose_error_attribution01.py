"""One-off dev diagnostic: where do door/column (and other class) errors come from?

FN of GT class c (voxel-weighted), each point in exactly one bucket:
  residual      point not in any element (Layer 1.1 coverage; label from Layer 3 rules)
  mixed_elem    point in an element whose GT majority != c (Layer 1 boundary)
  class_elem    point in an element whose GT majority == c but element class != c (Layer 2)
  upper_split   element class == c, point relabelled ceiling above z_u (Layer 3)
FP of predicted class c (GT != c), analogous buckets.
"""
import sys
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(r"C:\Disk_C\Grounded-Segment-Anything")
sys.path.insert(0, str(ROOT))
import evaluate_semantic_point_gt02 as ev  # noqa: E402

PRED = ROOT / "outputs246_step06_point_labels01_final07_A_joint_door_station5_6_7"
GT = ROOT / "outputs235_semantic_point_gt02_labels_station5_6_7"
CLOUD = ROOT / "outputs231_semantic_point_gt01_station5_6_7"
PUR = ROOT / "outputs248_semantic_point_eval02_final07_A_joint_door_station5_6_7/element_purity.csv"
SCORES = ROOT / "outputs224_step05c_door_station5_6_7/station_image_semantic_scores.csv"
CODE = {c: i for i, c in enumerate(ev.CLASSES, start=1)}
NAME = {i: c for c, i in CODE.items()}

manifest = json.loads((CLOUD / "manifest.json").read_text(encoding="utf-8"))
rep_keys = np.load(CLOUD / "rep_voxel_keys.npy")
pur = pd.read_csv(PUR)
sc = pd.read_csv(SCORES)
sc["img"] = sc["rectilinear_image_top1"].replace({"other": "furniture"})
sc["joint"] = sc["joint_top1"].replace({"other": "furniture"})
elem_cls = pd.read_csv(PRED / "element_classes.csv").set_index("element_id")["class_code"]

fn_rows, fp_rows, elem_rows = [], [], []
for s in manifest["stations"]:
    eid = np.load(PRED / f"{s}_element_id.npy").astype(np.int32)
    pred = np.load(PRED / f"{s}_pred_class.npy")
    src = np.load(PRED / f"{s}_label_source.npy")
    gt = ev.GT_LUT[np.load(GT / f"{s}_gt_fine.npy")]
    pos = ev.station_index(s, manifest, rep_keys)
    w = 1.0 / np.bincount(pos, minlength=len(rep_keys))[pos]
    k = gt > 0
    eid, pred, src, gt, w = eid[k], pred[k], src[k], gt[k], w[k]
    p = pur[pur.station == s].set_index("element_id")
    maj = np.zeros(eid.max() + 2, np.uint8)
    maj[p.index.values] = [CODE[m] for m in p["gt_majority"]]
    ecls = np.zeros(eid.max() + 2, np.uint8)
    ecls[elem_cls.index.values[elem_cls.index.values <= eid.max()]] = elem_cls.values[elem_cls.index.values <= eid.max()]
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
            elem_rows.append({"cls": NAME[c], "station": s, "element": int(e), "gt_c_vox": round(gw), "pred_c_vox": round(pw),
                              "elem_vox": round(w[me].sum()), "gt_majority": NAME.get(int(maj[e]), "?"),
                              "purity": p.loc[e, "purity"] if e in p.index else None,
                              "final": NAME.get(int(ecls[e]), "?"),
                              "qwen_image_top1": q["img"].iloc[0] if len(q) else "",
                              "qwen_joint_top1": q["joint"].iloc[0] if len(q) else ""})
    print("done", s, flush=True)

pd.set_option("display.width", 250)
fn = pd.DataFrame(fn_rows).groupby("cls").sum(numeric_only=True)
fn["fn_total"] = fn[["residual", "mixed_elem", "class_elem", "upper_split"]].sum(axis=1)
print("\nFN attribution (share of GT class support, voxel-weighted)")
print((fn[["residual", "mixed_elem", "class_elem", "upper_split", "fn_total"]].div(fn["support"], axis=0)).round(3))
fp = pd.DataFrame(fp_rows).groupby("cls").sum(numeric_only=True)
print("\nFP attribution (share of GT class support, voxel-weighted)")
print((fp.div(fn["support"], axis=0)).round(3))
print("\nFN by station for door/column (share of station support)")
fs = pd.DataFrame(fn_rows)
fs = fs[fs.cls.isin(["door", "column"])].set_index(["cls", "station"])
print(fs.assign(**{c: (fs[c] / fs["support"]).round(3) for c in ["residual", "mixed_elem", "class_elem", "upper_split"]}).round(1))
print("\nElements touching door/column")
print(pd.DataFrame(elem_rows).sort_values(["cls", "gt_c_vox"], ascending=[True, False]).to_string(index=False))
