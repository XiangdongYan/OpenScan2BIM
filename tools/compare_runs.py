"""Compare a run with its parent run (dev): metrics, element classes, Qwen records, error attribution.

Usage (workspace root):
    D:/Anaconda/python pipeline/tools/compare_runs.py --parent runs/R00_reproduce_phase1 --child runs/R01_...

Writes <child>/compare_vs_<parent name>/comparison.md and comparison.json. Element
class changes are listed with the element's GT majority class and purity from the
child's s11 element_purity.csv, so each change can be read as a fix or a regression.
"""
import argparse
import json
from pathlib import Path

import pandas as pd

CLASSES7 = ["ceiling", "wall", "door", "window", "floor", "column", "furniture"]
KEY = ["provisional_geometric_element_id", "station_id"]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--parent", required=True)
    p.add_argument("--child", required=True)
    return p.parse_args()


def pooled(run, weighting):
    m = pd.read_csv(Path(run) / "s11_evaluate" / "metrics_summary.csv")
    row = m[(m.scope == "pooled") & (m.subset == "all") & (m.weighting == weighting)].iloc[0]
    return {"miou": float(row.miou), "oa": float(row.overall_accuracy), **{c: float(row[f"iou_{c}"]) for c in CLASSES7}}


def main():
    args = parse_args()
    parent, child = Path(args.parent), Path(args.child)
    out = child / f"compare_vs_{parent.name}"
    if out.exists():
        raise FileExistsError(out)
    r = {"parent": str(parent), "child": str(child), "metrics": {}}
    for w in ("voxel_weighted", "point_weighted"):
        a, b = pooled(parent, w), pooled(child, w)
        r["metrics"][w] = {k: {"parent": a[k], "child": b[k], "delta": b[k] - a[k]} for k in a}

    fp = pd.read_csv(parent / "s09_final" / "element_semantic_final.csv").set_index("provisional_geometric_element_id")
    fc = pd.read_csv(child / "s09_final" / "element_semantic_final.csv").set_index("provisional_geometric_element_id")
    purity = pd.read_csv(child / "s11_evaluate" / "element_purity.csv")
    changes = []
    if fp.index.equals(fc.index):
        for e in fc.index[fc.final_semantic_class != fp.final_semantic_class]:
            g = purity[purity.element_id == e]
            vox = g.gt_voxel_weight.sum()
            majority = g.groupby("gt_majority").gt_voxel_weight.sum().idxmax() if len(g) else ""
            changes.append({"element": int(e), "parent": fp.loc[e, "final_semantic_class"],
                            "child": fc.loc[e, "final_semantic_class"], "gt_majority": majority,
                            "gt_voxels": round(float(vox)),
                            "purity": round(float((g.purity * g.gt_voxel_weight).sum() / vox), 3) if vox else None})
        r["element_ids_match"] = True
    else:
        r["element_ids_match"] = False  # geometry changed: element ids are not comparable
    r["element_class_changes"] = changes

    qp = pd.read_csv(parent / "s08_qwen_scoring" / "station_image_semantic_scores.csv").set_index(KEY)
    qc = pd.read_csv(child / "s08_qwen_scoring" / "station_image_semantic_scores.csv").set_index(KEY)
    common = qp.index.intersection(qc.index)
    r["qwen"] = {"records_parent": len(qp), "records_child": len(qc), "records_common": len(common)}
    for pre in ("rectilinear_image", "joint"):
        r["qwen"][f"{pre}_top1_changed"] = int((qp.loc[common, f"{pre}_top1"] != qc.loc[common, f"{pre}_top1"]).sum())
        col = f"{pre}_order_agreement"
        if col in qc.columns:
            agr = qc[col].astype(float)
            r["qwen"][f"{pre}_order_agreement"] = {"records_below_1": int((agr < 1).sum()),
                                                   "records_at_or_below_0.5": int((agr <= 0.5).sum()),
                                                   "mean": float(agr.mean())}

    fn_p = pd.read_csv(parent / "s11b_error_attribution" / "fn_attribution_pooled.csv", index_col=0)
    fn_c = pd.read_csv(child / "s11b_error_attribution" / "fn_attribution_pooled.csv", index_col=0)
    r["fn_attribution_delta"] = (fn_c - fn_p).round(4).to_dict(orient="index")

    out.mkdir(parents=True)
    (out / "comparison.json").write_text(json.dumps(r, indent=2, default=str), encoding="utf-8")
    m = r["metrics"]["voxel_weighted"]
    lines = [f"# {child.name} vs {parent.name}", "",
             "Pooled, all points, voxel-weighted:", "",
             "| | " + " | ".join(m) + " |", "|---|" + "---|" * len(m),
             "| parent | " + " | ".join(f"{v['parent']:.4f}" for v in m.values()) + " |",
             "| child | " + " | ".join(f"{v['child']:.4f}" for v in m.values()) + " |",
             "| delta | " + " | ".join(f"{v['delta']:+.4f}" for v in m.values()) + " |", "",
             f"Element class changes ({len(changes)}):", ""]
    if changes:
        lines += ["| element | parent | child | GT majority | GT voxels | purity |", "|---|---|---|---|---|---|"]
        lines += [f"| {c['element']} | {c['parent']} | {c['child']} | {c['gt_majority']} | {c['gt_voxels']} | {c['purity']} |"
                  for c in changes]
    if not r["element_ids_match"]:
        lines.append("Element ids differ between the runs (geometry changed); per-element changes are not listed.")
    lines += ["", f"Qwen: {json.dumps(r['qwen'])}"]
    (out / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
