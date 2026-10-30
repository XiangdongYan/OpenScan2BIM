"""Compare run R00 stage by stage with the legacy Phase-1 outputs (clean-run plan, step D).

Acceptance (plan Section 3 D):
  L1 geometry (s01-s05c)     identical arrays and tables (path and provenance columns excluded)
  L2 views (s06a, s06b, s07)  identical tables; rendered images byte-identical
  L2 Qwen (s08)               reported, not required: top-1 agreement, probability differences,
                              and whether each differing legacy record was reused from older runs
  Final, labels, metrics      reported: element classes, point labels, mIoU / OA

Writes regression_report.json and regression_report.md into --output_dir.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

STATIONS = ["2ndLab-5", "2ndLab-6", "2ndLab-7"]
LEGACY = {
    "s01_atomic_patches": "outputs49_step01b_atomic04_7station",
    "s02_door_cuts": "outputs216_door_integrated_inputs/atomic",
    "s03_local_surfaces": "outputs216_door_integrated_inputs/surfaces",
    "s04a_station_references": "outputs220_step01c_elevation_door_station5_6_7",
    "s04b_region_references": "outputs237_step01c_region_references01_level02lab",
    "s05a_association": "outputs217_fusion09_door_station5_6_7",
    "s05b_association_qa": "outputs218_fusion07_door_station5_6_7",
    "s05c_element_records": "outputs219_step04b_door_station5_6_7",
    "s06a_station_projection": "outputs221_step04d_pano_door_station5_6_7",
    "s06b_rectilinear_views": "outputs222_step04d_rectilinear_door_station5_6_7",
    "s07_area_metrics": "outputs223_step04e_area_door_station5_6_7",
    "s08_qwen_scoring": "outputs224_step05c_door_station5_6_7",
    "s09_final": "outputs244_step05d_final07_A_joint_door_station5_6_7",
    "s10_point_labels": "outputs246_step06_point_labels01_final07_A_joint_door_station5_6_7",
    "s11_evaluate": "outputs248_semantic_point_eval02_final07_A_joint_door_station5_6_7",
}
SKIP_COLUMN_WORDS = ("path", "source_file", "_dir", "sha256", "record_version", "provenance")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--run_dir", required=True)
    p.add_argument("--legacy_root", default=".", help="workspace root holding the legacy outputs")
    p.add_argument("--output_dir", required=True)
    return p.parse_args()


def compare_tables(new_path, old_path, keys=None):
    new, old = pd.read_csv(new_path, low_memory=False), pd.read_csv(old_path, low_memory=False)
    if keys:
        new, old = new.sort_values(keys).reset_index(drop=True), old.sort_values(keys).reset_index(drop=True)
    skip = [c for c in set(new.columns) | set(old.columns) if any(w in c for w in SKIP_COLUMN_WORDS)]
    common = [c for c in old.columns if c in new.columns and c not in skip]
    result = {"rows_new": len(new), "rows_old": len(old),
              "only_new_columns": sorted(set(new.columns) - set(old.columns)),
              "only_old_columns": sorted(set(old.columns) - set(new.columns)),
              "skipped_columns": sorted(skip), "differing_columns": {}}
    if len(new) != len(old):
        result["identical"] = False
        return result
    for c in common:
        a, b = new[c], old[c]
        if pd.api.types.is_numeric_dtype(a) and pd.api.types.is_numeric_dtype(b):
            a, b = a.to_numpy(float), b.to_numpy(float)
            diff = np.abs(a - b)
            bad = ~((a == b) | (np.isnan(a) & np.isnan(b)))
            if bad.any():
                result["differing_columns"][c] = {"n": int(bad.sum()), "max_abs": float(np.nanmax(diff[bad]))}
        else:
            bad = ~((a.astype(str) == b.astype(str)) | (a.isna() & b.isna()))
            if bad.any():
                result["differing_columns"][c] = {"n": int(bad.sum()), "example_new": str(a[bad].iloc[0])[:80],
                                                  "example_old": str(b[bad].iloc[0])[:80]}
    result["identical"] = not result["differing_columns"]
    return result


def compare_npy(new_path, old_path):
    a, b = np.load(new_path), np.load(old_path)
    same = a.shape == b.shape and np.array_equal(a, b)
    return {"identical": bool(same), "shape": list(a.shape),
            "n_differ": None if a.shape != b.shape else int((a != b).sum())}


def compare_npz(new_path, old_path):
    a, b = np.load(new_path), np.load(old_path)
    out = {k: bool(k in b and a[k].shape == b[k].shape and np.array_equal(a[k], b[k])) for k in a.files}
    return {"identical": all(out.values()) and set(a.files) == set(b.files), "arrays": out}


def compare_images(new_dir, old_dir):
    digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    new = {p.name: digest(p) for p in Path(new_dir).iterdir() if p.is_file()}
    old = {p.name: digest(p) for p in Path(old_dir).iterdir() if p.is_file()}
    differ = sorted(n for n in set(new) & set(old) if new[n] != old[n])
    return {"identical": not differ and set(new) == set(old), "n_new": len(new), "n_old": len(old),
            "n_differ": len(differ), "missing_in_new": sorted(set(old) - set(new))[:10],
            "extra_in_new": sorted(set(new) - set(old))[:10], "examples": differ[:10]}


def main():
    args = parse_args()
    run, root = Path(args.run_dir), Path(args.legacy_root)
    out = Path(args.output_dir)
    if out.exists():
        raise FileExistsError(out)
    new = lambda stage, name="": run / stage / name
    old = lambda stage, name="": root / LEGACY[stage] / name
    r = {}

    for s in STATIONS:
        r[f"s01 {s} patch_ids"] = compare_npy(new("s01_atomic_patches", f"{s}_patch_ids.npy"), old("s01_atomic_patches", f"{s}_patch_ids.npy"))
        r[f"s01 {s} patches"] = compare_tables(new("s01_atomic_patches", f"{s}_patches.csv"), old("s01_atomic_patches", f"{s}_patches.csv"))
        r[f"s01 {s} point_quality"] = compare_npz(new("s01_atomic_patches", f"{s}_point_quality.npz"), old("s01_atomic_patches", f"{s}_point_quality.npz"))
        r[f"s02 {s} patch_ids"] = compare_npy(new("s02_door_cuts", f"{s}_patch_ids.npy"), old("s02_door_cuts", f"{s}_patch_ids.npy"))
        r[f"s02 {s} patches"] = compare_tables(new("s02_door_cuts", f"{s}_patches.csv"), old("s02_door_cuts", f"{s}_patches.csv"))
        for t in ("patch_to_surface", "surfaces", "elevation_reference_bands"):
            r[f"s03 {s} {t}"] = compare_tables(new("s03_local_surfaces", f"{s}_{t}.csv"), old("s03_local_surfaces", f"{s}_{t}.csv"))
        for t in ("surfaces", "elevation_reference_bands"):
            r[f"s04a {s} {t}"] = compare_tables(new("s04a_station_references", f"{s}_{t}.csv"), old("s04a_station_references", f"{s}_{t}.csv"))
    nb = json.loads(new("s04b_region_references", "region_references.json").read_text(encoding="utf-8"))
    ob = json.loads(old("s04b_region_references", "region_references.json").read_text(encoding="utf-8"))
    keys = ("floor_reference_z_m", "upper_zone_min_z_m", "upper_reference_z_m", "station_floor_reference_z_m")
    r["s04b region_references"] = {"identical": all(nb[k] == ob[k] for k in keys), **{k: [nb[k], ob[k]] for k in keys}}
    r["s04b bands"] = compare_tables(new("s04b_region_references", "region_elevation_reference_bands.csv"), old("s04b_region_references", "region_elevation_reference_bands.csv"))
    for t in ("surface_element_map_theta_07deg", "surface_association_candidates", "normal_angle_sensitivity"):
        r[f"s05a {t}"] = compare_tables(new("s05a_association", f"{t}.csv"), old("s05a_association", f"{t}.csv"))
    for t in ("provisional_element_geometry_qa", "local_surface_component_geometry_qa", "component_geometry_summary"):
        r[f"s05b {t}"] = compare_tables(new("s05b_association_qa", f"{t}.csv"), old("s05b_association_qa", f"{t}.csv"))
    for t in ("provisional_geometric_element_records", "local_surface_member_records", "element_view_slots"):
        r[f"s05c {t}"] = compare_tables(new("s05c_element_records", f"{t}.csv"), old("s05c_element_records", f"{t}.csv"))

    r["s06a regions"] = compare_tables(new("s06a_station_projection", "station_element_evidence_regions.csv"), old("s06a_station_projection", "station_element_evidence_regions.csv"))
    for d in ("context_crops", "context_point_masks"):
        r[f"s06a {d}"] = compare_images(new("s06a_station_projection", d), old("s06a_station_projection", d))
    r["s06b evidence"] = compare_tables(new("s06b_rectilinear_views", "rectilinear_metric_evidence.csv"), old("s06b_rectilinear_views", "rectilinear_metric_evidence.csv"),
                                        keys=["provisional_geometric_element_id", "station_id"])
    for d in ("rectilinear_clean", "rectilinear_overlays"):
        r[f"s06b {d}"] = compare_images(new("s06b_rectilinear_views", d), old("s06b_rectilinear_views", d))
    for t in ("station_element_area_metrics", "element_area_metrics"):
        r[f"s07 {t}"] = compare_tables(new("s07_area_metrics", f"{t}.csv"), old("s07_area_metrics", f"{t}.csv"))

    key = ["provisional_geometric_element_id", "station_id"]
    qn = pd.read_csv(new("s08_qwen_scoring", "station_image_semantic_scores.csv")).sort_values(key).reset_index(drop=True)
    qo = pd.read_csv(old("s08_qwen_scoring", "station_image_semantic_scores.csv")).sort_values(key).reset_index(drop=True)
    classes = ["wall", "floor", "ceiling", "column", "door", "window", "furniture", "other"]
    q = {"identical": False, "n": len(qn), "keys_match": bool((qn[key].to_numpy() == qo[key].to_numpy()).all())}
    for prefix in ("rectilinear_image", "joint"):
        pn, po = qn[[f"{prefix}_p_{c}" for c in classes]].to_numpy(float), qo[[f"{prefix}_p_{c}" for c in classes]].to_numpy(float)
        top_differ = qn[f"{prefix}_top1"] != qo[f"{prefix}_top1"]
        q[prefix] = {"top1_agreement": float(1 - top_differ.mean()), "n_top1_differ": int(top_differ.sum()),
                     "max_abs_p_diff": float(np.abs(pn - po).max()), "median_max_abs_p_diff": float(np.median(np.abs(pn - po).max(axis=1)))}
    q["card_identical"] = int((qn["metric_geometry_card"] == qo["metric_geometry_card"]).sum())
    q["option_order_identical"] = int((qn["qwen_option_order"] == qo["qwen_option_order"]).sum())
    differ = (qn["joint_top1"] != qo["joint_top1"]) | (qn["rectilinear_image_top1"] != qo["rectilinear_image_top1"])
    q["legacy_status_of_differing_records"] = qo.loc[differ, "extension_inference_status"].value_counts().to_dict()
    q["legacy_status_all"] = qo["extension_inference_status"].value_counts().to_dict()
    q["differing_records"] = [
        {"element": int(a.provisional_geometric_element_id), "station": a.station_id,
         "image_new": a.rectilinear_image_top1, "image_old": b.rectilinear_image_top1,
         "joint_new": a.joint_top1, "joint_old": b.joint_top1, "legacy_status": b.extension_inference_status,
         "card_same": a.metric_geometry_card == b.metric_geometry_card}
        for (_, a), (_, b) in zip(qn[differ].iterrows(), qo[differ].iterrows())]
    q["identical"] = not differ.any()
    r["s08 qwen"] = q

    fn = pd.read_csv(new("s09_final", "element_semantic_final.csv")).set_index("provisional_geometric_element_id")["final_semantic_class"]
    fo = pd.read_csv(old("s09_final", "element_semantic_final.csv")).set_index("provisional_geometric_element_id")["final_semantic_class"]
    changed = fn[fn != fo.reindex(fn.index)]
    r["s09 final"] = {"identical": changed.empty, "n_elements": len(fn), "n_changed": len(changed),
                      "changed": {int(e): [c, fo.get(e)] for e, c in changed.items()}}
    agree = {}
    for s in STATIONS:
        a = np.load(new("s10_point_labels", f"{s}_pred_class.npy"))
        b = np.load(old("s10_point_labels", f"{s}_pred_class.npy"))
        agree[s] = float((a == b).mean())
    r["s10 point labels"] = {"identical": all(v == 1.0 for v in agree.values()), "agreement": agree}
    mn = pd.read_csv(new("s11_evaluate", "metrics_summary.csv"))
    mo = pd.read_csv(old("s11_evaluate", "metrics_summary.csv"))
    pick = lambda m: m[(m.scope == "pooled") & (m.subset == "all") & (m.weighting == "voxel_weighted")].iloc[0]
    a, b = pick(mn), pick(mo)
    r["s11 metrics"] = {"identical": bool(abs(a.miou - b.miou) < 1e-9 and abs(a.overall_accuracy - b.overall_accuracy) < 1e-9),
                        "miou": [float(a.miou), float(b.miou)], "oa": [float(a.overall_accuracy), float(b.overall_accuracy)],
                        "per_class_iou_new_old": {c: [float(a[f"iou_{c}"]), float(b[f"iou_{c}"])]
                                                  for c in ["ceiling", "wall", "door", "window", "floor", "column", "furniture"]
                                                  if f"iou_{c}" in a.index}}

    out.mkdir(parents=True)
    (out / "regression_report.json").write_text(json.dumps(r, indent=2, default=str), encoding="utf-8")
    lines = ["# R00 regression against Phase 1", "", "| Check | Identical | Note |", "|---|---|---|"]
    for k, v in r.items():
        note = ""
        if v.get("differing_columns"):
            note = "; ".join(f"{c}: {d['n']}" for c, d in list(v["differing_columns"].items())[:4])
        if v.get("only_new_columns") or v.get("only_old_columns"):
            note += f" new-only cols {v.get('only_new_columns')}, old-only {v.get('only_old_columns')}"
        if k == "s08 qwen":
            note = (f"image top1 agree {v['rectilinear_image']['top1_agreement']:.3f}, joint {v['joint']['top1_agreement']:.3f}; "
                    f"cards identical {v['card_identical']}/{v['n']}")
        if k == "s11 metrics":
            note = f"mIoU new/old {v['miou'][0]:.4f}/{v['miou'][1]:.4f}, OA {v['oa'][0]:.4f}/{v['oa'][1]:.4f}"
        if k == "s09 final":
            note = f"{v['n_changed']} of {v['n_elements']} elements changed"
        lines.append(f"| {k} | {'yes' if v['identical'] else '**no**'} | {note} |")
    (out / "regression_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
