"""Run the OpenScan2BIM pipeline for one run configuration.

Usage (from the workspace root C:/Disk_C/Grounded-Segment-Anything):
    D:/Anaconda/python pipeline/run_pipeline.py --config pipeline/config/R00_reproduce_phase1.json

Every stage is a separate process started from the workspace root, so all paths
in the configuration and in the stage outputs are relative to that root. Each
stage writes runs/<run_id>/<stage>/ (the stage script name), its log to
runs/<run_id>/logs/<stage>.log, and a `_stage_done.json` marker. The run
directory must not exist, unless --resume is given: then stages with a marker
are skipped, s08 continues from its per-record checkpoints, and any other stage
whose output exists without a marker stops the run for manual inspection.
run_manifest.json records the configuration, the pipeline git commit, the
SHA-256 of every pipeline source file, the interpreters with their package
versions, and every command line with its timing.
"""
import argparse
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

PIPELINE = Path(__file__).resolve().parent
ROOT = PIPELINE.parent
VERSION_PROBE = ("import importlib, json, sys; out = {'python': sys.version.split()[0]}\n"
                 "for m in ('numpy', 'scipy', 'pandas', 'torch', 'transformers', 'cv2', 'plyfile', 'pye57'):\n"
                 "    try: out[m] = str(getattr(importlib.import_module(m), '__version__', 'present'))\n"
                 "    except Exception: out[m] = None\n"
                 "print(json.dumps(out))")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--config", required=True)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--stop_after", default=None, help="stage name after which to stop")
    p.add_argument("--allow_dirty", action="store_true", help="run with uncommitted pipeline changes")
    return p.parse_args()


def stages(cfg, run):
    """(name, python key, per-station argument lists or one argument list)."""
    st, prm = cfg["stations"], cfg["params"]
    stage0, e57, gt = cfg["stage0_dir"], cfg["e57_dir"], cfg["gt"]
    d = {name: f"{run}/{name}" for name in (
        "s01_atomic_patches", "s02_door_cuts", "s03_local_surfaces", "s04a_station_references",
        "s04b_region_references", "s05a_association", "s05b_association_qa", "s05c_element_records",
        "s06a_station_projection", "s06b_rectilinear_views", "s07_area_metrics", "s08_qwen_scoring",
        "s09_final", "s10_point_labels", "s11_evaluate", "s11b_error_attribution")}
    region_refs = f"{d['s04b_region_references']}/region_references.json"
    return [
        ("s01_atomic_patches", "cpu", [["--ply", f"{stage0}/{s}/{s}_pano_labeled.ply", "--e57", f"{e57}/{s}.e57",
                                        "--output_dir", d["s01_atomic_patches"]] for s in st]),
        ("s02_door_cuts", "cpu", [["--stations", *st, "--door_cut_stations", *prm["door_cut_stations"],
                                   "--door_guard", prm["door_cut_guard"],
                                   "--pano_dir", stage0, "--atomic_dir", d["s01_atomic_patches"],
                                   "--output_dir", d["s02_door_cuts"]]]),
        ("s03_local_surfaces", "cpu", [["--station", s, "--pano_dir", stage0, "--primitives_dir", d["s02_door_cuts"],
                                        "--output_dir", d["s03_local_surfaces"]] for s in st]),
        ("s04a_station_references", "cpu", [["--source_dir", d["s03_local_surfaces"], "--stations", *st,
                                             "--clearance_m", str(prm["clearance_m"]),
                                             "--output_dir", d["s04a_station_references"]]]),
        ("s04b_region_references", "cpu", [["--source_dir", d["s03_local_surfaces"], "--region", cfg["region"],
                                            "--stations", *st, "--clearance_m", str(prm["clearance_m"]),
                                            "--output_dir", d["s04b_region_references"]]]),
        ("s05a_association", "cpu", [["--stations", *st, "--pano_dir", stage0, "--atomic_dir", d["s02_door_cuts"],
                                      "--surface_dir", d["s03_local_surfaces"], "--output_dir", d["s05a_association"]]]),
        ("s05b_association_qa", "cpu", [["--stations", *st, "--association_dir", d["s05a_association"],
                                         "--output_dir", d["s05b_association_qa"]]]),
        ("s05c_element_records", "cpu", [["--stations", *st, "--fusion_dir", d["s05a_association"],
                                          "--qa_dir", d["s05b_association_qa"], "--surface_dir", d["s03_local_surfaces"],
                                          "--pano_dir", stage0, "--output_dir", d["s05c_element_records"]]]),
        ("s06a_station_projection", "cpu", [["--record_dir", d["s05c_element_records"], "--atomic_dir", d["s02_door_cuts"],
                                             "--e57_dir", e57, "--output_dir", d["s06a_station_projection"]]]),
        ("s06b_rectilinear_views", "cpu", [["--record_dir", d["s05c_element_records"], "--atomic_dir", d["s02_door_cuts"],
                                            "--reference_dir", d["s04a_station_references"],
                                            "--source_projection_dir", d["s06a_station_projection"], "--e57_dir", e57,
                                            "--stations", *st, "--output_dir", d["s06b_rectilinear_views"]]]),
        ("s07_area_metrics", "cpu", [["--record_dir", d["s05c_element_records"], "--atomic_dir", d["s02_door_cuts"],
                                      "--stations", *st, "--output_dir", d["s07_area_metrics"]]]),
        ("s08_qwen_scoring", "qwen", [["--projection_dir", d["s06b_rectilinear_views"],
                                       "--element_record_dir", d["s05c_element_records"],
                                       "--area_dir", d["s07_area_metrics"], "--model_id", prm["qwen_model_id"],
                                       "--max_image_px", str(prm["qwen_max_image_px"]),
                                       "--option_orders", prm["qwen_option_orders"],
                                       "--output_dir", d["s08_qwen_scoring"]]]),
        ("s09_final", "cpu", [["--score_dir", d["s08_qwen_scoring"], "--region_refs", region_refs,
                               "--config", json.dumps(prm["final_config"]), "--output_dir", d["s09_final"]]]),
        ("s10_point_labels", "cpu", [["--region_refs", region_refs, "--atomic_dir", d["s02_door_cuts"],
                                      "--member_csv", f"{d['s05c_element_records']}/local_surface_member_records.csv",
                                      "--final_csv", f"{d['s09_final']}/element_semantic_final.csv",
                                      "--pano_dir", stage0, "--floor_band_m", str(prm["floor_band_m"]),
                                      "--propagation_m", str(prm["propagation_m"]), "--output_dir", d["s10_point_labels"]]]),
        ("s11_evaluate", "cpu", [["--gt_dir", gt["labels_dir"], "--cloud_dir", gt["cloud_dir"], "--pano_dir", stage0,
                                  "--pred_dir", d["s10_point_labels"], "--output_dir", d["s11_evaluate"]]]),
        ("s11b_error_attribution", "cpu", [["--pred_dir", d["s10_point_labels"], "--eval_dir", d["s11_evaluate"],
                                            "--score_csv", f"{d['s08_qwen_scoring']}/station_image_semantic_scores.csv",
                                            "--gt_dir", gt["labels_dir"], "--cloud_dir", gt["cloud_dir"], "--pano_dir", stage0,
                                            "--output_dir", d["s11b_error_attribution"]]]),
    ]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def git(*args):
    return subprocess.run(["git", "-C", str(PIPELINE), *args], capture_output=True, text=True).stdout.strip()


def run_stage(name, python, arg_lists, log_path):
    env = dict(os.environ, KMP_DUPLICATE_LIB_OK="TRUE", PYTHONIOENCODING="utf-8", HF_HUB_OFFLINE="1")
    records = []
    with open(log_path, "a", encoding="utf-8") as log:
        for args in arg_lists:
            cmd = [python, f"pipeline/{name}.py", *args]
            log.write(f"\n$ {subprocess.list2cmdline(cmd)}\n")
            log.flush()
            start = time.time()
            proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                    text=True, encoding="utf-8", errors="replace")
            for line in proc.stdout:
                log.write(line)
                log.flush()
            code = proc.wait()
            records.append({"command": cmd, "seconds": round(time.time() - start, 1), "returncode": code})
            if code != 0:
                raise RuntimeError(f"{name} failed with exit code {code}; see {log_path}")
    return records


def main():
    args = parse_args()
    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    run = f"runs/{cfg['run_id']}"
    run_dir = ROOT / run
    dirty = git("status", "--porcelain")
    if dirty and not args.allow_dirty:
        raise SystemExit(f"pipeline/ has uncommitted changes; commit first or pass --allow_dirty:\n{dirty}")
    if run_dir.exists() and not args.resume:
        raise FileExistsError(f"Refusing to reuse existing run directory: {run_dir}")
    (run_dir / "logs").mkdir(parents=True, exist_ok=True)

    manifest_path = run_dir / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {
        "run_id": cfg["run_id"], "config": cfg, "stages": {}, "sessions": []}
    interpreters = {key: path for key, path in cfg["python"].items()}
    manifest["sessions"].append({
        "started": time.strftime("%Y-%m-%d %H:%M:%S"),
        "pipeline_commit": git("rev-parse", "HEAD"), "pipeline_dirty": bool(dirty),
        "source_sha256": {str(p.relative_to(PIPELINE)).replace(os.sep, "/"): sha256(p)
                          for p in sorted(PIPELINE.rglob("*.py"))},
        "interpreters": {key: {"path": path, "versions": json.loads(subprocess.run(
            [path, "-c", VERSION_PROBE], capture_output=True, text=True).stdout or "{}")}
            for key, path in interpreters.items()},
    })
    (run_dir / "run_config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")

    for name, python_key, arg_lists in stages(cfg, run):
        out = run_dir / name
        marker = out / "_stage_done.json"
        if marker.exists():
            print(f"[skip] {name} (done)", flush=True)
        else:
            if out.exists() and name != "s08_qwen_scoring":
                raise SystemExit(f"{out} exists without a done marker; inspect or remove it before resuming")
            print(f"[run ] {name} ...", flush=True)
            started = time.strftime("%Y-%m-%d %H:%M:%S")
            records = run_stage(name, interpreters[python_key], arg_lists, run_dir / "logs" / f"{name}.log")
            seconds = sum(r["seconds"] for r in records)
            marker.write_text(json.dumps({"stage": name, "started": started, "seconds": seconds}, indent=2),
                              encoding="utf-8")
            manifest["stages"][name] = {"started": started, "seconds": seconds, "runs": records}
            manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
            print(f"[done] {name} in {seconds:.0f} s", flush=True)
        if name == args.stop_after:
            break
    manifest["sessions"][-1]["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
