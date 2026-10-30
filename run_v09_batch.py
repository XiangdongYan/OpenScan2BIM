"""
Batch driver for Segmentation09 (M4): 7-station refresh + EXP-A matrix.

Phase 1 — station refresh: all 7 stations, default config (yaw_k=4,
          τ=0.29) → ./outputs15/<station>/          (~13 min each)
Phase 2 — EXP-A configuration-sensitivity matrix on 2ndLab-7:
          yaw_k ∈ {1, 4} × τ ∈ {0.25, 0.27, 0.29, 0.31, 0.33}
          → ./outputs15/expA/K<k>_tau<τ>/           (~8–13 min each)
          Seeds are fixed (v07+), so identical-config reruns are
          byte-identical — EXP-A variance IS configuration sensitivity.

Resumable: a combination whose <scan>_run_manifest.json already exists is
skipped.  At the end, every manifest found is summarised into
./outputs15/expA_summary.csv (instances, label histogram, persistence).

Usage:  python run_v09_batch.py            (run everything pending)
        python run_v09_batch.py --summary_only
"""
import os
import csv
import json
import glob
import time
import argparse
import subprocess

PY       = r"D:\Anaconda\envs\GroundingDino\python"
SCRIPT   = "OpenScan2BIM_Step01_Segmentation09.py"
OUT_ROOT = "./outputs15"
STATIONS = [f"2ndLab-{i}" for i in range(1, 8)]
EXPA_SCAN = "2ndLab-7"
EXPA_K    = [1, 4]
EXPA_TAU  = [0.25, 0.27, 0.29, 0.31, 0.33]


def log(msg: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def run_one(e57: str, out_dir: str, yaw_k: int = 4,
            tau: float = 0.29) -> None:
    scan = os.path.splitext(os.path.basename(e57))[0]
    manifest = os.path.join(out_dir, f"{scan}_run_manifest.json")
    if os.path.exists(manifest):
        log(f"skip (manifest exists): {out_dir}")
        return
    os.makedirs(out_dir, exist_ok=True)
    cmd = [PY, SCRIPT, "--input_e57", e57, "--output_dir", out_dir,
           "--yaw_k", str(yaw_k), "--conf_threshold", str(tau)]
    log(f"RUN {scan}  yaw_k={yaw_k}  τ={tau}  → {out_dir}")
    t0 = time.time()
    env = dict(os.environ, KMP_DUPLICATE_LIB_OK="TRUE",
               PYTHONIOENCODING="utf-8")
    r = subprocess.run(cmd, env=env)
    if r.returncode != 0:
        log(f"[ERROR] exit {r.returncode} for {out_dir} — continuing")
    else:
        log(f"done in {time.time() - t0:.0f} s")


def summarize() -> None:
    rows = []
    for mf in sorted(glob.glob(os.path.join(OUT_ROOT, "**",
                                            "*_run_manifest.json"),
                               recursive=True)):
        with open(mf, encoding="utf-8") as fh:
            m = json.load(fh)
        res = m.get("results", {})
        pers = res.get("persistence") or {}
        rows.append({
            "dir": os.path.relpath(os.path.dirname(mf), OUT_ROOT),
            "scan": os.path.basename(mf).replace("_run_manifest.json", ""),
            "yaw_k": m.get("constants", {}).get("YAW_ENSEMBLE_K"),
            "tau": m.get("args", {}).get("conf_threshold"),
            "n_instances": res.get("n_instances"),
            "labels": json.dumps(res.get("label_histogram", {}),
                                 ensure_ascii=False),
            "persistence_mean": pers.get("mean"),
            "persistence_min": pers.get("min"),
            "elapsed_s": m.get("elapsed_s"),
        })
    out = os.path.join(OUT_ROOT, "expA_summary.csv")
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()) if rows
                           else ["dir"])
        w.writeheader(); w.writerows(rows)
    log(f"summary → {out} ({len(rows)} runs)")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--summary_only", action="store_true")
    args = p.parse_args()
    if not args.summary_only:
        for s in STATIONS:                      # Phase 1
            run_one(f"./assets/Level02Lab/{s}.e57", os.path.join(OUT_ROOT, s))
        for k in EXPA_K:                        # Phase 2
            for tau in EXPA_TAU:
                run_one(f"./assets/Level02Lab/{EXPA_SCAN}.e57",
                        os.path.join(OUT_ROOT, "expA",
                                     f"K{k}_tau{tau:.2f}"),
                        yaw_k=k, tau=tau)
    summarize()


if __name__ == "__main__":
    main()
