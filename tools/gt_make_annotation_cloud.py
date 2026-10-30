"""Build a blind, merged multi-station cloud for point-level semantic GT annotation.

Pipeline copy of make_semantic_point_gt_cloud01.py (2026-10-30), which built
gt/dev_station5_6_7/annotation_cloud; arguments are now required.

Stations are merged in their shared registered world frame (the E57 poses that
Step01 already applied) and reduced to one representative point per voxel: the
point with the highest TLS observation weight w = w_range * w_incidence from
Step01b. Every original E57 point falls in exactly one occupied voxel, so a
label painted on the representative transfers to all original points of that
voxel (see transfer_semantic_point_gt01.py).

The cloud deliberately carries no pipeline output (no Grounded-SAM labels, no
ghost/mixed flags, no patch or element ids) so that the annotation stays blind.

Scalar fields in the PLY:
  rep_hi, rep_lo   representative id split into two 16-bit halves, because
                   CloudCompare stores scalar fields as float32 (exact only
                   below 2^24); rep_id = rep_hi * 65536 + rep_lo
  station          station number of the representative point (e.g. 5)
  n_obs_stations   how many of the merged stations have any point within the
                   same 10 cm cell (co-visibility; 1 cm cells undercount it)
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
from plyfile import PlyData

BITS = 21
COVIS_CELL_M = 0.10


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--stations", nargs="+", required=True)
    p.add_argument("--pano_dir", required=True, help="Stage-0 directory")
    p.add_argument("--quality_dir", required=True, help="s01 output (point quality)")
    p.add_argument("--voxel", type=float, default=0.01)
    p.add_argument("--output_dir", required=True)
    return p.parse_args(argv)


def station_number(name):
    return int(name.rsplit("-", 1)[1])


def read_xyz(pano_dir, station):
    v = PlyData.read(Path(pano_dir) / station / f"{station}_pano_labeled.ply", mmap=True)["vertex"].data
    return v, np.column_stack((v["x"], v["y"], v["z"])).astype(np.float64)


def voxel_keys(xyz, origin, voxel):
    ijk = np.floor((xyz - origin) / voxel).astype(np.int64)
    if ijk.min() < 0 or ijk.max() >= (1 << BITS):
        raise ValueError("scene extent exceeds the packed voxel-key range")
    return (ijk[:, 0] << (2 * BITS)) | (ijk[:, 1] << BITS) | ijk[:, 2]


def best_per_voxel(keys, weight):
    order = np.lexsort((-weight.astype(np.int16), keys))
    first = np.ones(len(order), bool)
    first[1:] = keys[order[1:]] != keys[order[:-1]]
    return order[first]


def main(argv=None):
    args = parse_args(argv)
    out = Path(args.output_dir)
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite existing output directory: {out}")

    log("pass 1: shared voxel origin")
    lo = np.full(3, np.inf)
    for s in args.stations:
        _, xyz = read_xyz(args.pano_dir, s)
        lo = np.minimum(lo, xyz.min(axis=0))
        del xyz
    origin = np.floor(lo / args.voxel) * args.voxel - args.voxel

    log("pass 2: per-station best point per voxel")
    parts = []
    covis = []
    station_stats = []
    for code, s in enumerate(args.stations):
        _, xyz = read_xyz(args.pano_dir, s)
        w = np.load(Path(args.quality_dir) / f"{s}_point_quality.npz")["w_uint8"]
        if len(w) != len(xyz):
            raise ValueError(f"{s}: quality array length differs from PLY")
        keys = voxel_keys(xyz, origin, args.voxel)
        covis.append(np.unique(voxel_keys(xyz, origin, COVIS_CELL_M)))
        del xyz
        idx = best_per_voxel(keys, w)
        parts.append((keys[idx], w[idx], np.full(len(idx), code, np.uint8), idx.astype(np.int64)))
        station_stats.append({"station": s, "original_points": int(len(keys)),
                              "occupied_voxels": int(len(idx))})
        log(f"  {s}: {len(keys):,} points -> {len(idx):,} voxels")
        del keys, w

    keys = np.concatenate([p[0] for p in parts])
    w = np.concatenate([p[1] for p in parts])
    st = np.concatenate([p[2] for p in parts])
    src = np.concatenate([p[3] for p in parts])
    del parts

    log("pass 3: merge stations")
    pick = best_per_voxel(keys, w)
    rep_keys = np.unique(keys)
    rep_keys_pick = keys[pick]
    if not np.array_equal(rep_keys, rep_keys_pick):
        raise RuntimeError("merged representative keys are not aligned")
    rep_station, rep_src = st[pick], src[pick]
    del keys, w, st, src, pick
    m = len(rep_keys)
    log(f"  merged representatives: {m:,}")

    log("pass 4: gather coordinates and colours")
    rec = np.empty(m, dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                             ("red", "u1"), ("green", "u1"), ("blue", "u1"),
                             ("rep_hi", "<u2"), ("rep_lo", "<u2"),
                             ("station", "u1"), ("n_obs_stations", "u1")])
    for code, s in enumerate(args.stations):
        sel = np.flatnonzero(rep_station == code)
        v = PlyData.read(Path(args.pano_dir) / s / f"{s}_pano_labeled.ply", mmap=True)["vertex"].data
        rows = rep_src[sel]
        for f in ("x", "y", "z", "red", "green", "blue"):
            rec[f][sel] = np.asarray(v[f])[rows]
        rec["station"][sel] = station_number(s)
        station_stats[code]["representatives_from_station"] = int(len(sel))
    rep_id = np.arange(m, dtype=np.int64)
    rec["rep_hi"], rec["rep_lo"] = rep_id >> 16, rep_id & 0xFFFF
    rep_xyz = np.column_stack((rec["x"], rec["y"], rec["z"])).astype(np.float64)
    rep_cell = voxel_keys(rep_xyz, origin, COVIS_CELL_M)
    n_obs = sum(np.isin(rep_cell, c).astype(np.uint8) for c in covis)
    rec["n_obs_stations"] = n_obs

    out.mkdir(parents=True)
    ply_path = out / "annotation_cloud_station5_6_7.ply"
    header = "\n".join(["ply", "format binary_little_endian 1.0", f"element vertex {m}",
                        "property float x", "property float y", "property float z",
                        "property uchar red", "property uchar green", "property uchar blue",
                        "property ushort rep_hi", "property ushort rep_lo",
                        "property uchar station", "property uchar n_obs_stations",
                        "end_header", ""]).encode("ascii")
    with open(ply_path, "wb") as f:
        f.write(header)
        rec.tofile(f)
    np.save(out / "rep_voxel_keys.npy", rep_keys)

    obs_hist = {int(k): int(c) for k, c in zip(*np.unique(n_obs, return_counts=True))}
    manifest = {
        "purpose": "blind point-level semantic GT annotation cloud (dev region)",
        "script": "make_semantic_point_gt_cloud01.py",
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "stations": args.stations,
        "pano_dir": args.pano_dir,
        "quality_dir": args.quality_dir,
        "voxel_m": args.voxel,
        "voxel_origin": origin.tolist(),
        "voxel_key_bits_per_axis": BITS,
        "representative_rule": "max Step01b observation weight w_uint8 per voxel; ties -> lowest station order then lowest point index",
        "point_correspondence": "outputs18 <station>_pano_labeled.ply point i == E57 scan-0 point i (point counts verified equal)",
        "n_representatives": int(m),
        "n_obs_stations_cell_m": COVIS_CELL_M,
        "n_obs_stations_histogram": obs_hist,
        "stations_detail": station_stats,
        "excluded_on_purpose": ["Grounded-SAM semantic/instance labels", "ghost/mixed/isolated flags",
                                "patch, surface, element ids", "any semantic prediction"],
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    log(f"wrote {ply_path} ({ply_path.stat().st_size / 1e9:.2f} GB)")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
