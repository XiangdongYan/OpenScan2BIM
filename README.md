# OpenScan2BIM pipeline

This folder holds the current code for the OpenScan2BIM semantic segmentation
pipeline only. It is a standalone git repository: commit before every run;
the run manifest records the commit and each file's SHA-256. File names carry
no version suffix; history lives in git.

The first commit imported the legacy scripts verbatim. Every later change is a
reviewable diff: `git log --follow <file>` shows where a stage came from. The
method is described in `docs/research_design/OpenScan2BIM_Framework_v01.md`. The
consolidation plan is in
`docs/research_design/OpenScan2BIM_Workspace_CleanRun_Plan_v01.md`.

## Run

From the workspace root `C:\Disk_C\Grounded-Segment-Anything`:

```
D:/Anaconda/python pipeline/run_pipeline.py --config pipeline/config/<run>.json
```

`run_pipeline.py` starts every stage as a separate process from the workspace
root. A stage writes `runs/<run_id>/<stage>/` (named after its script), its log
`runs/<run_id>/logs/<stage>.log`, and a `_stage_done.json` marker.

Options:
- `--resume` continues an interrupted run. s08 resumes from its per-record checkpoints.
- `--stop_after <stage>` stops after the named stage.

A run directory is never reused. Each change to the pipeline gets a new run id,
whose configuration differs from its parent's in one item. Every run is
recorded in `runs/RUNS.md`.

## Stages

| Stage | Layer | Purpose | Legacy source |
|---|---|---|---|
| `s00_pano_gsam` | 0 data | E57 -> panorama, Grounded-SAM masks, labeled PLY. Run only for new stations; dev data is frozen in `data/stage0_pano_gsam_level02lab` (conf_threshold 0.27) | Step01_Segmentation09 |
| `s01_atomic_patches` | 1.1 | Atomic planar patches and the point-quality layer, one station per call | Step01b_Primitives04 |
| `s02_door_cuts` | 1.2 | Grounded-SAM door cuts for `door_cut_stations`; the other stations are passed through as hard links | Step01b2_DoorPreservation01 |
| `s03_local_surfaces` | 1.3 | Station-local surfaces with the door cannot-link, one station per call | Step01c_LocalSurfaces16 |
| `s04a_station_references` | 0 | Per-station floor/upper references (Qwen geometry card) | Step01c_ElevationBands02 |
| `s04b_region_references` | 0 | Region references `z_f`, `z_u` (Final, point labels) | Step01c_ElevationBands03 |
| `s05a_association` | 1.4 | Cross-station surface association | Step04_Fusion09 |
| `s05b_association_qa` | 1.4 | Component plane QA | Step04_Fusion07 |
| `s05c_element_records` | 1.4 | Provisional element records | Step04b_ElementRecords02 |
| `s06a_station_projection` | 2 inputs | Panorama evidence per station-element | Step04d_StationElementProjection01 |
| `s06b_rectilinear_views` | 2 inputs | Rectilinear views and metric geometry | Step04d_RectilinearElementProjection05 |
| `s07_area_metrics` | 2.1 | Deduplicated areas | Step04e_ElementAreaMetrics02 |
| `s08_qwen_scoring` | 2.2 | Qwen3-VL Stage 1 + Stage 2, every record scored in the run | Step05c_RectilinearMetricJointScoring09 (reuse removed) |
| `s09_final` | 2.3 | Element Final policy (`lib/final_policy.py`) | Step05d_ElementSemanticFinal07 + SceneGeometryConstraint10 |
| `s10_point_labels` | 3 | Point labels, upper split, residual completion | Step06_PointLabels01 |
| `s11_evaluate` | eval | Dev metrics against the point GT | evaluate_semantic_point_gt02 |
| `s11b_error_attribution` | eval | Error attribution by layer | tools/diagnose_error_attribution01 |

`lib/` holds the shared helpers that the stages import. They are verbatim
definitions from the legacy modules named in each file's docstring:
`scan_geometry`, `panorama`, `rectilinear`, `qwen_scoring`, `final_policy`.

`tools/` holds GT construction (`gt_make_annotation_cloud`,
`gt_transfer_labels`) and one-off verification scripts.

## Environments

- `cpu`: `D:/Anaconda/python`, for every stage except the two below.
- `qwen`: `D:/Anaconda/envs/qwen_vl_pilot/python`, for s08. Qwen3-VL-8B is read from the local HF cache.
- `s00` needs `D:/Anaconda/envs/GroundingDino/python`.
