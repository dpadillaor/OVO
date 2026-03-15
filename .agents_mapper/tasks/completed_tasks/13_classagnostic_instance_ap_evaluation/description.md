# Purpose

Implement a working class-agnostic instance AP evaluation pipeline for OVO experiments on the Replica dataset. The existing `ins_eval_example/class_agnostic_ap_scratch.py` loads predicted instance masks (saved by `run_eval.py --segment`) and computes mAP using `ins_eval_utils.evaluate()`, but the resulting metrics do not make sense. The root cause is suspected to be a misalignment between the predicted point cloud (OVO's internal 3D map) and the GT mesh vertices — a prerequisite that `match_labels_to_vtx` assumes but does not verify. The task involves: (1) diagnosing the actual failure point by visualizing `pcd_pred` vs `pcd_gt` before the matching step, (2) fixing any alignment or data format issues, and (3) producing a clean script that reliably computes class-agnostic AP metrics for a given experiment, so that instance segmentation quality can be measured independently of semantic classification accuracy.

## Task files
- `strategy.md`: Describes the architecture and the strategy to be implemented.
- `tests.md`: Summarizes the proposed TDD tests to validate the implementation.
