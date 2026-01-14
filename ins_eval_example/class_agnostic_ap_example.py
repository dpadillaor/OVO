import pathlib
import numpy as np
import argparse

import ins_eval_utils
import replica
import scannet200


def read_gt(gt_file, dataset_module=None):
    with open(gt_file, 'r') as f:
        gt_ids = f.read().splitlines()
        gt_ids = np.array(gt_ids, dtype=np.int64)
    ids = np.unique(gt_ids)
    binary_masks = ids[None] == gt_ids[:, None]
    # The first three digit encode the instance number of a classe, while from the 4th encode the class.
    # E.g.: ids 13000 and 13001 are instances 0 and 1 of class 13
    classes = ids // 1000
    if dataset_module:
        # Evaluation function assumes the queries used to classify instances where dataset_module.ins_class_names
        # Therefore, we have to map gt ids to the same range of ids.
        new_classes = np.ones_like(classes) * len(
            dataset_module.valid_ins_class_ids)
        for i, id in enumerate(dataset_module.valid_ins_class_ids):
            new_classes[classes == id] = i
        classes = new_classes
    return binary_masks, classes


def test_class_agnostic_ap(args):
    if args.dataset == 'scannet200':
        dataset_module = scannet200
    elif args.dataset == 'replica':
        dataset_module = replica
    else:
        raise NotImplementedError(args.dataset)


    gt_dir = pathlib.Path(args.gt_path)
    if not gt_dir.exists():
        raise ValueError("Wrong path for ground_truth data")

    predictions = {}

    # Expected predictions format:
    # dict:
    #   - scene (dict):
    #       - pred_masks (np.array): array of dim (nº points, nº instances). Each point has a value of (1, 0) depending if it belong or not to an instance.
    #       - pred_classes (np.ndarray): array of dim (nº instances). Id of the predicted class for each instance.
    #       - pred_scores (np.ndarray): array of dim (nº instances). Confidence of prediction for each instance.

    for scene in dataset_module.scenes:
        # Compute masks from gt
        preds_3d, classes = read_gt(gt_dir / f"{scene}.txt", dataset_module)
        predictions[scene] = {
            "pred_masks": preds_3d,
            "pred_classes": classes,
            "pred_scores": np.ones((preds_3d.shape[1]))
        }
    metrics = ins_eval_utils.evaluate(predictions,
                                dataset_module,
                                gt_path=gt_dir,
                                class_agnostic=False)

    print(f"mAP using gt masks and classes: {metrics['AP']:.3f}")

    # Change classes prediction to wrong values
    for scene, pred in predictions.items():
        pred["pred_classes"] = np.zeros_like(pred["pred_classes"])

    metrics = ins_eval_utils.evaluate(predictions,
                                dataset_module,
                                gt_path=gt_dir,
                                class_agnostic=False)
    print(f"mAP using gt masks but wrong classes: {metrics['AP']:.3f}")

    metrics = ins_eval_utils.evaluate(predictions,
                                dataset_module,
                                gt_path=gt_dir,
                                class_agnostic=True)
    print(
        f"class-agnostic mAP using gt masks but wrong classes: {metrics['AP']:.3f}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--dataset', default='replica', type=str)
    parser.add_argument('--gt_path',
                        default="./replica_gt/",
                        type=str)
    args = parser.parse_args()
    test_class_agnostic_ap(args)
