import pathlib
import numpy as np
import argparse

import ins_eval_utils
import replica
import scannet200
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from ovo.utils.io_utils import rle_decode
import json
import ins_eval_utils
import replica
import scannet200


def load_predictions(experiment_path, scene_name):
    """
    Carga las predicciones desde la salida del experimento.
    
    Args:
        experiment_path: Path al experimento (ej: "data/output/Replica/experiment_name")
        scene_name: Nombre de la escena (ej: "office0")
    
    Returns:
        dict con: pred_masks, pred_classes, pred_scores
    """
    pred_dir = pathlib.Path(experiment_path) / "instance_pred"
    pred_file = pred_dir / f"{scene_name}.txt"
    
    if not pred_file.exists():
        raise ValueError(f"Archivo de predicciones no encontrado: {pred_file}")
    
    with open(pred_file, 'r') as f:
        lines = f.read().splitlines()
    
    masks_list = []
    classes_list = []
    scores_list = []
    
    for line in lines:
        parts = line.split()
        mask_file = parts[0]
        label = int(parts[1])
        conf = float(parts[2])
        
        # Cargar máscara desde JSON
        mask_path = pred_dir / mask_file
        with open(mask_path, 'r') as f:
            rle = json.load(f)
        
        mask = rle_decode(rle)
        masks_list.append(mask)
        classes_list.append(label)
        scores_list.append(conf)
    
    # Convertir a arrays numpy con forma correcta
    # pred_masks debe ser (n_points, n_instances)
    if len(masks_list) > 0:
        pred_masks = np.stack(masks_list, axis=1)
        pred_classes = np.array(classes_list)
        pred_scores = np.array(scores_list)
    else:
        pred_masks = np.zeros((0, 0))
        pred_classes = np.array([])
        pred_scores = np.array([])
    
    return {
        "pred_masks": pred_masks,
        "pred_classes": pred_classes,
        "pred_scores": pred_scores
    }


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

    # for scene in dataset_module.scenes:
    #     # Compute masks from gt
    #     preds_3d, classes = read_gt(gt_dir / f"{scene}.txt", dataset_module)
    #     predictions[scene] = {
    #         "pred_masks": preds_3d,
    #         "pred_classes": classes,
    #         "pred_scores": np.ones((preds_3d.shape[1]))
    #     }
    # metrics = ins_eval_utils.evaluate(predictions,
    #                             dataset_module,
    #                             gt_path=gt_dir,
    #                             class_agnostic=False)

    # print(f"mAP using gt masks and classes: {metrics['AP']:.3f}")

    # ========== HARDCODED: Siempre Replica y office0 ==========
    dataset_module = replica
    SCENE = "office0"
    EXPERIMENT_PATH = "data/output/Replica/20260121_office0_GTNoise-T0p05-R0p01_CLIP_CLIP-Fusion"
    # EXPERIMENT_PATH = "data/output/Replica/20260121_office0_GTNoise-T0p05-R0p01_SAM3_SAM3-Fusion"

    gt_dir = pathlib.Path(args.gt_path)
    if not gt_dir.exists():
        raise ValueError("Wrong path for ground_truth data")

    predictions = {}

    # Cargar predicciones
    print(f"Cargando predicciones de: {EXPERIMENT_PATH}")
    predictions[SCENE] = load_predictions(EXPERIMENT_PATH, SCENE)
    
    # Verificar formato
    print(f"\n{'='*60}")
    print(f"Verificando formato de predicciones para {SCENE}:")
    print(f"{'='*60}")
    pred_masks = predictions[SCENE]['pred_masks']
    pred_classes = predictions[SCENE]['pred_classes']
    pred_scores = predictions[SCENE]['pred_scores']
    
    print(f"✓ pred_masks shape: {pred_masks.shape}")
    print(f"  - Esperado: (nº points, nº instances)")
    print(f"  - Obtenido: ({pred_masks.shape[0]} points, {pred_masks.shape[1]} instances)")
    print(f"  - Tipo: {pred_masks.dtype}")
    print(f"  - Valores únicos: {np.unique(pred_masks)}")
    
    print(f"\n✓ pred_classes shape: {pred_classes.shape}")
    print(f"  - Esperado: (nº instances,)")
    print(f"  - Obtenido: ({pred_classes.shape[0]} instances,)")
    print(f"  - Tipo: {pred_classes.dtype}")
    print(f"  - Clases únicas: {np.unique(pred_classes)}")
    
    print(f"\n✓ pred_scores shape: {pred_scores.shape}")
    print(f"  - Esperado: (nº instances,)")
    print(f"  - Obtenido: ({pred_scores.shape[0]} instances,)")
    print(f"  - Tipo: {pred_scores.dtype}")
    print(f"  - Rango: [{pred_scores.min():.4f}, {pred_scores.max():.4f}]")
    print(f"{'='*60}\n")


    # # Change classes prediction to wrong values
    # for scene, pred in predictions.items():
    #     pred["pred_classes"] = np.zeros_like(pred["pred_classes"])

    # metrics = ins_eval_utils.evaluate(predictions,
    #                             dataset_module,
    #                             gt_path=gt_dir,
    #                             class_agnostic=False)
    # print(f"mAP using gt masks but wrong classes: {metrics['AP']:.3f}")

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





