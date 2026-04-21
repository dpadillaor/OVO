# Input:
#   - path to .txt prediction files
#   - path to .txt ground truth files
# The relative paths to predicted masks must contain one integer per line,
# where each line corresponds to vertices in the *_vh_clean_2.ply (in that order).
# Non-zero integers indicate part of the predicted instance.
# The label ids specify the class of the corresponding mask.
# Confidence is a float confidence score of the mask.
#
# Note that only the valid classes are used for evaluation,
# i.e., any ground truth label not in the valid label set
# is ignored in the evaluation.

import os
import sys
from uuid import uuid4
import numpy as np
import json
import collections


class Instance(object):
    instance_id = 0
    label_id = 0
    vert_count = 0
    med_dist = -1
    dist_conf = 0.0

    def __init__(self, mesh_vert_instances, instance_id):
        if instance_id == -1:
            return
        self.instance_id = int(instance_id)
        self.label_id = int(self.get_label_id(instance_id))
        self.vert_count = int(
            self.get_instance_verts(mesh_vert_instances, instance_id))

    def get_label_id(self, instance_id):
        return int(instance_id // 1000)

    def get_instance_verts(self, mesh_vert_instances, instance_id):
        return (mesh_vert_instances == instance_id).sum()

    def to_json(self):
        return json.dumps(self,
                          default=lambda o: o.__dict__,
                          sort_keys=True,
                          indent=4)

    def to_dict(self):
        dict = {}
        dict["instance_id"] = self.instance_id
        dict["label_id"] = self.label_id
        dict["vert_count"] = self.vert_count
        dict["med_dist"] = self.med_dist
        dict["dist_conf"] = self.dist_conf
        dict["matched_pred"] = []
        return dict

    def from_json(self, data):
        self.instance_id = int(data["instance_id"])
        self.label_id = int(data["label_id"])
        self.vert_count = int(data["vert_count"])
        if "med_dist" in data:
            self.med_dist = float(data["med_dist"])
            self.dist_conf = float(data["dist_conf"])

    def __str__(self):
        return "(" + str(self.instance_id) + ")"


def make_pred_info(pred: dict, pred_id_to_id):
    pred_info = {}
    assert (pred['pred_classes'].shape[0] == pred['pred_scores'].shape[0] ==
            pred['pred_masks'].shape[1])
    for i in range(len(pred['pred_classes'])):
        info = {}
        info["label_id"] = pred_id_to_id.get(int(pred['pred_classes'][i]), -1)
        info["conf"] = pred['pred_scores'][i]
        info["mask"] = pred['pred_masks'][:, i]
        pred_info[uuid4()] = info  # we later need to identify these objects
    return pred_info


def get_instances(ids, class_ids, class_labels, id2label):
    instances = {}
    for label in class_labels:
        instances[label] = []
    instance_ids = np.unique(ids)
    for id in instance_ids:
        if id == 0:
            continue
        inst = Instance(ids, id)
        if inst.label_id in class_ids:
            inst_dict = inst.to_dict()
            instances[id2label[inst.label_id]].append(inst_dict)
    return instances


def assign_instances_for_scan(pred_info: dict, gt_ids, class_labels,
                              id_to_label, valid_class_ids, opt):
    # get gt instances
    gt2pred = get_instances(gt_ids, valid_class_ids, class_labels, id_to_label)

    pred2gt = {}
    for label in class_labels:
        pred2gt[label] = []
    num_pred_instances = 0
    # mask of void labels in the groundtruth
    bool_void = np.logical_not(np.isin(gt_ids // 1000, valid_class_ids))
    # go thru all prediction masks
    for uuid, info in pred_info.items():
        label_id = int(info['label_id'])
        conf = info['conf']
        if label_id not in id_to_label:
            continue
        label_name = id_to_label[label_id]
        # read the mask
        pred_mask = info['mask']
        assert (len(pred_mask) == len(gt_ids))
        # convert to binary
        pred_mask = np.not_equal(pred_mask, 0)
        num = np.count_nonzero(pred_mask)
        if num < opt['min_region_sizes'][0]:
            continue  # skip if empty

        pred_instance = {}
        pred_instance['uuid'] = uuid
        pred_instance['pred_id'] = num_pred_instances
        pred_instance['label_id'] = label_id
        pred_instance['vert_count'] = num
        pred_instance['confidence'] = conf
        pred_instance['void_intersection'] = np.count_nonzero(
            np.logical_and(bool_void, pred_mask))

        # matched gt instances
        matched_gt = []
        # go thru all gt instances with matching label
        for (gt_num, gt_inst) in enumerate(gt2pred[label_name]):
            intersection = np.count_nonzero(
                np.logical_and(gt_ids == gt_inst['instance_id'], pred_mask))
            if intersection > 0:
                gt_copy = gt_inst.copy()
                pred_copy = pred_instance.copy()
                gt_copy['intersection'] = intersection
                pred_copy['intersection'] = intersection
                matched_gt.append(gt_copy)
                gt2pred[label_name][gt_num]['matched_pred'].append(pred_copy)

        pred_instance['matched_gt'] = matched_gt
        num_pred_instances += 1
        pred2gt[label_name].append(pred_instance)

    return gt2pred, pred2gt


def evaluate_matches(matches, class_labels, id_to_label, opt):

    overlaps = opt['overlaps']
    min_region_sizes = [opt['min_region_sizes'][0]]
    dist_threshes = [opt['distance_threshes'][0]]
    dist_confs = [opt['distance_confs'][0]]

    # results: class x overlap
    ap = np.zeros((len(dist_threshes), len(class_labels), len(overlaps)),
                  float)
    ar = np.zeros((len(dist_threshes), len(class_labels), len(overlaps)),
                  float)  # average recall
    rc = np.zeros((len(dist_threshes), len(class_labels), len(overlaps)),
                  float)  # recall
    matched_predictions_category_names = {}
    gt_category_names = {}

    for di, (min_region_size, distance_thresh, distance_conf) in enumerate(
            zip(min_region_sizes, dist_threshes, dist_confs)):
        matched_predictions_category_names[di] = {}
        for oi, overlap_th in enumerate(overlaps):
            matched_predictions_category_names[di][oi] = {}
            for m in matches.keys():
                gt_category_names[m] = set([])
                matched_predictions_category_names[di][oi][m] = set([])

    for di, (min_region_size, distance_thresh, distance_conf) in enumerate(
            zip(min_region_sizes, dist_threshes, dist_confs)):

        for oi, overlap_th in enumerate(overlaps):
            pred_visited = {}
            for m in matches:
                for p in matches[m]['pred']:
                    for label_name in class_labels:
                        for p in matches[m]['pred'][label_name]:
                            if 'uuid' in p:
                                pred_visited[p['uuid']] = False

            for li, label_name in enumerate(class_labels):
                y_true = np.empty(0)
                y_score = np.empty(0)
                hard_false_negatives = 0
                has_gt = False
                has_pred = False

                for m in matches:

                    #group all common and tail under unknown
                    pred_instances = matches[m]['pred'][label_name]
                    gt_instances = matches[m]['gt'][label_name]

                    # filter groups in ground truth
                    gt_instances = [
                        gt for gt in gt_instances
                        if gt['instance_id'] >= 1000 and gt['vert_count'] >=
                        min_region_size and gt['med_dist'] <= distance_thresh
                        and gt['dist_conf'] >= distance_conf
                    ]
                    if gt_instances:
                        has_gt = True
                    if pred_instances:
                        has_pred = True

                    cur_true = np.ones(len(gt_instances))
                    cur_score = np.ones(len(gt_instances)) * (-float("inf"))
                    cur_match = np.zeros(len(gt_instances), dtype=bool)
                    # collect matches
                    for (gti, gt) in enumerate(gt_instances):
                        gt_category_names[m].add(id_to_label[gt['label_id']])
                        found_match = False
                        for pred in gt['matched_pred']:
                            # greedy assignments
                            if pred_visited[pred['uuid']]:
                                continue
                            overlap = float(pred['intersection']) / (
                                gt['vert_count'] + pred['vert_count'] -
                                pred['intersection'])
                            if overlap > overlap_th:
                                confidence = pred['confidence']
                                # if already have a prediction for this gt,
                                # the prediction with the lower score is automatically a false positive
                                if cur_match[gti]:
                                    max_score = max(cur_score[gti], confidence)
                                    min_score = min(cur_score[gti], confidence)
                                    cur_score[gti] = max_score
                                    # append false positive
                                    cur_true = np.append(cur_true, 0)
                                    cur_score = np.append(cur_score, min_score)
                                    cur_match = np.append(cur_match, True)
                                # otherwise set score
                                else:
                                    found_match = True
                                    cur_match[gti] = True
                                    cur_score[gti] = confidence
                                    pred_visited[pred['uuid']] = True
                        if not found_match:
                            hard_false_negatives += 1
                        if found_match:
                            matched_predictions_category_names[di][oi][m].add(
                                id_to_label[gt['label_id']])
                    # remove non-matched ground truth instances
                    cur_true = cur_true[cur_match]
                    cur_score = cur_score[cur_match]

                    # collect non-matched predictions as false positive
                    for pred in pred_instances:
                        found_gt = False
                        for gt in pred['matched_gt']:

                            overlap = float(gt['intersection']) / (
                                gt['vert_count'] + pred['vert_count'] -
                                gt['intersection'])
                            if overlap > overlap_th:
                                found_gt = True
                                break
                        if not found_gt:
                            num_ignore = pred['void_intersection']
                            for gt in pred['matched_gt']:
                                # group?
                                if gt['instance_id'] < 1000:
                                    num_ignore += gt['intersection']
                                # small ground truth instances
                                if gt['vert_count'] < min_region_size or gt[
                                        'med_dist'] > distance_thresh or gt[
                                            'dist_conf'] < distance_conf:
                                    num_ignore += gt['intersection']
                            proportion_ignore = float(
                                num_ignore) / pred['vert_count']
                            # if not ignored append false positive
                            if proportion_ignore <= overlap_th:
                                cur_true = np.append(cur_true, 0)
                                confidence = pred["confidence"]
                                cur_score = np.append(cur_score, confidence)

                    # append to overall results
                    y_true = np.append(y_true, cur_true)
                    y_score = np.append(y_score, cur_score)

                # compute average precision
                if has_gt and has_pred:
                    # compute precision recall curve first

                    # sorting and cumsum
                    score_arg_sort = np.argsort(y_score)
                    y_score_sorted = y_score[score_arg_sort]
                    y_true_sorted = y_true[score_arg_sort]
                    y_true_sorted_cumsum = np.cumsum(y_true_sorted)

                    # unique thresholds
                    (thresholds, unique_indices) = np.unique(y_score_sorted,
                                                             return_index=True)
                    num_prec_recall = len(unique_indices) + 1

                    # prepare precision recall
                    num_examples = len(y_score_sorted)
                    # https://github.com/ScanNet/ScanNet/pull/26
                    # all predictions are non-matched but also all of them are ignored and not counted as FP
                    # y_true_sorted_cumsum is empty
                    num_true_examples = y_true_sorted_cumsum[-1] if len(
                        y_true_sorted_cumsum) > 0 else 0
                    precision = np.zeros(num_prec_recall)
                    recall = np.zeros(num_prec_recall)

                    # deal with the first point
                    y_true_sorted_cumsum = np.append(y_true_sorted_cumsum, 0)
                    # deal with remaining
                    for idx_res, idx_scores in enumerate(unique_indices):
                        cumsum = y_true_sorted_cumsum[idx_scores - 1]
                        tp = num_true_examples - cumsum
                        fp = num_examples - idx_scores - tp
                        fn = cumsum + hard_false_negatives
                        p = float(tp) / (tp + fp)
                        r = float(tp) / (tp + fn)
                        precision[idx_res] = p
                        recall[idx_res] = r

                    # recall is the first point on recall curve # from softgroup
                    #https://github.com/thangvubk/SoftGroup/blob/main/softgroup/evaluation/instance_eval.py
                    rc_current = recall[0]

                    # first point in curve is artificial
                    precision[-1] = 1.
                    recall[-1] = 0.

                    # compute average of precision-recall curve
                    recall_for_conv = np.copy(recall)
                    recall_for_conv = np.append(recall_for_conv[0],
                                                recall_for_conv)
                    recall_for_conv = np.append(recall_for_conv, 0.)

                    stepWidths = np.convolve(recall_for_conv, [-0.5, 0, 0.5],
                                             'valid')
                    # integrate is now simply a dot product
                    ap_current = np.dot(precision, stepWidths)

                    stepWidths_ar = stepWidths
                    ar_current = np.dot(recall, stepWidths_ar)

                elif has_gt:
                    ap_current = 0.0
                    ar_current = 0.0
                    rc_current = 0.0
                else:
                    ap_current = float('nan')
                    ar_current = float('nan')
                    rc_current = float('nan')
                ap[di, li, oi] = ap_current
                ar[di, li, oi] = ar_current
                rc[di, li, oi] = rc_current

    return ap, ar, rc


def compute_averages(aps,
                     class_labels,
                     opt,
                     head_cats=[],
                     comm_cats=[],
                     tail_cats=[],
                     new_cats=[]):
    d_inf = 0
    o50 = np.where(np.isclose(opt['overlaps'], 0.5))
    o25 = np.where(np.isclose(opt['overlaps'], 0.25))
    oAllBut25 = np.where(np.logical_not(np.isclose(opt['overlaps'], 0.25)))
    avg_dict = {}

    avg_dict['all_ap'] = np.nanmean(aps[d_inf, :, oAllBut25])
    avg_dict['all_ap_50%'] = np.nanmean(aps[d_inf, :, o50])
    avg_dict['all_ap_25%'] = np.nanmean(aps[d_inf, :, o25])
    avg_dict["classes"] = {}

    head_scores = {title: [] for title in ['ap', 'ap25%', 'ap50%']}
    common_scores = {title: [] for title in ['ap', 'ap25%', 'ap50%']}
    tail_scores = {title: [] for title in ['ap', 'ap25%', 'ap50%']}
    new_scores = {title: [] for title in ['ap', 'ap25%', 'ap50%']}

    for (li, label_name) in enumerate(class_labels):
        if label_name not in avg_dict["classes"]:
            avg_dict["classes"][label_name] = {}

        avg_dict["classes"][label_name]["ap"] = np.average(aps[d_inf, li,
                                                               oAllBut25])
        avg_dict["classes"][label_name]["ap50%"] = np.average(aps[d_inf, li,
                                                                  o50])
        avg_dict["classes"][label_name]["ap25%"] = np.average(aps[d_inf, li,
                                                                  o25])
        if head_cats and comm_cats and tail_cats:
            if (label_name in head_cats):
                for ap_type in ['ap', 'ap25%', 'ap50%']:
                    head_scores[ap_type].append(
                        avg_dict["classes"][label_name][ap_type])
            elif (label_name in comm_cats):
                for ap_type in ['ap', 'ap25%', 'ap50%']:
                    common_scores[ap_type].append(
                        avg_dict["classes"][label_name][ap_type])
            elif (label_name in tail_cats):
                for ap_type in ['ap', 'ap25%', 'ap50%']:
                    tail_scores[ap_type].append(
                        avg_dict["classes"][label_name][ap_type])
            else:
                raise NotImplementedError(label_name)
        if new_cats:
            new_labels = []
            if (label_name in new_cats):
                new_labels.append(label_name)
                for ap_type in ['ap', 'ap25%', 'ap50%']:
                    new_scores[ap_type].append(
                        avg_dict["classes"][label_name][ap_type])
    if head_cats or comm_cats or tail_cats:

        for score_type in ['ap', 'ap25%', 'ap50%']:
            avg_dict['head_' + score_type] = np.nanmean(
                head_scores[score_type])
            avg_dict['common_' + score_type] = np.nanmean(
                common_scores[score_type])
            avg_dict['tail_' + score_type] = np.nanmean(
                tail_scores[score_type])
    if new_cats:
        for score_type in ['ap', 'ap25%', 'ap50%']:
            avg_dict['new_' + score_type] = np.nanmean(new_scores[score_type])

    return avg_dict

def evaluate(preds: dict,
             dataset_module,
             gt_path,
             class_agnostic: bool = False):

    valid_ins_class_ids = dataset_module.valid_ins_class_ids
    if not class_agnostic:
        ins_class_labels = dataset_module.ins_class_names
        try:
            head_cats = dataset_module.ins_head_cats
            comm_cats = dataset_module.ins_comm_cats
            tail_cats = dataset_module.ins_tail_cats
        except AttributeError:
            head_cats = []
            comm_cats = []
            tail_cats = []
            print(
                "Instance classes splits [head, comm, tail] not present on dataset module. Skipping corresponding metrics."
            )
        new_cats = []
        id_to_label = {}
        label_id_to_id = {}
        pred_id_to_id = {}
        for i in range(len(valid_ins_class_ids)):
            pred_id_to_id[i] = valid_ins_class_ids[i]
            label_id_to_id[ins_class_labels[i]] = valid_ins_class_ids[i]
            id_to_label[valid_ins_class_ids[i]] = ins_class_labels[i]

        pred_id_to_id[len(ins_class_labels)] = -1
    else:
        # Label all gt and all predictions as the same class
        head_cats, comm_cats, tail_cats, new_cats = [], [], [], []
        ins_class_labels = ["generic"]
        id_to_label = collections.defaultdict(lambda: "generic")
        id_to_label[1] = "generic"
        label_id_to_id = {"generic": 1}
        pred_id_to_id = {1: 1}
        for pred in preds.values():
            pred["pred_classes"] = np.ones_like(pred["pred_classes"])

    # ---------- Evaluation params ---------- #
    # overlaps for evaluation
    opt = {}
    opt['overlaps'] = np.append(np.arange(0.5, 0.95, 0.05), 0.25)
    # minimum region size for evaluation [verts]
    opt['min_region_sizes'] = np.array([100])  # 100 for scannet
    # distance thresholds [m]
    opt['distance_threshes'] = np.array([float('inf')])
    # distance confidences
    opt['distance_confs'] = np.array([-float('inf')])

    print('evaluating', len(preds), 'scans...')
    matches = {}
    for i, (k, v) in enumerate(preds.items()):
        gt_file = os.path.join(gt_path, k + ".txt")
        if not os.path.isfile(gt_file):
            print(f"Missing GT for {k}. Skipping Instance metrics!")
            return {}

        with open(gt_file, 'r') as f:
            gt_ids = f.read().splitlines()
            gt_ids = np.array(gt_ids, dtype=np.int64)

        matches_key = os.path.abspath(gt_file)
        # assign gt to predictions
        pred_info = make_pred_info(v, pred_id_to_id)
        gt2pred, pred2gt = assign_instances_for_scan(pred_info, gt_ids,
                                                     ins_class_labels,
                                                     id_to_label,
                                                     valid_ins_class_ids, opt)

        matches[matches_key] = {}
        matches[matches_key]['gt'] = gt2pred
        matches[matches_key]['pred'] = pred2gt
        sys.stdout.write("\rscans processed: {}".format(i + 1))
        sys.stdout.flush()
    print('')
    ap_scores, ar_scores, rc_scores = evaluate_matches(matches,
                                                       ins_class_labels,
                                                       id_to_label, opt)
    avgs = compute_averages(ap_scores, ins_class_labels, opt, head_cats,
                            comm_cats, tail_cats, new_cats)

    metrics = {
        "AP": avgs["all_ap"],
        "AP_50": avgs["all_ap_50%"],
        "AP_25": avgs["all_ap_25%"],
        "AP_head": avgs.get("head_ap"),
        "AP_comm": avgs.get("common_ap"),
        "AP_tail": avgs.get("tail_ap"),
    }
    if "new_ap" in avgs:
        metrics["AP_new"] = avgs["new_ap"]

    return metrics
