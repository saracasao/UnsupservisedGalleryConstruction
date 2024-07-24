import json 
import os 
import numpy as np
import sys
import pathlib

sys.path.append(str(pathlib.Path().absolute()))

from UnsupservisedGalleryConstruction.DataInformation import getDataList, extract_info_clusters, load_identities


def load_metrics(dir_saveMetric):
    # Metrics in the classification process
    metrics = {'predictions': [],
               'ground_truth': [],
               'pred_identity': [],
               'gallery_precision': [],
               'cluster_labels': []}

    name_files = os.listdir(dir_saveMetric)
    name_files = np.sort(name_files)
    for name_file in name_files:
        with open(dir_saveMetric + '/' + name_file) as f:
            data = json.load(f)
        # Accuracy of class prediction and recall
        metrics['predictions'] = getDataList(data['pred_clasif'], metrics['predictions'])
        metrics['predictions'] = getDataList(data['pred_merge'], metrics['predictions'])
        metrics['predictions'] = getDataList(data['pred_init'], metrics['predictions'])

        metrics['pred_identity'] = getDataList(data['id_clasif'], metrics['pred_identity'])
        metrics['pred_identity'] = getDataList(data['id_merge'], metrics['pred_identity'])
        metrics['pred_identity'] = getDataList(data['id_init'], metrics['pred_identity'])

        metrics['ground_truth'] = getDataList(data['gt_clasif'], metrics['ground_truth'])
        metrics['ground_truth'] = getDataList(data['gt_merge'], metrics['ground_truth'])
        metrics['ground_truth'] = getDataList(data['gt_init'], metrics['ground_truth'])

        metrics['gallery_precision'] = getDataList(data['GalleryPrecision'], metrics['gallery_precision'])
        metrics['cluster_labels'] = getDataList(data['cluster_labels'], metrics['cluster_labels'])

    return metrics


def get_sample_reid_evaluation(metrics, gt_identities, data_info, dir_final_models, N=4):
    clusters = metrics['cluster_labels']

    # Precision
    predictions = metrics['predictions']
    pred_identity = metrics['pred_identity']
    groundTruth = metrics['ground_truth']

    assert len(predictions) == len(pred_identity) == len(groundTruth)
    id_clusters_initialized, amount_clusters_same_id = np.unique(clusters, return_counts=True)
    id_scenario, amount_clusters_same_id = get_info_scenario(id_clusters_initialized, gt_identities,
                                                             amount_clusters_same_id)

    clusters_info = extract_info_clusters(dir_final_models)
    clusters_label = [c[3] for c in clusters_info]
    clusters_id = [c[4] for c in clusters_info]

    assert len(clusters_label) == len(clusters_id)

    models_dict = {}
    for cl in id_scenario:
        cl_idx = [i for i, l in enumerate(clusters_label) if l == cl]
        c_ids = sorted([c_id for i, c_id in enumerate(clusters_id) if i in cl_idx])
        models_dict[cl] = c_ids

    avg_precision, avg_recall = get_precision_recall(predictions, pred_identity, groundTruth, id_scenario, models_dict, data_info, N)
    return avg_precision, avg_recall


def get_precision_recall(predictions, pred_identity, groundTruth, id_scenario, models_dict, data_info, N):
    predictions = np.array(predictions)
    groundTruth = np.array(groundTruth)
    pred_identity = np.array(pred_identity)

    precision, recall = [], []
    for i, cl in enumerate(id_scenario):
        if len(models_dict[cl]) > 0:
            # Get data
            predictions_cluster_idx = np.where(predictions == cl)
            predictions_cluster = predictions[predictions_cluster_idx]
            predictions_cluster_identity = pred_identity[predictions_cluster_idx]
            gt = groundTruth[predictions_cluster_idx]

            # Select valid clusters
            pred_id, freq_id = np.unique(predictions_cluster_identity, return_counts=True)
            idx_freq = np.argsort(freq_id)[::-1]
            modelID_valids = pred_id[idx_freq][:N]
            modelID_valids = list(modelID_valids)

            # Get matches
            match_gt = (predictions_cluster == gt)
            match_id = [p_id in modelID_valids for p_id in predictions_cluster_identity]
            match_id = np.array(match_id)
            match = match_gt & match_id

            valid_predictions = match_gt[match_id]
            failures = np.invert(valid_predictions)

            assert match_id.sum() >= match.sum()

            tp = match.sum()
            fp = failures.sum()
            fn = get_FN(cl, data_info, tp)

            P = tp / (tp + fp)
            R = tp / (tp + fn)

            precision.append(P)
            recall.append(R)
        else:
            precision.append(0)
            recall.append(0)

    avg_precision = sum(precision) / len(id_scenario)
    avg_recall = sum(recall) / len(id_scenario)
    return avg_precision, avg_recall


def get_FN(identity, data_info, TP):
    data_id = [d for d in data_info if d[1] == identity]

    total_img_id = 0
    for d in data_id:
        tracklets = d[0]
        total_img_id = total_img_id + len(tracklets)
    fn = total_img_id - TP
    return fn


def get_info_scenario(id_initialized, gt_identities, n_clusters_same_id):
    id_scenario, num_clusters_same_id = [], []
    for i in gt_identities:
        if i in id_initialized:
            i_idx = np.where(id_initialized == i)
            id_scenario.append(id_initialized[i_idx][0])
            num_clusters_same_id.append(n_clusters_same_id[i_idx][0])
        else:
            id_scenario.append(i)
            num_clusters_same_id.append(0)
    assert len(id_scenario) == len(num_clusters_same_id) == len(gt_identities)

    return id_scenario, num_clusters_same_id



