import os
import numpy as np
import json
import sys
import pathlib
from .distance import compute_distance_matrix
from .rank import evaluate_rank
from .utils_eval import load_metrics, get_sample_reid_evaluation

sys.path.append(str(pathlib.Path().absolute()))

from UnsupservisedGalleryConstruction.DataInformation import extract_info_gallery, extract_info_query, getDataList
from UnsupservisedGalleryConstruction.Dataset import DataLoader


def Gallery_Evaluation(config):
    print('*GALLERY EVALUATION:')
    print('---------------------')
    # Directories for loading results
    abs_path = str(pathlib.Path().absolute())
    dir_evaluation_files = (abs_path + '/evaluation/' + config.name_test +
                            '/' + str(pathlib.Path(config.iteration).stem))
    dir_metrics = dir_evaluation_files + '/metrics_gallery'
    dir_final_models = dir_evaluation_files + '/final_appearance_models'
    dir_features = os.path.join(abs_path, config.dir_features.replace('./', ''))

    # Load metrics
    metrics = load_metrics(dir_metrics)

    # Load ground truth gallery
    query_info, data_info, total_imgs = DataLoader('DukeMTMC-Video', 'individualMetrics',path_saveFeat= dir_features + '/')
    print('total imgs', total_imgs)

    data_id = [d[1] for d in data_info if d[1] != 0]
    data_id = list(set(data_id))

    # Precision & Recall ReId process
    precision, recall = get_sample_reid_evaluation(metrics, data_id, data_info, dir_final_models, N=4)
    F1_pred = (2 * precision * recall) / (precision + recall)

    # Gallery
    gallery_precision_array = np.array(metrics['gallery_precision'])
    gallery_precision = np.sum(gallery_precision_array) / len(gallery_precision_array)

    # Clusters Analysis
    total_clusters_initialized = len(metrics['cluster_labels'])
    new_clusters_initialized = list(set(metrics['cluster_labels']))

    cluster_precision = len(new_clusters_initialized) / total_clusters_initialized
    cluster_recall = len(new_clusters_initialized) / len(data_id)
    F1_clusters = (2 * cluster_precision * cluster_recall) / (cluster_precision + cluster_recall)

    print('** Results **')
    print('Accuracy of {} images included in the gallery: {:.1%}'.format(len(gallery_precision_array),
                                                                         gallery_precision))
    print('Analysis of the label assigned:')
    print('-> Precision, recall, F1 of predicted:{:.2%}, {:.2%},{:.2%}'.format(precision, recall,
                                                                               F1_pred))
    print('Cluster analysis:')
    print('-> Total of clusters initialized:', total_clusters_initialized)
    print('-> New clusters initialize correctly:', len(new_clusters_initialized))
    print('-> Precision and recall in classes identification:{:.2%}, {:.2%}, {:.2%}'.format(cluster_precision,
                                                                                            cluster_recall,
                                                                                            F1_clusters))

def Query_ReIdentification(config):
    """
    Traditional re-identification process
    """
    # Directories for loading data
    abs_path = str(pathlib.Path().absolute())
    dir_evaluation_files = (abs_path + '/evaluation/' + config.name_test +
                            '/' + str(pathlib.Path(config.iteration).stem))
    dir_final_models = dir_evaluation_files + '/final_appearance_models'
    dir_features = os.path.join(abs_path, config.dir_features.replace('./', ''))

    # Save query evaluation sampled
    dir_save_query_eval = dir_evaluation_files + '/query_eval'
    if not os.path.exists(dir_save_query_eval):
        os.makedirs(dir_save_query_eval)

    # Load queries
    query_info, data_info, total_imgs = DataLoader('DukeMTMC-Video', 'individualMetrics',
                                                   path_saveFeat=dir_features + '/')

    print('*QUERY EVALUATION:')
    print('---------------------')

    print('Extracting information from gallery...')
    gf, g_pids, g_camids = extract_info_gallery(dir_final_models)
    print('Extracting information from query set...')
    qf, q_pids, q_camids = extract_info_query(query_info)

    # Get batch_size
    n = 70     # Number of batches
    batch_size = np.int0(len(qf) / n)
    for i in range(n):
        init = i * batch_size
        if i < n:
            end = (i + 1) * batch_size
        else:
            end = len(qf)

        qf_batch = qf[init:end]
        q_pids_batch = q_pids[init:end]
        q_camid_batch = q_camids[init:end]
        print('Batch {}: init {} end {}'.format(i, init, end))

        # Obtain distance matrix
        distmat = compute_distance_matrix(qf_batch, gf, 'cosine')
        distmat = distmat.numpy()

        # Delete patches with same ID same camera
        cmc, AP, num_valid_q = evaluate_rank(
            distmat,
            q_pids_batch,
            g_pids,
            q_camid_batch,
            g_camids,
            video=True
        )

        results = {'cmc': cmc,
                   'AP': AP,
                   'numValid': num_valid_q}

        index = str(i)
        index = index.zfill(4)
        name_file = dir_save_query_eval + '/' + index + '.json'
        with open(name_file, 'w') as outfile:
            json.dump(results, outfile)
    ComputeFinalMetrics(dir_save_query_eval)


def ComputeFinalMetrics(dir_eval):
    """
    Compute the final metrics of the re-identification process
    """
    all_cmc, all_AP = [], []
    num_valid = 0.

    name_files = sorted([f for f in os.listdir(dir_eval)])
    for name_file in name_files:
        with open(dir_eval + '/' + name_file) as f:
            data = json.load(f)
        # Unify all data in diferent files in one unique list
        all_cmc = getDataList(data['cmc'], all_cmc)
        all_AP = getDataList(data['AP'], all_AP)
        num_valid = num_valid + data['numValid']
    all_cmc = np.array(all_cmc).astype(np.float32)
    cmc = all_cmc.sum(0) / num_valid
    mAP = np.mean(all_AP)

    ranks = [1, 5]
    print('** Results **')
    print('mAP: {:.1%}'.format(mAP))
    print('CMC curve')
    for r in ranks:
        print('Rank-{:<3}: {:.1%}'.format(r, cmc[r - 1]))