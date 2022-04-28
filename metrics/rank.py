from __future__ import division, print_function, absolute_import
import numpy as np
import warnings
import torch
from collections import defaultdict

def eval_market1501(distmat, q_pids, g_pids, q_camids, g_camids, max_rank, video):
    """Evaluation with market1501 metric
    Key: for each query identity, its gallery images from the same camera view are discarded.
    """
    num_q, num_g = distmat.shape
    # print('num_q',num_q)
    if num_g < max_rank:
        max_rank = num_g
        print(
            'Note: number of gallery samples is quite small, got {}'.
            format(num_g)
        )

    indices = np.argsort(distmat, axis=1) 
    matches = (g_pids[indices] == q_pids[:, np.newaxis])
    matches = torch.from_numpy(np.array(matches))
    matches = matches.gt(0).to(torch.int32)

    # compute cmc curve for each query
    all_cmc = []
    all_AP = []
    num_valid_q = 0. # number of valid query

    for q_idx in range(num_q):
        # get query pid and camid
        q_pid = q_pids[q_idx]
        q_camid = q_camids[q_idx]
        # remove gallery samples that have the same pid and camid with query
        order = indices[q_idx]
        remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)
        keep = np.invert(remove)
        # compute cmc curve
        raw_cmc = matches[q_idx][keep] # binary vector, positions with value 1 are correct matches
        raw_cmc = raw_cmc.numpy()
        if not np.any(raw_cmc):
            # this condition is true when query identity does not appear in gallery
            continue
        
        cmc = raw_cmc.cumsum()
        cmc[cmc > 1] = 1
        
        if video:
            cmc = cmc.tolist()

        all_cmc.append(cmc[:max_rank])
        num_valid_q += 1.

        # compute average precision
        # reference: https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Average_precision
        num_rel = raw_cmc.sum()
        tmp_cmc = raw_cmc.cumsum()
        tmp_cmc = [x / (i+1.) for i, x in enumerate(tmp_cmc)]
        tmp_cmc = np.asarray(tmp_cmc) * raw_cmc
        AP = tmp_cmc.sum() / num_rel
        all_AP.append(AP)

    assert num_valid_q > 0, 'Error: all query identities do not appear in gallery'
    
    if video:
        return all_cmc, all_AP, num_valid_q
    else: 
        all_cmc = np.asarray(all_cmc).astype(np.float32)
        all_cmc = all_cmc.sum(0) / num_valid_q
        mAP = np.mean(all_AP)
        
        return all_cmc, mAP


def evaluate_rank(
    distmat,
    q_pids,
    g_pids,
    q_camids,
    g_camids,
    video = False,
    max_rank=50,
):
    """Evaluates CMC rank.
    Parameters:
    -----------
        distmat (numpy.ndarray): distance matrix of shape (num_query, num_gallery).
        q_pids (numpy.ndarray): 1-D array containing person identities
            of each query instance.
        g_pids (numpy.ndarray): 1-D array containing person identities
            of each gallery instance.
        q_camids (numpy.ndarray): 1-D array containing camera views under
            which each query instance is captured.
        g_camids (numpy.ndarray): 1-D array containing camera views under
            which each gallery instance is captured.
        max_rank (int, optional): maximum CMC rank to be computed. Default is 50.
        video (bool): if True return the vector of each metric
                      if False return the final evaluation 
    """
    return eval_market1501(
            distmat, q_pids, g_pids, q_camids, g_camids, max_rank, video
        )
