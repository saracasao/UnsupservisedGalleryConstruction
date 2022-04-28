import numpy as np
import torch
import warnings
from scipy.spatial import distance

def evaluate_py(distmat,g_pids,q_pids, all_cmc, all_AP, num_valid_q):
    max_rank = 10
    indices = np.argsort(distmat, axis=1)[0]
    matches = (g_pids[indices] == q_pids)#[:, np.newaxis])#.astype(np.int32))
    matches = torch.from_numpy(np.array(matches)) #ADD
    matches = matches.gt(0).to(torch.int32)

    # compute cmc curve
    raw_cmc = matches # binary vector, positions with value 1 are correct matches
    raw_cmc = raw_cmc.numpy()
    cmc = raw_cmc.cumsum()
    cmc[cmc > 1] = 1   
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

    return all_cmc, all_AP, num_valid_q

def evaluateRank(distmat, q_pids, g_pids, max_rank, video = False):
    num_q, num_g = distmat.shape
    # print('num_q',num_q)
    if num_g < max_rank:
        max_rank = num_g
        print(
            'Note: number of gallery samples is quite small, got {}'.
            format(num_g)
        )
    
    indices = np.argsort(distmat, axis=1) #se ordenan de menor a mayor 
    matches = (g_pids[indices] == q_pids[:, np.newaxis].astype(np.int32)) #se genera un vector de 1 y 0 ordenado con los indices
    matches = torch.from_numpy(np.array(matches)) #ADD
    matches = matches.gt(0).to(torch.int32)

    # compute cmc curve for each query
    all_cmc = []
    all_AP = []
    num_valid_q = 0. # number of valid query

    for q_idx in range(num_q):
        # compute cmc curve
        raw_cmc = matches[q_idx] # binary vector, positions with value 1 are correct matches
        raw_cmc = raw_cmc.numpy()
        if not np.any(raw_cmc):
            # this condition is true when query identity does not appear in gallery
            continue

        cmc = raw_cmc.cumsum() #suma acumulada
        cmc[cmc > 1] = 1 #donde haya un número mayor de 1 se define 1
        
        #ADD
        if video:
            cmc = cmc.tolist()

        all_cmc.append(cmc[:max_rank]) #se guarda los max_rank primeros numeros 
        num_valid_q += 1.

        # compute average precision
        # reference: https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Average_precision
        num_rel = raw_cmc.sum()
        tmp_cmc = raw_cmc.cumsum()
        tmp_cmc = [x / (i+1.) for i, x in enumerate(tmp_cmc)]
        tmp_cmc = np.asarray(tmp_cmc) * raw_cmc
        AP = tmp_cmc.sum() / num_rel
        all_AP.append(AP)

    # assert num_valid_q > 0, 'Error: all query identities do not appear in gallery'
    
    #ADD
    if video:
        return all_cmc, all_AP, num_valid_q
    else: 
        all_cmc = np.asarray(all_cmc).astype(np.float32)
        all_cmc = all_cmc.sum(0) / num_valid_q
        mAP = np.mean(all_AP)
        
    return all_cmc, mAP


