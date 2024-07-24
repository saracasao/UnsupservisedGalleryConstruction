import numpy as np
import random
import torch

from numpy.linalg.linalg import norm
from Utils import get_Hmodel, get_IntraClustersDistances


def merge_new_model(model, model_index,  config, num_models, evaluation, new_model_samples = None):
    algorithm = config.data_selection_algorithm
    
    if algorithm == 'CPR':
        model, evaluation = add_new_model_CPR(model, model_index, config, num_models, evaluation, new_model_samples)
    elif algorithm == 'IOM':                                                     
        model, evaluation = add_new_model_IOM(model, config, evaluation, new_model_samples)
    elif algorithm == 'ExStream':                                                     
        model, evaluation = add_new_model_ExStream(model, config, evaluation, new_model_samples)
    elif algorithm == 'Random':                                            
        model, evaluation = add_new_model_Random(model, config, evaluation, new_model_samples)
    elif algorithm == 'Temporal':                      
        model, evaluation = add_new_model_Temporal(model, config, evaluation, new_model_samples)
    else: 
        assert 'METHOD DEFINED FOR DATA SELECION {} DOES NOT EXIST'.format(algorithm)
    return model, evaluation


def add_new_model_CPR(model, model_index, config, num_models, evaluation, new_model_samples = None): 
    """
    Data selection process for model adapting
    Parameters:
    -----------
        model (AppearanceModel() Class): model to update
        query (Sample() Class): query to add to the model 
        memory_budget (int): limited memory for each identity
    """       
    memory_budget = config.memory_budget
    # Parameters of the proposed approach
    gamma = (-1)*np.log2(1/num_models)
    alpha1 = config.alpha
    alpha2 = gamma*(1-alpha1)
    
    # Add new samples
    init_samples = model.samples.copy()
    final_samples = model.samples   

    # Check memory budget
    if len(final_samples) > memory_budget:   
        # Add centroid dist of the new query
        inter_cluster_dist = model.cluster_distances

        # Extract the diversity of each sample of the model
        D = get_IntraClustersDistances(final_samples)
        
        # Extract the uncertainty of the class model for each remove point
        model_size = len(final_samples)
        while model_size > memory_budget: 
            assert D.shape[0] == D.shape[1]
            D_min = D.copy()
            np.fill_diagonal(D_min,0)           
            D_min = np.where(D_min>0, D_min, np.inf)
            D_min = D_min.min(axis=1) 
    
            features = [s.feat for s in final_samples]
            H = []
            for ind in range(len(features)):  
                # Copy
                features_aux = features.copy()
                samples_aux = final_samples.copy()
                int_clusterD_aux = inter_cluster_dist.copy()    
    
                # Select data
                features_aux.pop(ind)
                samples_aux.pop(ind)
                int_clusterD_aux = np.delete(int_clusterD_aux, ind, axis = 0)  
    
                H_model = get_Hmodel(model_index, features_aux, samples_aux, int_clusterD_aux, num_models) 
                H.append(np.sum(H_model))
            H = np.array(H)
    
            # Compute optimization function and get the worst data
            opt = alpha1*H + alpha2*D_min
            minimum = np.where(opt == np.amin(opt))[0][0]
            
            # Delete selected sample
            final_samples.pop(minimum)
            D = np.delete(D, minimum, axis = 0)
            D = np.delete(D, minimum, axis = 1)
            inter_cluster_dist = np.delete(inter_cluster_dist, minimum, axis = 0)
            
            # Reevaluate model size
            model_size = len(final_samples)
        assert inter_cluster_dist.shape[0] == len(final_samples) and inter_cluster_dist.shape[1] == num_models 
        # Update model
        model.samples = final_samples
        model.cluster_distances = inter_cluster_dist
        model.n_newModel += 1
        
    # If the memory budget is not exceed
    else:
        model.n_newModel += 1     
    
    assert len(model.samples) <= memory_budget
    
    # Evaluation
    if new_model_samples is not None:
        evaluation.eval_model_merged(model, new_model_samples, final_samples)
    else:
        evaluation.eval_new_model(model, model.samples)
    return model, evaluation


def add_new_model_IOM(model, config, evaluation, new_model_samples = None):
    """
    Delete one of the closest information and every 5 updates the farest data
    """
    memory_budget = config.memory_budget
    NF_CLEAN = 5
    ind = 0
    
    init_samples = model.samples.copy()
    final_samples = model.samples    

    features = [c.feat for c in final_samples]
    while len(final_samples) > memory_budget:
        if ind % NF_CLEAN == 0:
            scores = [c.score for c in final_samples]
            i = scores.index(min(scores))
            final_samples.pop(i)
        else: 
            ind += 1
            Total = np.array(features)
            R = cosine_matrix_simi(Total)
            closest = np.where(R == np.amin(R[np.nonzero(R)]))[0][:]
            farest = np.where(R == np.amax(R[np.nonzero(R)]))[0][:]
            if final_samples[closest[0]].score >= final_samples[closest[1]].score:
                if farest[0] not in closest:
                    final_samples[farest[0]].score -= 1
                if farest[1] not in closest:
                    final_samples[farest[1]].score -= 1
                final_samples[closest[0]].score += 1
                final_samples.pop(closest[1])
                i = closest[1]
            else:
                if farest[0] not in closest:
                    final_samples[farest[0]].score -= 1
                if farest[1] not in closest:
                    final_samples[farest[1]].score -= 1
                final_samples[closest[1]].score +=1
                final_samples.pop(closest[0])
                i = closest[0]
                
    assert len(final_samples) <= memory_budget   
    
    model.n_newModel += 1
    model.samples = final_samples
    
    if new_model_samples is not None:
        evaluation.eval_model_merged(model, new_model_samples, final_samples)
    else:
        evaluation.eval_new_model(model, model.samples)

    return model, evaluation


def cosine_matrix_simi(M):
    # dot products of rows against themselves
    DotProducts = M.dot(M.T)

    # kronecker product of row norms
    NormKronecker = np.array([norm(M, axis=1)]) * np.array([norm(M, axis=1)]).T

    CosineSimilarity = DotProducts / NormKronecker
    CosineDistance = 1 - CosineSimilarity
    np.fill_diagonal(CosineDistance,0)
    return CosineDistance


def add_new_model_Random(model, config, evaluation, new_model_samples = None):
    """
    Delete a random component when the memory budget is achieve
    """
    memory_budget = config.memory_budget
    
    init_samples = model.samples.copy()
    final_samples = model.samples  
    
    while len(final_samples) > memory_budget:
        to_delete = random.randint(0,len(final_samples)-1)
        final_samples.pop(to_delete)
        
    assert len(final_samples) <= memory_budget
    
    model.n_newModel += 1
    model.samples = final_samples
    if new_model_samples is not None:
        evaluation.eval_model_merged(model, new_model_samples, final_samples)
    else:
        evaluation.eval_new_model(model, model.samples)

    return model, evaluation


def add_new_model_Temporal(model, config, evaluation, new_model_samples = None):
    """
    Uniform sampling of the gallery, every N updates of the model the oldest sample is delete
    """
    memory_budget = config.memory_budget

    init_samples = model.samples.copy()
    final_samples = model.samples  
            
    while len(final_samples) > memory_budget:
        final_samples.pop(0)
    
    assert len(final_samples) <= memory_budget
    
    model.n_newModel += 1
    model.samples = final_samples
    if new_model_samples is not None:
        evaluation.eval_model_merged(model, new_model_samples, final_samples)
    else:
        evaluation.eval_new_model(model, model.samples)
        
    return model, evaluation


def add_new_model_ExStream(model, config, evaluation, new_model_samples = None):
    """
    If using class buffers, store points in class buffer until full. Once class buffer is full and new point
    arrives, merge two closest samples from that class and store new point.
    :param x: a single data point (PyTorch Tensor)
    :param y: a single label for data point (PyTorch Tensor)
    :return:
    """
    memory_budget = config.memory_budget
    
    init_samples = model.samples
    final_samples = []
    
    i = 0
    add_sample = True
    while add_sample:
        new_sample = init_samples[i]
        new_sample.score = 1
        final_samples.append(new_sample)        
        i += 1
        
        if len(final_samples) == memory_budget or len(final_samples) == len(init_samples):
            add_sample = False
    
    out_of_gallery = [s for s in init_samples if s not in final_samples]
    for new_sample in out_of_gallery:
        # class buffer is full, merge closest clusters from current class
        scores = [s.score for s in final_samples]
        features = [s.feat for s in final_samples]
        prct_joints = [s.perct_key_points for s in final_samples]
        
        # weighted average of closest clusters
        X = torch.Tensor(features).cuda(0) 
        idx0, idx1 = l2_dist_metric(X)
        
        pt1 = X[idx0, :]
        pt2 = X[idx1, :]
        w1 = scores[idx0]
        w2 = scores[idx1]
        pj1 = prct_joints[idx0]
        pj2 = prct_joints[idx1]
        merged_pt = ((pt1 * w1 + pt2 * w2) / (w1 + w2)).cpu()
    
        # store new sample at idx0
        new_sample.score = 1
        final_samples[idx0] = new_sample
    
        # store merged cluster at idx1
        final_samples[idx1].feat = merged_pt.numpy()
        final_samples[idx1].score = w1 + w2
        final_samples[idx1].perct_key_points = (pj1 + pj2) / 2
        
    assert len(final_samples) <= memory_budget 
    
    model.n_newModel += 1
    model.samples = final_samples
    
    if new_model_samples is not None:
        evaluation.eval_model_merged(model, new_model_samples, new_model_samples)
    else:
        evaluation.eval_new_model(model, init_samples)
    return model, evaluation


def l2_dist_metric(H):
    """
    Given an array of data, compute the indices of the two closest samples.
    :param H: an Nxd array of data (PyTorch Tensor)
    :return: the two indices of the closest samples
    """
    with torch.no_grad():
        M, d = H.shape
        H2 = torch.reshape(H, (M, 1, d))  # reshaping for broadcasting
        inside = H2 - H
        square_sub = torch.mul(inside, inside)  # square all elements
        psi = torch.sum(square_sub, dim=2)  # capacity x batch_size

        # infinity on diagonal
        mb = psi.shape[0]
        diag_vec = torch.ones(mb).cuda(0) * np.inf
        mask = torch.diag(torch.ones_like(diag_vec).cuda(0))
        psi = mask * torch.diag(diag_vec) + (1. - mask) * psi

        # grab indices
        idx = torch.argmin(psi)
        idx_row = torch.div(idx, mb, rounding_mode='trunc')
        idx_col = idx % mb
    return torch.min(idx_row, idx_col), torch.max(idx_row, idx_col)