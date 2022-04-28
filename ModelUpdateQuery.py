import numpy as np
import random
import torch
import cv2

from numpy.linalg.linalg import norm
from scipy.spatial import distance
from Utils import update_inter_cluster_distance, get_DiversitySamples, get_DiversityQuery, get_Hmodel, get_IntraClustersDistances

def Labeled_DataSelection(sequence, N_GT):
   """
   Remove redundant info from the sequence to comply with the number of limited labeled data
   Parameters
   ----------
       sequences (list): list of Sample() Class
       N_GT (int): number of labeled image use in the appearance model

    """
   while not len(sequence) <= N_GT:
        features = [s.feat for s in sequence]
        distances = distance.cdist(features, features, 'cosine')[0]
        closest = np.where(distances == np.amin(distances))[0]
        
        sequence.pop(closest[0])
   return sequence

def add_query_CPR(model, model_index,  query, app_models, memory_budget, evaluation, config): 
    """
    Data selection process for model adapting
    Parameters:
    -----------
        model (AppearanceModel() Class): model to update
        query (Sample() Class): query to add to the model 
        memory_budget (int): limited memory for each identity
    """
    keep = False
    num_models = len(app_models)
    
    #Parameters of the proposed approach
    gamma = (-1)*np.log2(1/num_models)
    alpha1 = config.alpha
    alpha2 = gamma*(1-alpha1)
    
    #Compute the minimum distances between the query and the current model (minimum diversity)
    samples  = model.samples    
    query_diversity = get_DiversityQuery(query, samples)

    #Check diversity criterian before add to the model 
    if query_diversity > model.diversity: 
        model.samples.append(query)
        samples = model.samples
          
        #Check memory budget
        if len(samples) > memory_budget:   
            #Add centroid dist of the new query
            new_queryDist = query.distmat
            inter_cluster_dist = model.cluster_distances
            inter_cluster_dist = np.append(inter_cluster_dist, new_queryDist, axis = 0)
            
            inter_cluster_dist, n_updates = update_inter_cluster_distance(app_models, model, inter_cluster_dist)
            
            #Extract the diversity of each sample of the model 
            D = get_DiversitySamples(samples)

            #Extract the uncertainty of the class model for each remove point                       
            H = []
            features = [s.feat for s in samples]
            for ind in range(len(features)):  
                #Copy
                features_aux = features.copy()
                samples_aux = samples.copy()
                int_clusterD_aux = inter_cluster_dist.copy()    

                #Select data
                features_aux.pop(ind)
                samples_aux.pop(ind)
                int_clusterD_aux = np.delete(int_clusterD_aux, ind, axis = 0)  

                H_model = get_Hmodel(model_index, features_aux, samples_aux, int_clusterD_aux, num_models) 
                H.append(np.sum(H_model))
            H = np.array(H)

            #Compute optimization function and get the worst data
            opt = alpha1*H + alpha2*D
            minimum = np.where(opt == np.amin(opt))[0][0]
            
            #Delete selected sample 
            samples.pop(minimum)
            inter_cluster_dist = np.delete(inter_cluster_dist, minimum, axis = 0)

            #Update model
            model.samples = samples
            model.cluster_distances = inter_cluster_dist
            model.ref_updateModels = n_updates 
            
            if minimum != len(features) - 1:
                model, keep = evaluation.eval_query_selected(model, query)

            assert len(model.samples) == memory_budget 
        #If the memory budget is not exceed
        else:
            model.cluster_distances = np.append(model.cluster_distances, query.distmat, axis = 0)
            model, keep = evaluation.eval_query_selected(model, query)
    
    evaluation.eval_noise_discarded(query, keep)
    
    return model, evaluation 

def add_query_IOM(model, query, memory_budget, evaluation):
    """
    Delete one of the closest information and every 5 updates the farest data
    """
    keep = False
    NF_CLEAN = 5
    
    model.samples.append(query)
    model_samples = model.samples

    features = [c.feat for c in model.samples]
    if len(features) > memory_budget:
        if model.n_newModel % NF_CLEAN == 0: #OBTIENES LOS SCORES DE TODOS LOS CLUSTER DE LA CLASE (LABEL) Y TIRAS EL QUE MENOS SCORE TENGA
            scores = [c.score for c in model_samples]
            i = scores.index(min(scores))
            model_samples.pop(i)
        else: 
            Total = np.array(features)
            R = cosine_matrix_simi(Total)
            closest = np.where(R == np.amin(R[np.nonzero(R)]))[0][:]#combinación de clusters que dan la minima distancia
            farest = np.where(R == np.amax(R[np.nonzero(R)]))[0][:]#combinación de clusters que dan la maxima distancia
            if model_samples[closest[0]].score >= model_samples[closest[1]].score:
                if farest[0] not in closest:
                    model_samples[farest[0]].score -= 1
                if farest[1] not in closest:
                    model_samples[farest[1]].score -= 1
                model_samples[closest[0]].score += 1
                model_samples.pop(closest[1])
                i = closest[1]
            else:
                if farest[0] not in closest:
                    model_samples[farest[0]].score -= 1
                if farest[1] not in closest:
                    model_samples[farest[1]].score -= 1
                model_samples[closest[1]].score +=1
                model_samples.pop(closest[0])
                i = closest[0]
        #Update changes
        model.samples = model_samples
        
        if i != (len(features)-1):
            model, keep = evaluation.eval_query_selected(model, query)  
        assert len(model.samples) == memory_budget 

    else:
        model, keep = evaluation.eval_query_selected(model, query)
    
    evaluation.eval_noise_discarded(query, keep)
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

def add_query_Random(model, query, memory_budget, evaluation):
    """
    Delete a random component when the memory budget is achieve
    """
    keep = False
    model.samples.append(query)
    model_samples = model.samples
    
    if len(model_samples) > memory_budget:
        to_delete = random.randint(0,len(model_samples)-1)
        model_samples.pop(to_delete)
        
        model.samples = model_samples
        if to_delete != (len(model_samples)-1):           
            model, keep = evaluation.eval_query_selected(model, query)  
        assert len(model.samples) == memory_budget 
    else:
        model, keep = evaluation.eval_query_selected(model, query) 
    
    evaluation.eval_noise_discarded(query, keep)
    return model, evaluation

def add_query_Temporal(model, query, memory_budget, evaluation):
    """
    Uniform sampling of the gallery, every N updates of the model the oldest sample is delete
    """
    N_UPDATE = 5
    if model.n_newModel % N_UPDATE == 0:
        model.samples.append(query)
        model_samples = model.samples
        
        if len(model_samples) > memory_budget:
            model_samples.pop(0)
            assert len(model.samples) == memory_budget 
        model, keep = evaluation.eval_query_selected(model, query) 
        evaluation.eval_noise_discarded(query, keep)
        
    return model, evaluation

def add_query_ExStream(model, query, memory_budget, evaluation):
    """
    If using class buffers, store points in class buffer until full. Once class buffer is full and new point
    arrives, merge two closest samples from that class and store new point.
    :param x: a single data point (PyTorch Tensor)
    :param y: a single label for data point (PyTorch Tensor)
    :return:
    """
    model_samples = model.samples
    
    features = [s.feat for s in model_samples]
    if len(model_samples) < memory_budget:
        # class buffer not full --> store point
        query.score = 1
        model_samples.append(query)
    else:
        # class buffer is full, merge closest clusters from current class
        scores = [s.score for s in model_samples]
        prct_joints = [s.perct_key_points for s in model_samples]
        
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
        query.score = 1
        model_samples[idx0] = query

        # store merged cluster at idx1
        model_samples[idx1].feat = merged_pt.numpy()
        model_samples[idx1].score = w1 + w2
        model_samples[idx1].perct_key_points = (pj1 + pj2) / 2
        
        assert len(model.samples) == memory_budget
    model.samples = model_samples

    model, keep = evaluation.eval_query_selected(model, query) 
    evaluation.eval_noise_discarded(query, keep)

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

def add_DunnIndex(app_models, model, ind_model, query, model_size, qid, n_newModel, acc):
    samples = model.samples 
    samples.append(query)
    features = [c.feat for c in samples]
    
    n_updates = model.ref_updateModels
    cluster_distances = model.cluster_distances
    
    if len(features) > model_size:
        Dintra_cluster  = distance.cdist(features, features,'cosine')
        
        n_updates = np.array([m.n_newModel for m in app_models])
        diff_updated = n_updates - model.ref_updateModels
        DistancesTOupdate = np.where(diff_updated != 0)[0]

        min_Distances = getminDist(app_models, query)
        model.cluster_distances = np.append(model.cluster_distances,[min_Distances], axis = 0)
        cluster_distances = get_InterClusterDistances(app_models, features, model.cluster_distances, DistancesTOupdate)
        cluster_distances_s = np.delete(cluster_distances, ind_model, axis = 1)
        DunnIndex = []
        for i in range(len(features)):
            aux_maxDistances = Dintra_cluster.copy()

            aux_maxDistances = np.delete(aux_maxDistances, i, axis=0)
            aux_maxDistances = np.delete(aux_maxDistances, i, axis=1)
            
            aux_cluster_distances = cluster_distances_s.copy()
            aux_cluster_distances = np.delete(aux_cluster_distances, i, axis=0)
            
            intra_cluster = np.max(aux_maxDistances)
            inter_cluster = np.min(aux_cluster_distances)
            DunnIndex.append(inter_cluster/intra_cluster)
        indexTOdelete = np.argmax(DunnIndex)
        samples.pop(indexTOdelete)
        cluster_distances = np.delete(cluster_distances, indexTOdelete, axis = 0)
        if indexTOdelete != (len(features)-1) and cluster.gt == qid:
            n_newModel += 1
            acc.append(1)
        elif indexTOdelete != (len(features)-1) and cluster.gt != qid:
            n_newModel += 1
            acc.append(0)
    else:
        if cluster.gt == qid:
            n_newModel += 1
            acc.append(1)
        elif cluster.gt != qid:
            n_newModel += 1
            acc.append(0)
        
    return samples, n_updates, cluster_distances, n_newModel, acc