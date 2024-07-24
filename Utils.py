import numpy as np
from scipy.spatial import distance


def update_inter_cluster_distance(app_models, model, inter_cluster_dist):
    samples = model.samples
    features = [s.feat for s in samples]
    
    n_updates = np.array([m.n_newModel for m in app_models])
    diff_updates = n_updates - model.ref_updateModels
    DistancesTOupdate = np.where(diff_updates != 0)[0]
    
    inter_cluster_dist = get_InterClusterDistances(app_models, features, inter_cluster_dist, DistancesTOupdate)
    
    return inter_cluster_dist, n_updates


def get_wCentroid(samples):
    """
    Given a model of a class return the mean of the features (centroid)
    """
    features    = [s.feat for s in samples]
    pct_joints  = [s.perct_key_points for s in samples]
    w_feats = [i*j for i,j in zip(pct_joints,features)]

    num = np.sum(w_feats, axis = 0)
    den = np.sum(pct_joints)
    centroid = num/den
    
    feature_arr = np.array(features) 
    assert centroid.shape == feature_arr[0].shape
    return centroid


def get_DiversityModel(samples):
    """
    Given a model of a class return the minimum distance within the samples (diversity)
    """
    distmat = get_IntraClustersDistances(samples)
    np.fill_diagonal(distmat,0)
    diversity = np.amin(np.where(distmat>0, distmat, np.inf))

    return diversity


def get_DiversityQuery(query, samples):
    features = [s.feat for s in samples]
    query_distmat  = distance.cdist([query.feat], features,'cosine')[0]
    query_diversity = np.min(query_distmat)

    return query_diversity


def get_DiversitySamples(samples):
    distmat = get_IntraClustersDistances(samples)
    np.fill_diagonal(distmat,0)  
    distmat = np.where(distmat>0, distmat, np.inf)
    
    samples_diversity = distmat.min(axis=1)  
    return samples_diversity


def get_InterClusterDistances(app_models, features, clustersDistance, indexTOupdate):   
    if len(indexTOupdate) > 0:    
        modelsTOupdate = [m for i, m in enumerate(app_models) if i in indexTOupdate]
        centroids = [m.centroid for m in modelsTOupdate]
        distmat = distance.cdist(features, centroids,'cosine')   
        for i in range(len((indexTOupdate))):
            clustersDistance[:,indexTOupdate[i]] = distmat[:,i]      
    return clustersDistance


def get_IntraClustersDistances(samples):
    features = [s.feat for s in samples] 
    distmat  = distance.cdist(features, features,'cosine')
    
    return distmat


def get_Hmodel(model_index, features, samples, int_clusterD, num_models = None):     
    # Update values
    new_centroid = get_wCentroid(samples) 
    dist_new_centroid = distance.cdist([new_centroid], features,'cosine')
    int_clusterD[:,model_index] = dist_new_centroid
    
    P = get_Pmodel(int_clusterD, num_models)
    P = np.where(P == 0, 1, P)

    H_i = (-1)*np.sum(P*np.log2(P), axis = 1)

    assert len(H_i) == len(samples), 'Error in entrophy dimensions: len entrophy {} but len samples {}'.format(len(H_i), len(samples))
    
    idx_labeled = [i for i, s in enumerate(samples) if s.labeled == 1]
    H_i[idx_labeled] = 0
    
    return H_i


def get_Pmodel(D, num_models = None):    
    # Get Probabilies
    D = D/0.1 
    num = np.exp((-1)*D)
    den = np.sum(num, axis = 1) 
    den.shape = (D.shape[0], 1)
    P = num/den
    assert round(np.sum(P[0]), 0) == 1
    return P

