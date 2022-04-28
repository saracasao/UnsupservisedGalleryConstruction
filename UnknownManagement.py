import numpy as np
import sys

from scipy import stats
from scipy.spatial import distance
from sklearn.cluster import DBSCAN
from ComponentsGallery import AppearanceModel
from Utils import get_Centroid, get_DiversityModel, update_inter_cluster_distance, get_Pmodel, JS_divergence, get_wCentroid
from ModelUpdateMerge import merge_new_model

def candidatesTOmodels(gallery, config):
    candidatesTOmodels = gallery.candidatesTOmodel
    models = [m for m in candidatesTOmodels if len(m.samples) >= config.min_size]
    if len(models) >= config.min_models:
        gallery.unsupervisedInitialization(models, config)
        initialization = True
    else:
        initialization = False
    return gallery, initialization

def InitCandidateModel(gallery, query, config):
    if query.gt != 0:
        new_model = AppearanceModel(query.gt)
        new_model.samples.append(query)

        new_model.diversity = 0
        new_model.centroid = query.feat
        new_model.ref_updateModels = (-1)*np.ones(1)
        new_model.cluster_distances = np.zeros((len(new_model.samples), 1))
         
        gallery.candidatesTOmodel.append(new_model)           
        
    return gallery

def Unknown_Analysis(gallery, query, evaluation, config):
    person, evaluation = check_person(query, evaluation)
    if person:
        gallery.add_unknown_sample(query)
        
    #If enough unkown data ->evaluation of unknown data
    if gallery.batch_unknown_samples >= gallery.size_to_cluster:
        gallery.batch_unknown_samples = 0
        gallery, evaluation = unknown_samples_management(gallery, evaluation, config)
    return gallery, evaluation
    
def check_person(query, evaluation):
    #Save name query to check after
    _, nameSample = query.get_name()
    perct_joints = query.perct_key_points

    if perct_joints >= 0.5:
        person = True
        keep_query = query.get_name()
        evaluation.personName.append(keep_query)
    else:
        person = False
        discarded_query = query.get_name()
        evaluation.noiseName.append(discarded_query)
    return person, evaluation 

def person_exist(query, evaluation):
    #Save name query to check after
    _, nameSample = query.get_name()
    perct_joints = query.perct_key_points

    if perct_joints > 0:
        person = True
        keep_query = query.get_name()
        evaluation.personName.append(keep_query)
    else:
        person = False
        discarded_query = query.get_name()
        evaluation.noiseName.append(discarded_query)
    return person, evaluation 

def unknown_samples_management(gallery, evaluation, config):
    min_size = config.min_size
    
    unknown_samples = gallery.unknown_samples
    features = [u.feat for u in unknown_samples]
    
    X = np.array(features)
    clustering = DBSCAN(eps=0.1, min_samples=2, metric = 'cosine').fit(X)
    
    prediction  = clustering.labels_
    labels, counts = np.unique(prediction, return_counts = True)
    
    #Check if any of the clusters has enough size
    counts_idx = np.where(counts >= min_size)
    labels_new_models = labels[counts_idx]
    
    trash_idx = np.where(prediction == -1)[0]
    trash_samples = [u for i,u in enumerate(unknown_samples) if i in trash_idx]

    for l in labels_new_models:
        selected_samples = [u for i, u in enumerate(unknown_samples) if prediction[i] == l]

        new_model = AppearanceModel(None)                
        new_model.samples = selected_samples
        if config.merge_distance == 'JenshenShannon':
            gallery, evaluation = new_class_identification_JS(gallery, new_model, evaluation, config)
        elif config.merge_distance == 'cosine':
            gallery, evaluation = new_class_identification_COSINE(gallery, new_model, evaluation, config)
        else: 
            sys.exit('Merge distance does not exist')
    gallery = clean_trash(gallery, trash_samples)

    return gallery, evaluation

def new_class_identification_COSINE(gallery, new_model, evaluation, config):
    k_nearest = 3
    
    app_models = gallery.models
    new_model_samples = new_model.samples
    new_model_feat = [s.feat for s in new_model_samples]

    centroids_gallery = [m.centroid for m in app_models]
    
    #Select models candidates -> 3 with the closest centroids
    if config.merge_candidates_method == 'NearestCentroids':          
        centroid_model = get_wCentroid(new_model_samples)
        candidates_model, idx_candidate_models = get_candidates_models_centroids(app_models, centroids_gallery, centroid_model, k_nearest)
    elif config.merge_candidates_method == 'Moda_nearest':
        new_model_D = distance.cdist(new_model_feat, centroids_gallery, 'cosine')
        candidates_model, idx_candidate_models = get_candidates_models_MODA(app_models, new_model_D, k_nearest)
    elif config.merge_candidates_method == 'all':
        candidates_model, idx_candidate_models = get_candidates_models_all(app_models)

    cosine_min = []
    for i, cand_m in enumerate(candidates_model):
        cand_samples = cand_m.samples
        cand_feat = [s.feat for s in cand_samples]
        cosine_dist = distance.cdist(new_model_feat, cand_feat,'cosine')
        cosine_min.append(np.amin(cosine_dist))
        
    similar_value = min(cosine_min)
    if similar_value < 0.1:
        idxL_most_similar_model = cosine_min.index(similar_value)
        idxG_most_similar_model = idx_candidate_models[idxL_most_similar_model]
        
        modelTOmerge = candidates_model[idxL_most_similar_model]
        gallery, evaluation = merged_clusters(gallery, new_model, modelTOmerge, idxG_most_similar_model, config, evaluation)
    else: 
        gallery, evaluation = init_new_cluster(gallery, new_model, config, evaluation)
    return gallery, evaluation
  
def merged_clusters(gallery, new_model, modelTOmerge, index_model, config, evaluation):
    #Adapt new model inter cluster dist in function of the merge distance selected
    new_model_samples = new_model.samples 
    if config.merge_distance == 'JenshenShannon':    
        new_model_distmat = np.delete(new_model.cluster_distances, -1, 1) 
    elif config.merge_distance == 'cosine':
        centroids_gallery = [m.centroid for m in gallery.models]  
        new_model_feats = [s.feat for s in new_model_samples]
        new_model_distmat = distance.cdist(new_model_feats, centroids_gallery, 'cosine')
        
    num_models = len(gallery.models)
    
    #Update data of the selected model to merge
    if config.use_of_distmat:
        inter_cluster_dist = modelTOmerge.cluster_distances
        inter_cluster_dist, n_updates = update_inter_cluster_distance(gallery.models, modelTOmerge, inter_cluster_dist)
    
        #Unify both models
        final_inter_cluster_dist = np.append(inter_cluster_dist, new_model_distmat, axis = 0)

        modelTOmerge.cluster_distances = final_inter_cluster_dist
        modelTOmerge.ref_updateModels = n_updates
        assert modelTOmerge.cluster_distances.shape[0] == len(modelTOmerge.samples + new_model.samples) and modelTOmerge.cluster_distances.shape[1] == num_models 
      
    #Update the model 
    modelTOmerge.samples = modelTOmerge.samples + new_model.samples
 
    #Select the final samples to compose the cluster    
    final_model, evaluation = merge_new_model(modelTOmerge, index_model, config, num_models, evaluation, new_model_samples = new_model.samples)
    
    #Update paremters of the final model
    final_model.centroid = get_wCentroid(final_model.samples)
    final_model.diversity = get_DiversityModel(final_model.samples)
    gallery.models[index_model] = final_model 
    
    #Update gallery 
    unknown_s = gallery.unknown_samples
    unknown_s = [u for u in unknown_s if u not in new_model_samples]
    gallery.unknown_samples = unknown_s
    
    #Evaluation of the prediction class of the queries that compose the new_model
    evaluation.n_merge += 1
    for s in new_model_samples:
        evaluation.pred_merge.append(int(modelTOmerge.label))
        evaluation.id_merge.append(int(modelTOmerge.identity))
        evaluation.gt_merge.append(int(s.gt))
    return gallery, evaluation

def init_new_cluster(gallery, new_model, config, evaluation):
    min_size = config.min_size
    
    model_samples = new_model.samples
    new_model_samples = [s for i, s in enumerate(model_samples) if s.gt != 0]  
    new_model_labels = [s.gt for s in new_model_samples]
    moda_label = stats.mode(new_model_labels, axis = None)[0]
    
    noise_model_samples = [s for i, s in enumerate(model_samples) if s.gt == 0]    

    #Initialization new model  
    if len(new_model_samples) >= min_size:
        num_models = len(gallery.models)
        
        #Initialization of the new model
        if type(moda_label) is np.ndarray:
            moda_label = moda_label[0]
        new_model.label = moda_label
        new_model.samples = new_model_samples
        new_model.centroid = get_wCentroid(new_model_samples)
        new_model.ref_updateModels = (-1)*np.ones(num_models + 1)
        new_model.cluster_distances = np.zeros((len(new_model.samples), num_models + 1))
        
        #Add model to the gallery and update parameters
        gallery.add_new_model(new_model, config)  
        app_models = gallery.models
        new_model.cluster_distances, new_model.ref_updateModels = update_inter_cluster_distance(app_models, new_model, new_model.cluster_distances)
        assert new_model.cluster_distances.shape[0] == len(new_model.samples) and new_model.cluster_distances.shape[1] == len(gallery.models) 
        
        #Selection of the final samples to compose the new model
        index_model = app_models.index(new_model)
        final_model, evaluation = merge_new_model(new_model, index_model, config, len(gallery.models), evaluation)

        final_model.centroid = get_wCentroid(final_model.samples)
        final_model.diversity = get_DiversityModel(final_model.samples)
        
        #Update gallery
        gallery.models[index_model] = final_model
        gallery.unknown_samples = [u for u in gallery.unknown_samples if u not in noise_model_samples]
        evaluation.cluster_labels.append(int(final_model.label))
        
        #Query prediction evaluation
        for s in new_model_samples:
            evaluation.pred_init.append(int(final_model.label))
            evaluation.id_init.append(int(final_model.identity))
            evaluation.gt_init.append(int(s.gt))
    
    return gallery, evaluation

def get_candidates_models_centroids(app_models, centroids_gallery, new_model_centroid, k_nearest):
    #Select the three models with highes probability through mode of the samples
    inter_clusters_dist = distance.cdist([new_model_centroid], centroids_gallery,'cosine')[0]
    idx_closest_cluster = np.argsort(inter_clusters_dist)
    idx_candidate_models = np.sort(idx_closest_cluster[0:k_nearest])

    #Update distmat of each candidate cluster including the new model
    candidates_model = [m for i,m in enumerate(app_models) if i in idx_candidate_models]

    return candidates_model, idx_candidate_models

def get_candidates_models_MODA(app_models, new_model_distmat, k_nearest):
    #Select the three models with highes probability through mode of the samples
    min_dist_list = np.argsort(new_model_distmat, axis = 1)
    k_min = min_dist_list[:,0:k_nearest]

    idx_models, counter = np.unique(k_min, return_counts = True)
    idx_max_counter = np.argsort(counter)[::-1][:k_nearest]
    idx_candidate_models = idx_models[idx_max_counter]
    idx_candidate_models = np.sort(idx_candidate_models)

    #Update distmat of each candidate cluster including the new model
    candidates_model = [m for i,m in enumerate(app_models) if i in idx_candidate_models]
    return candidates_model, idx_candidate_models

def get_candidates_models_all(app_models):
    num_models = len(app_models)
    idx_candidate_models = list(range(num_models))
    
    return app_models, idx_candidate_models

def clean_trash(gallery, trash_samples):
    unknown_samples = gallery.unknown_samples
    unknown_samples = [u for u in unknown_samples if u not in trash_samples]
    
    gallery.unknown_samples = unknown_samples

    return gallery
    
#############################################################################################################    

def get_candidates_models_absP(app_models, new_model_P, k_nearest):
    # print('MODELS WITH HIGEST P')
    #Select the three models with highes probability through mode of the samples
    P_candidates = np.delete(new_model_P, -1, 1)
    idx_arg_max = np.argmax(P_candidates, axis = 1)
    # print('idx max p', idx_arg_max)
    
    idx_models, counter = np.unique(idx_arg_max, return_counts = True)
    idx_max_counter = np.argsort(counter)[::-1][:k_nearest]
    idx_candidate_models = idx_models[idx_max_counter]
    idx_candidate_models = np.sort(idx_candidate_models)
    
    #Update distmat of each candidate cluster including the new model
    candidates_model = [m for i,m in enumerate(app_models) if i in idx_candidate_models]
    label_cand_m = [m.label for m in candidates_model]
    return candidates_model, idx_candidate_models


def get_candidates_models_H(app_models, new_model_distmat, k_nearest):
    # print('MINIMUM H')
    # print('shape distmat', new_model_distmat.shape)
    new_model_distmat = np.delete(new_model_distmat,-1, 1)
    new_model_P = get_Pmodel(new_model_distmat)  
    
    H = (-1)*np.sum(new_model_P*np.log2(new_model_P), axis = 1)
    # print('SHAPE H', H.shape)
    idx_min_H = np.argsort(H)[:k_nearest]

    new_model_P = np.delete(new_model_P, -1, 1)
    #Select the three samples with lowest uncertainty
    most_certains_P = new_model_P[idx_min_H]
    # print('matrix most certains P',most_certains_P.shape)
    #Obtain the index of the maximum probability
    idx_max_prob = np.argmax(most_certains_P, axis = 1)
    # print('idx most certains P',idx_max_prob.shape)
    idx_candidate_models = np.sort(idx_max_prob)
    # print('idx candidates', idx_candidate_models)
    #Update distmat of each candidate cluster including the new model
    candidates_model = [m for i,m in enumerate(app_models) if i in idx_candidate_models]
    label_cand_m = [m.label for m in candidates_model]
    # print('label model candidates',label_cand_m)
    return candidates_model, idx_candidate_models

def new_class_identification_JS(gallery, new_model, evaluation, config):
    k_nearest = 3
    
    app_models = gallery.models
    new_model_samples = new_model.samples
    
    #Select models candidates -> 3 with the closest centroids
    centroid_model = get_wCentroid(new_model_samples)
    centroids_gallery = [m.centroid for m in app_models]
    centroids_gallery.append(centroid_model)

    #Obtain probability distribution of the samples belong to the new model
    new_model_feats = [s.feat for s in new_model_samples]
    new_model_distmat = distance.cdist(new_model_feats, centroids_gallery, 'cosine')
    new_model.P = get_Pmodel(new_model_distmat) 
    new_model.cluster_distances = new_model_distmat
        
    if config.merge_candidates_method == 'NearestCentroids':
        centroids_candidates = centroids_gallery.copy()
        centroids_candidates.pop(-1)
        candidates_model, idx_candidate_models = get_candidates_models_centroids(app_models, centroids_candidates, centroid_model, k_nearest)
    elif config.merge_candidates_method == 'Moda_nearest':
        new_model_D =  np.delete(new_model_distmat, -1, 1)
        candidates_model, idx_candidate_models = get_candidates_models_MODA(app_models, new_model_D, k_nearest)
    elif config.merge_candidates_method == 'all':
        candidates_model, idx_candidate_models = get_candidates_models_all(app_models)
    else:
        sys.exit('Candidate method to merge does not exist')
        
    JS_min, labels = [], []
    for i, cand_m in enumerate(candidates_model):
        #Update distmat of the cluster candidate
        inter_cluster_dist = cand_m.cluster_distances
        inter_cluster_dist, n_updates = update_inter_cluster_distance(app_models, cand_m, inter_cluster_dist)
        
        #Compute the distance to the new cluster
        cand_m_samples = cand_m.samples
        features = [s.feat for s in cand_m_samples]
        dist_new_model = distance.cdist(features, [centroid_model], 'cosine')
        inter_cluster_dist = np.append(inter_cluster_dist, dist_new_model, axis = 1)

        #Get probability distribution 
        P_model = get_Pmodel(inter_cluster_dist)
        JS = JS_divergence(new_model.P, P_model)   

        JS_min.append(np.amin(JS))
        labels.append(cand_m.label)
    
    similar_value = min(JS_min)
    threshold_to_merge = config.merge_threshold
    if similar_value < threshold_to_merge:
        idxL_most_similar_model = JS_min.index(similar_value)
        idxG_most_similar_model = idx_candidate_models[idxL_most_similar_model]
        modelTOmerge = candidates_model[idxL_most_similar_model]
        
        gallery, evaluation = merged_clusters(gallery, new_model, modelTOmerge, idxG_most_similar_model, config, evaluation)
    else: 
        gallery, evaluation = init_new_cluster(gallery, new_model, config, evaluation)
    return gallery, evaluation   
# =============================================================================
# # if merged_m == 'Moda_nearest':
# #     candidates_model, idx_candidate_models = get_candidates_models(app_models, new_model.P, k_nearest)
# # elif merged_m == 'Nearest_centroids':
# #     candidates_model, idx_candidate_models = get_candidates_models_centroids(app_models, centroids_gallery, centroid_model, k_nearest)
# # elif merged_m == 'highest_P':
# #     candidates_model, idx_candidate_models = get_candidates_models_absP(app_models, new_model.P, k_nearest)
# # elif merged_m == 'min_H':
# #     candidates_model, idx_candidate_models = get_candidates_models_H(app_models, new_model_distmat, k_nearest)
# # elif merged_m == 'all':
# #      candidates_model, idx_candidate_models = get_candidates_models_all(app_models)
# # else:
# #     print('Merged method selected does not exist')
# =============================================================================

# =============================================================================
# def check_new_class_CONFIDENCE(gallery, new_model, evaluation, config):
#     app_models = gallery.models
#     new_model_samples = new_model.samples
# 
#     #Select models candidates -> 3 with the closest centroids
#     centroid_model = get_Centroid(new_model_samples)
#     centroids_gallery = [m.centroid for m in app_models]  
#     
#     D = distance.cdist([centroid_model], centroids_gallery,'cosine')
#     D = D/0.1 
#     P = (np.exp((-1)*D)/np.sum(np.exp((-1)*D)))[0]
#     assert P.shape[0] == len(app_models) and round(np.sum(P),0) == 1
#     
#     sort_index = np.argsort(P)[::-1]    
#     if P[sort_index[0]]/P[sort_index[1]] >= 2:
#         modelTOmerge = gallery.models[sort_index[0]]
#         gallery, evaluation = merged_clusters(gallery, new_model, modelTOmerge, sort_index[0], config, evaluation)
#     else: 
#         gallery, evaluation = init_new_cluster(gallery, new_model, config, evaluation)
#     return gallery, evaluation    
# =============================================================================
