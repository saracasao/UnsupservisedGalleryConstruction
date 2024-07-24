import numpy as np

from scipy.spatial import distance
from ModelUpdateQuery import add_query_CPR, add_query_IOM, add_query_ExStream, add_query_Random, add_query_Temporal
from Utils import get_DiversityModel, get_wCentroid


def get_PseudoLabel(AppModels, query, evaluation, config):
    """
    Get pseudo-label assigned and compute uncertainty of the query
    Parameters
    ----------
        query (Sample() Class): new unlabeled query
        evaluation (Evaluator() Class): save data evaluation
    """
    # Compute distances between query and model centroids
    centroids = np.array([model.centroid for model in AppModels])
    D = distance.cdist([query.feat], centroids, 'cosine')
    
    # Compute probability distribution 
    D_t = D/0.1
    P = (np.exp((-1)*D_t)/np.sum(np.exp((-1)*D_t)))[0]
    sort_index = np.argsort(P)[::-1]
    
    query.P = P
    # Check confidence threshold
    if P[sort_index[0]]/P[sort_index[1]] >= config.confidence_threshold:
        # Obtain identity predicted
        model_predicted = AppModels[sort_index[0]]

        # Save pseudo-label predicted metrics
        evaluation.pred_clasif_kept.append(int(model_predicted.label))
        evaluation.id_clasif_kept.append(int(model_predicted.identity))
        evaluation.gt_clasif_kept.append(int(query.gt))
        
        final_model = model_predicted
    else: 
        final_model = None
    query.distmat = D
    return query, final_model, evaluation


def get_PseudoLabel_Dist(AppModels, query):
    """
    Get pseudo-label assigned and compute uncertainty of the query
    Parameters
    ----------
        query (Sample() Class): new unlabeled query
        evaluation (Evaluator() Class): save data evaluation
    """
    final_model = None
    if len(AppModels) > 0:
        #Compute distances between query and model centroids
        distmat = []
        for model in AppModels:
            model_samples = model.samples
            feat_model = [s.feat for s in model_samples]
            
            d = distance.cdist([query.feat], feat_model, 'cosine')
            distmat.append(np.amin(d))
        dist_sort = np.argsort(distmat)

        #Check confidence threshold 
        if distmat[dist_sort[0]] <= 0.1:
            #Obtain identity predicted
            final_model = AppModels[dist_sort[0]]            
    return query, final_model


def DataSelection(AppModels, query, model, config, evaluation):    
    """
    Select the appearance model (class) to update and send the info to the information selection process
    Parameters
    ----------
        AppModels (list): list of AppModels() include in the gallery 
        query (Sample() Class): query to include in the model
        model (AppModel() Class): model to which the query belongs
        evaluation (Evaluator() Class): needed to perform the evaluation of the algorithm
    """
    algorithm = config.data_selection_algorithm
    memory_budget = config.memory_budget
    
    # Select the model that corresponds to the pseudo-label assigned to the query
    model_index = AppModels.index(model)

    if algorithm == 'CPR':
        model, evaluation = add_query_CPR(model, model_index, query, AppModels, memory_budget, evaluation, config)
    elif algorithm == 'IOM':                                                     
        model, evaluation = add_query_IOM(model, query, memory_budget, evaluation)
    elif algorithm == 'ExStream':                                                     
        model, evaluation = add_query_ExStream(model, query, memory_budget, evaluation)
    elif algorithm == 'Random':                                            
        model, evaluation = add_query_Random(model, query, memory_budget, evaluation)
    elif algorithm == 'Temporal':                      
        model, evaluation = add_query_Temporal(model, query, memory_budget, evaluation)
    else: 
        assert 'METHOD DEFINED FOR DATA SELECION {} DOES NOT EXIST'.format(algorithm)
  
    model.centroid = get_wCentroid(model.samples)
    model.diversity = get_DiversityModel(model.samples)
    
    AppModels[model_index] = model
    
    return AppModels, evaluation


def DataSelection_InitStage(gallery, candidatesTOmodel, query, model, config, evaluation):
    """
    Check the appearance model size. If minimum number of samples -> new class in the gallery.
    If it is an appearance model already with the minimum size -> analysis of the samples in the class
    Parameters
    ----------
        gallery (Gallery Class): gallery created
        candidatesTOmodel (list): list of AppModels() class candidates to be classes in the actual gallery
        query (Sample() Class): query to include in the model
        evaluation (Evaluator() Class): needed to perform the evaluation of the algorithm
    """

    min_size = config.min_size
    memory_budget = config.memory_budget
    algorithm = config.data_selection_algorithm

    model_init = False
    model_index = candidatesTOmodel.index(model)

    query_dist = distance.cosine(query.feat, model.centroid)
    query.distmat = np.array([[query_dist]])
    if len(model.samples) < min_size:
        model.samples.append(query)
        model.cluster_distances = np.append(model.cluster_distances, query.distmat, axis = 0)
        if len(model.samples) == min_size:
            model_init = True
            model.identity = gallery.get_next_id()
    else:
        if algorithm == 'CPR':
            model, evaluation = add_query_CPR(model, 0, query, [model], memory_budget, evaluation, config)
        elif algorithm == 'IOM':
            model, evaluation = add_query_IOM(model, query, memory_budget, evaluation)
        elif algorithm == 'ExStream':
            model, evaluation = add_query_ExStream(model, query, memory_budget, evaluation)
        elif algorithm == 'Random':
            model, evaluation = add_query_Random(model, query, memory_budget, evaluation)
        elif algorithm == 'Temporal':
            model, evaluation = add_query_Temporal(model, query, memory_budget, evaluation)
        else:
            assert 'METHOD DEFINED FOR DATA SELECION {} DOES NOT EXIST'.format(algorithm)

    if model_init:
        evaluation.eval_new_model(model, model.samples)
        evaluation.cluster_labels.append(int(model.label))

        for s in model.samples:
            evaluation.pred_init.append(int(model.label))
            evaluation.id_init.append(int(model.identity))
            evaluation.gt_init.append(int(s.gt))

    model.centroid = get_wCentroid(model.samples)
    model.diversity = get_DiversityModel(model.samples)
    candidatesTOmodel[model_index] = model

    return candidatesTOmodel, evaluation
    