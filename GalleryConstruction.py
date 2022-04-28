import numpy as np
import cv2
import time

from QueryClassification import get_PseudoLabel, DataSelection, get_PseudoLabel_NewModels, get_PseudoLabel_Unsupervised, Unsupervised_DataSelection
from Utils import get_Centroid, get_DiversityModel, get_InterClusterDistances, get_wCentroid
from UnknownManagement import Unknown_Analysis, InitCandidateModel, candidatesTOmodels, check_person, person_exist
from ModelUpdateQuery import Labeled_DataSelection
from Evaluation import Evaluator
from ComponentsGallery import AppearanceModel, Sample

def features_extraction():
    print('not implemented')
  
def ClassesInitialization(gallery, data, ids_selected, config, evaluation):
    """
    Parameters
    ----------
        data (list): gallery info ((features_path0), ID0, camID0),(features_path0), ID0, camID1),...)
        algorithm (str): method for data selection 

    Returns
    -------
        AppModels (list): list of AppearanceModel() Class initialize with the first tracklet of each class
    """
    
    #Find labels of classes to initialize
    listIDS = [f[1] for f in data] 
    class_labels = list(set(listIDS))
    class_labels = [l for l in class_labels if l in ids_selected]
    AppModels = [AppearanceModel(f) for f in class_labels if f != 0]
    
    #Appearance class model initialization
    if len(AppModels) > 0:
        gallery, data, evaluation = SupervisedInitialization(gallery, AppModels, data, listIDS, config, evaluation)
    
    return gallery, data, evaluation

def SupervisedInitialization(gallery, AppModels, data, listIDS, config, evaluation):
    """
    Intialization of the Appearance Models with the first labeled tracklet
    """
    algorithm = config.data_selection_algorithm
    num_gt = config.num_labeled_data
    
    N = 0
    num_models = len(AppModels)
    indices = []
    for model in AppModels:
        N += 1
        sequence = []
        firstTracklet, indices = get_firstTracklet(model.label, data, listIDS, indices)
        path_features, id_gt, camid = firstTracklet
        
        for path_feat in path_features:
            feature = np.load(path_feat)[0]
            assert len(feature) > 0, 'ERROR LOADING FEATURES OF CLASS {}'.format(id_gt)
            
            #Initialization of Sample Class 
            s = Sample(feature, path_feat, camid, id_gt)
            if config.data_selection_algorithm == 'ExStream':
                s.score = 1
            sequence.append(s)  

        model.samples  = Labeled_DataSelection(sequence, num_gt)
        model.centroid = get_wCentroid(model.samples)
        
        model.diversity = get_DiversityModel(model.samples)
        model.ref_updateModels = (-1)*np.ones(num_models)
        model.cluster_distances = np.zeros((len(model.samples), num_models))
         
        gallery.add_new_labeled_model(model)
        
        #Evaluation parameters
        evaluation.cluster_labels.append(int(model.label))
        for s in model.samples:
            evaluation.id_gallery.append(model.identity) 
            evaluation.acc_Gallery.append(1)

    #Remove tracklets used during the initialization of the models
    updated_data = [i for j, i in enumerate(data) if j not in indices]
    if algorithm == 'CPR_Update':
        AppModels = init_cluster_distance(AppModels)
    
    return gallery, updated_data, evaluation

def init_cluster_distance(AppModels):
    indexTOupdate = list(range(len(AppModels)))
    for model in AppModels:
        samples = model.samples
        clustersDistance = model.cluster_distances
        
        features = [s.feat for s in samples]
        model.cluster_distances = get_InterClusterDistances(AppModels, features, clustersDistance, indexTOupdate)
    return AppModels
 
def get_firstTracklet(label, data, listIDS, indices):
    """
    Return the first tracklet of the class given by label
    """
    data_index = listIDS.index(label)
    indices.append(data_index)
    
    firstTracklet = data[data_index]
    return firstTracklet, indices

def UnsupervisedInitialization(gallery, data, config, evaluation):
    print('Number of tracklets to analize:', len(data)) 
    n, t = 0, 0  
    N_SAVE = 400    
    initialization = False
    
    data_analysed = []
    while not initialization: 
        d = data[n]
        path_feat_analyzed = []
        
        data_analysed.append(d)
        path_features, id_gt, camid = d
        for j, path_feat in enumerate(path_features):
            t += 1
            #Load feature
            feature = np.load(path_feat)[0]
            path_feat_analyzed.append(path_feat)
            
            #New query initialization as Sample() Class
            query = Sample(feature, path_feat, camid, id_gt) 
            person, evaluation = check_person(query, evaluation)
            if person:
                query, final_model = get_PseudoLabel_Unsupervised(gallery.candidatesTOmodel, query)
                #If query comply with confidence threshold
                if final_model is not None:
                    gallery.candidatesTOmodel, evaluation = Unsupervised_DataSelection(gallery, gallery.candidatesTOmodel, query, final_model, config, evaluation)
                else:
                    gallery = InitCandidateModel(gallery, query, config) 
                
                #Check if there is a minimum of models with a minimum of samples
                gallery, initialization = candidatesTOmodels(gallery, config)   
                if initialization:
                    break       
            
        ##Save metric data##
        if n%N_SAVE == 0 or initialization:
            print('SAVING METRICS AT TRACKLET',n)    
            evaluation.saveEvaluation(gallery.models, n, config)
            gallery.models = evaluation.saveEvolModels(gallery.models, t, config)
            evaluation = Evaluator()
        
        n += 1
 
    data_not_analysed = [d for d in data if d not in data_analysed]
    path_feats_not_analysed = [pf for pf in path_features if pf not in path_feat_analyzed]  
    path_feats_not_analysed = tuple(path_feats_not_analysed)
  
    data_not_analysed.append((path_feats_not_analysed, id_gt, camid))
    print('Unsupervised initialization finished')
    return gallery, data_not_analysed, evaluation, t
    
def IncrementalGalleryConstruction(gallery, data, config, evaluation, n):   
    """
    Load of the new query -> send to the model construction
    Parameters
    ----------
        AppModels (list): list of AppearanceModel() Class
        data (list): shuffle gallery info with the structure: ((features_path0), ID0, camID0),(features_path0), ID0, camID1),...)
        idsTOsave (list): list of identities(int) to save the evolution of their appearance models
        algorithm (str): method for data selection  
        memory_budget (int): maximum number of samples per identities 
        num_gt (int): number of labeled images used in the initialization
        iteration (str): name of the iteration being evaluated
    Returns
    -------
        AppModels (list): final appearance models 
    """
    N_SAVE = 400    
    t = n      
    print('Number of tracklets to analize:', len(data))
    for i, d in enumerate(data):                         
        path_features, id_gt, camid = d
        for j, path_feat in enumerate(path_features):
            t += 1
            #Load feature
            feature = np.load(path_feat)[0]
            
            #New query initialization as Sample() Class
            query = Sample(feature, path_feat, camid, id_gt)  
            
            #CHEKC JOINTS BEFORE CLASSIFICATION
            person, evaluation = person_exist(query, evaluation)
            if person:  
                query, final_model, evaluation = get_PseudoLabel_NewModels(gallery.models, query, evaluation, config)

                #If query comply with confidence threshold
                if final_model is not None:
                    gallery.models, evaluation = DataSelection(gallery.models, query, final_model, config, evaluation)
                else:
                    gallery, evaluation = Unknown_Analysis(gallery, query, evaluation, config)
                
                if final_model is not None and final_model.label in config.idsTOsave:
                    evaluation.saveEvolModels(gallery.models, t, config)
        ##Save metric data##
        if i%N_SAVE == 0 or i == (len(data)-1):
            print('SAVING METRICS AT TRACKLET',i)    
            evaluation.saveEvaluation(gallery.models, i, config)
            evaluation = Evaluator()
            
    #Save Final Models
    evaluation.save_models(gallery.models, config)
    evaluation.save_unknown_samples(gallery, config)
    
def IncrementalGalleryConstructionDistributed(distributed_gallery, distributed_data, id_cameras, config, evaluation):   
    """
    Load of the new query -> send to the model construction
    Parameters
    ----------
        AppModels (list): list of AppearanceModel() Class
        data (list): shuffle gallery info with the structure: ((features_path0), ID0, camID0),(features_path0), ID0, camID1),...)
        idsTOsave (list): list of identities(int) to save the evolution of their appearance models
        algorithm (str): method for data selection  
        memory_budget (int): maximum number of samples per identities 
        num_gt (int): number of labeled images used in the initialization
        iteration (str): name of the iteration being evaluated
    Returns
    -------
        AppModels (list): final appearance models 
    """
    N_SAVE = 400    
    
    n = {}
    for camid in id_cameras:
        n[camid] = 0
        
    for camid in id_cameras:
        n_cam = n[camid]
        gallery_cam = distributed_gallery[camid]
        data_cam = distributed_data[camid][n_cam]   
                
        path_features, id_gt, camid = data_cam
        for path_feat in enumerate(path_features):
            #Load feature
            feature = np.load(path_feat)[0]
            
            #New query initialization as Sample() Class
            query = Sample(feature, path_feat, camid, id_gt)            
            query, final_model, evaluation = get_PseudoLabel_NewModels(gallery_cam.models, query, evaluation)

            #If query comply with confidence threshold
            if final_model is not None:
                gallery_cam.models, evaluation = DataSelection(gallery_cam.models, query, final_model, config, evaluation)
            else:
                gallery_cam, evaluation = Unknown_Analysis(gallery_cam, query, evaluation, config)
        ##Save metric data##
        if n%N_SAVE == 0 or n == (len(data_cam)-1):
            print('SAVING METRICS AT TRACKLET',n)    
            evaluation.saveEvaluation(gallery_cam.models, n, config)
            evaluation = Evaluator()
            
        n_cam += 1  
        n[camid] = n_cam
    #Save Final Models
    evaluation.save_models_distributed(gallery.models, config)
    evaluation.save_unknown_samples_distributed(gallery, config)
    
    