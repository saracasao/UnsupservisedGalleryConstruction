import numpy as np

from QueryClassification import DataSelection, get_PseudoLabel, get_PseudoLabel_Dist, DataSelection_InitStage
from UnknownManagement import Unknown_Analysis, InitCandidateModel, candidatesTOmodels, check_person, person_exist
from Evaluation import Evaluator
from ComponentsGallery import Sample


def UnsupervisedInitialization(gallery, data, config, evaluation):
    """
    Initialize the empty gallery with a minimum number of classes (config.min_models) based on distance classification
    Parameters
    ----------
        gallery (Gallery Class): empty gallery
        data (list): shuffle gallery info with the structure: ((features_path0), ID0, camID0),(features_path0), ID0, camID1),...)
        evaluation (Evaluation Class)
    Returns
    -------
        gallery (Gallery Class): gallery initialized
        data_not_analysed (list): components of the data (list) that remain not analysed
        t (int): number of samples analysed
    """

    print('Number of tracklets to analize:', len(data)) 
    n, t = 0, 0
    initialization = False
    
    data_analysed = []
    while not initialization: 
        d = data[n]
        path_feat_analyzed = []
        
        data_analysed.append(d)
        path_features, id_gt, camid = d
        for j, path_feat in enumerate(path_features):
            t += 1
            # Load feature
            feature = np.load(path_feat)[0]
            path_feat_analyzed.append(path_feat)
            
            # New query initialization as Sample() Class
            query = Sample(feature, path_feat, camid, id_gt, config)
            person, evaluation = check_person(query, evaluation)
            if person:
                query, final_model = get_PseudoLabel_Dist(gallery.candidatesTOmodel, query)
                # If query comply with confidence threshold
                if final_model is not None:
                    gallery.candidatesTOmodel, evaluation = DataSelection_InitStage(gallery, gallery.candidatesTOmodel, query, final_model, config, evaluation)
                else:
                    gallery = InitCandidateModel(gallery, query)
                
                # Check if there is a minimum of models with a minimum of samples
                gallery, initialization = candidatesTOmodels(gallery, config)   
                if initialization:
                    break       
            
        # Save metric data
        if n % config.save_interval == 0 or initialization:
            print('SAVING METRICS AT TRACKLET',n)    
            evaluation.saveEvaluation(n)
            gallery.models = evaluation.saveEvolModels(gallery.models, t, config)
            evaluation = Evaluator(config)
        
        n += 1
 
    data_not_analysed = [d for d in data if d not in data_analysed]
    path_feats_not_analysed = [pf for pf in path_features if pf not in path_feat_analyzed]  
    path_feats_not_analysed = tuple(path_feats_not_analysed)
  
    data_not_analysed.append((path_feats_not_analysed, id_gt, camid))
    print('Unsupervised initialization finished')
    return gallery, data_not_analysed, evaluation, n, t


def IncrementalGalleryConstruction(gallery, data, config, evaluation, n, t):
    """
    Load of the new query -> send to the model construction
    Parameters
    ----------
        gallery (Gallery Class): initialized gallery with minimum number of classes
        data (list): shuffle gallery tacklets with the structure: ((features_path0), ID0, camID0),(features_path0), ID0, camID1),...)
        n (int): number of samples analyzed in the initialization stage
    """

    print('Number of tracklets to analize:', len(data))
    for i, d in enumerate(data):                         
        path_features, id_gt, camid = d
        for j, path_feat in enumerate(path_features):
            t += 1
            # Load feature
            feature = np.load(path_feat)[0]
            
            # New query initialization as Sample() Class
            query = Sample(feature, path_feat, camid, id_gt, config)
            
            # Check it a person is shown
            person, evaluation = person_exist(query, evaluation)
            if person:  
                query, final_model, evaluation = get_PseudoLabel(gallery.models, query, evaluation, config)

                # If query comply with confidence threshold
                if final_model is not None:
                    gallery.models, evaluation = DataSelection(gallery.models, query, final_model, config, evaluation)
                else:
                    gallery, evaluation = Unknown_Analysis(gallery, query, evaluation, config)
                
                if final_model is not None and final_model.label in config.idsTOsave:
                    evaluation.saveEvolModels(gallery.models, t, config)
        # Save metric data
        if i % config.save_interval == 0 or i == (len(data)-1):
            print('SAVING METRICS AT TRACKLET',i + n)
            evaluation.saveEvaluation(i + n)
            evaluation = Evaluator(config)
            
    # Save Final Models
    evaluation.save_models(gallery.models)
    evaluation.save_unknown_samples(gallery)

    