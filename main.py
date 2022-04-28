from Settings import Configuration, new_experiment
from Visualization import Visualization 
from Dataset import getDataset
from GalleryConstruction import IncrementalGalleryConstruction, features_extraction, UnsupervisedInitialization
from Evaluation import Gallery_Evaluation, Query_ReIdentification, Evaluator, ComputeFinalMetrics#, Query_ReId_FullGallery, ComputeFinalMetricsFullGallery
from DataInformation import select_known_identities, load_identities, extract_info_iteration_newModels, get_num_samples
from ComponentsGallery import Gallery


"""
EVALUATION OF CLASSIFICATION TECHNIQUES IN THE ASSIGNMENT OF A NEW PATCH TO 
ITS CLUSTER
ACTUAL PATCH SELECTION TECHNIQUE IGP
"""


def Gallery_Construction(extractFeatures, construction, visual_info, config):
    if extractFeatures:
        features_extraction()
    else:
        dir_features = '/home/scasao/Documents/0_DATASET/REID/People/Mars_normalize/'
        # dir_features = '/home/scasao/Documents/0_DATASET/REID/People/DukeMTMC-VideoReID-Feat/'
        
    print('Loading data...')
    query_info, data_info, total_imgs = getDataset('Mars', 'individualMetrics', dir_features, mode = 'tracks')
    # query_info, data_info, total_imgs = getDataset('DukeMTMC-Video', 'individualMetrics', dir_features, mode = 'tracks')
    IDS_SELECTED = select_known_identities(config, data = data_info, num_selected_ids = None, name_file = config.file_knownIDs)

    if construction:
        print('Data selection', config.data_selection_algorithm)
        #Classes initializations
        gallery = Gallery()
        evaluation = Evaluator()
        
        print('Iteration {} with identities {} and memory budget {}'.format(config.iteration,config.file_knownIDs, config.memory_budget))
        data = extract_info_iteration_newModels(config.iteration, dir_features)

        if config.ids_initilized == 'unsupervised':
            gallery, data, evaluation, n = UnsupervisedInitialization(gallery, data, config, evaluation)
        print('{} models have been initialized'.format(len(gallery.models)))
        
        assert len(gallery.models) >= config.min_models 
        IncrementalGalleryConstruction(gallery, data, config, evaluation, n)    
        print('Done')

    return query_info, total_imgs
               
if __name__ == '__main__':
    extractFeatures, construction = False, True
    
    visual_info = Visualization()

    memory_budget = [10]#,20] 
    min_size_cluster = [5]#,10]
    data_selection = ['CPR']#,'IOM', 'Temporal', 'Random', 'ExStream']
    
    # IDS_TO_SAVE_mars = [2,4,52,68,202,196,34,106,146,94,156,176,110,112,142,308,390,258,18,24,26]
    # IDS_TO_SAVE_duke = [23,80,126,143,159,21,30,86,83,194,229,254,2,3,5,180,31,77,147,197,220]
    
    for i, mb in enumerate(memory_budget):
        for ds in data_selection:
            setup = {'data_selection': ds,
                      'memory_budget': mb,
                      'num_labeled_data': None,
                      'min_size_cluster': min_size_cluster[i],
                      'confidence_threshold_clasif': 2,
                      'alpha': 0.6,
                      'merge_candidates_method': 'Moda_nearest',
                      'merge_distance': 'cosine', 
                      'merge_threshold': 0.1,
                      'min_models': 20,
                      'ids_initialized': 'unsupervised',
                      'total_identities':'all',
                      'name_file_knownIDs': [],
                      'idsTOsave': [],
                      'iteration': './Mars/unsupervised/unsupervised_iter0.npy',
                      'name_test': './Albation2/final_method_' + ds + '_final_eval/memory_budget_' + str(mb)
                      
                    }
            config = Configuration(setup)
            # query_info, total_imgs = Gallery_Construction(extractFeatures, construction, visual_info, config)
    
    # EVALUATION   
    # query_info, total_imgs = Gallery_Construction(False, False, visual_info, config)
    
    # Gallery_Evaluation(['Ablation2/final_method_IOM_final_eval/memory_budget50',
    #                     'Ablation2/final_method_Random_final_eval/memory_budget50',
    #                     'Ablation2/final_method_CPR_final_eval/memory_budget50',
    #                     'Ablation2/final_method_ExStream_final_eval/memory_budget50',
    #                     'Ablation2/final_method_Temporal_final_eval/memory_budget50',])
    
    Gallery_Evaluation(['Duke/final_method_CPR_final_eval/memory_budget_50'])
    
    # Query_ReId_FullGallery('DukeMTMC-Video')
    # ComputeFinalMetricsFullGallery('full_gallery_DukeMTMC-Video')    
    # Query_ReIdentification(['Duke/final_method_CPR_final_eval/memory_budget_50'], query_info)
