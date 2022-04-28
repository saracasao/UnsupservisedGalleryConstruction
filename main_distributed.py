
from Settings import Configuration, new_experiment_distributed
from Visualization import Visualization 
from Dataset import getDataset
from GalleryConstruction import ClassesInitialization, IncrementalGalleryConstructionDistributed, features_extraction
from Evaluation import Gallery_Evaluation, Query_ReIdentification, Evaluator
from DataInformation import select_known_identities, extract_info_iteration_distributed, get_distributed_data

"""
EVALUATION OF CLASSIFICATION TECHNIQUES IN THE ASSIGNMENT OF A NEW PATCH TO 
ITS CLUSTER
ACTUAL PATCH SELECTION TECHNIQUE IGP
"""

def Gallery_Construction(extractFeatures, construction, visual_info, dir_dataset, config):
    if extractFeatures:
        features_extraction()
    else:
        dir_features = '/home/scasao/Documents/0_DATASET/REID/People/Mars_normalize'
        
    print('Loading data...')
    query_info, data_info = getDataset('Mars', 'individualMetrics', dir_dataset, dir_features, mode = 'tracks')
    distributed_gallery, distributed_data, id_cameras = get_distributed_data(data_info)

    IDS_SELECTED = select_known_identities(config, data = data_info, num_selected_ids = None, name_file = 'random_10.txt')

    if construction:
        #Classes initializations
        evaluation = {}
        for id_c in id_cameras:
            evaluation[id_c] = Evaluator()

            print('Initilization appearance models...')
            distributed_gallery[id_c], distributed_data[id_c], evaluation[id_c] = ClassesInitialization(distributed_gallery[id_c], distributed_data[id_c], IDS_SELECTED, config, evaluation[id_c])
            print('Done, number of models initialized', len(distributed_gallery[id_c].models))
            
        print('Iteration', config.iteration, 'with', config.data_selection_algorithm)
        distributed_data = extract_info_iteration_distributed(config.iteration, dir_features)
        
        IncrementalGalleryConstructionDistributed(distributed_gallery, distributed_data, id_cameras, config, evaluation)    
        print('Done')

    return query_info
               
if __name__ == '__main__':
    dir_dataset = '/home/scasao/Documents/0_DATASET/REID/People/Mars'
    extractFeatures, construction = False, True
    
    visual_info = Visualization()
    # TEST
    data_selection = ['CPR']#'ExStream', 'IOM', 'Random', 'Temporal']
    for ds in data_selection:
        setup = {'data_selection': ds,
                  'memory_budget': 50, #50
                  'num_labeled_data': 25, #25
                  'min_size_cluster': 25,
                  'ids_initialized': 'ids_preselected', #'half', 'all','random', 'ids_preselected'
                  'confidence_threshold_clasif': 2,
                  'alpha': 0.2,
                  'merge_candidates_method': 'Moda_nearest',#all, 'Moda_nearest','NearestCentroids'
                  'merge_distance': 'cosine', # 'confidence', 'JenshenShannon'
                  'merge_threshold': 0.1,
                  'iteration': '10known_190unknown.npy',
                  'num_known_ID': 10,
                  'num_unknown_ID': 190,
                  'name_test': 'Distributed/DA_' + str(ds) + 'wCentroid',
                }
        config = Configuration(setup)
        
        query_info = Gallery_Construction(extractFeatures, construction, visual_info, dir_dataset, config)
    #EVALUATION
    # print('\n GALLERY EVALUATION')
    # Gallery_Evaluation(setup['data_selection'], setup['num_labeled_data'])
    Gallery_Evaluation('CPR', 25)
    # Query_ReIdentification(ALGORITHMS, query_info)
