import random
import numpy as np
import json 

class Configuration():
    def __init__(self, config):
        self.data_selection_algorithm = config['data_selection']
        self.memory_budget = config['memory_budget']
        self.num_labeled_data = config['num_labeled_data']
        self.min_size = config['min_size_cluster']
        self.ids_initilized = config['ids_initialized']
        self.num_identities = config['total_identities']
        self.min_models = config['min_models']
        self.confidence_threshold = config['confidence_threshold_clasif']
        self.alpha = config['alpha']
        self.merge_candidates_method = config['merge_candidates_method']
        self.merge_distance = config['merge_distance']
        self.iteration = config['iteration']
        self.name_test = config['name_test']
        self.use_of_distmat = False
        self.file_knownIDs = config['name_file_knownIDs']
        self.idsTOsave = config['idsTOsave']
        
        if 'num_known_ID' in config.keys() and 'num_identities' in config.keys():
            self.nID_known = config['num_known_ID']
            self.num_identities = config['num_identities']
        
        if config['data_selection'] == 'CPR':
            self.use_of_distmat = True
            
        if config['merge_threshold'] is None:
            self.merge_threshold = self.get_merge_threshold()
        else:
            self.merge_threshold = config['merge_threshold']
        
    def get_merge_threshold(self):
        if self.merge_distance == 'cosine':
            threshold = 0.1
        elif self.merge_distance == 'JenshenShannon':
            threshold = 0.01
        elif self.merge_distance == 'confidence':
            threshold = 2
        return threshold

def new_experiment(data, known_identities, config, name_experiment): 
    final_labels = get_selected_labels(data, known_identities, config)

    selected_data = [f for f in data if f[1] in final_labels]
    np.random.shuffle(selected_data)
    selected_data = np.array(selected_data, dtype = object)
    
    save_iteration(name_experiment, selected_data)

def new_experiment_distributed(data, known_identities, config, name_experiment): 
    final_labels = get_selected_labels(data, known_identities, config)
    final_data = {}
    for camid in data.keys():
        data_camid = data[camid]
        selected_data = [f for f in data_camid if f[1] in final_labels]
        selected_data = np.array(selected_data, dtype = object)
        
        np.random.shuffle(selected_data)
        final_data[camid] = selected_data
    save_iteration(name_experiment, final_data)
    
def get_selected_labels(data, known_identities, config):
    total_identities = config.num_identities
    
    #Identities that exists in the dataset
    listIDS = [f[1] for f in data] 
    listIDS = list(set(listIDS))
    
    #Unknown identities
    unknown_IDS = [i for i in listIDS if i not in known_identities if i != 0]
    amount_unknown_ids = len(unknown_IDS)
    if total_identities == 'all':
        unknown_labels = unknown_IDS
    else:
        num_unknown_identities = total_identities - len(known_identities)
        random_list = random.sample(range(amount_unknown_ids), num_unknown_identities)
        unknown_labels = [l for i,l in enumerate(unknown_IDS) if i in random_list]
        
    #Total identities 
    final_labels = known_identities + unknown_labels + [0]
    print('total number of identities selected {}: {} labeled, {} unlabeled'.format(len(final_labels), len(known_identities), len(unknown_labels) + 1))
    return final_labels

def save_iteration(name, data):
    np.save('./Iterations/' + name + '.npy', data)
    print('Iteration save')

def save_distributed_iteration(name, data):
    with open('./Iterations/distributed_iterations/' + name + '.json', 'w') as f:
        json.dump(data, f)
    
# =============================================================================
# def new_experiment(data, known_identities, num_unknown_identities, name_experiment):
#     #Identities that exists in the dataset
#     listIDS = [f[1] for f in data] 
#     listIDS = list(set(listIDS))
#     
#     #Unknown identities
#     unknown_IDS = [i for i in listIDS if i not in known_identities]
#     amount_unknown_ids = len(unknown_IDS)
#     random_list = random.sample(range(amount_unknown_ids), num_unknown_identities)
#     unknown_labels = [l for i,l in enumerate(unknown_IDS) if i in random_list]
#     
#     #Total identities 
#     final_labels = known_identities + unknown_labels + [0]
#     selected_data = [f for f in data if f[1] in final_labels]
#     np.random.shuffle(selected_data)
#        
#     save_iteration(name_experiment, selected_data)
# =============================================================================
