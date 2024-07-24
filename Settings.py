import os
import random
import numpy as np


class Configuration:
    def __init__(self, config):
        self.dir_dataset = config['dir_dataset']
        assert os.path.exists(self.dir_dataset), "Dataset directory doest not exist"

        self.data_selection_algorithm = config['data_selection']
        self.memory_budget = config['memory_budget']
        self.min_size = config['min_size_cluster']
        self.min_models = config['min_models']
        self.confidence_threshold = config['confidence_threshold_clasif']
        self.alpha = config['alpha']
        self.merge_candidates_method = 'Moda_nearest'
        self.merge_distance = 'cosine'
        self.merge_threshold = config['merge_threshold']
        self.iteration = config['iteration']
        self.name_test = config['name_test']
        self.use_of_distmat = False
        self.idsTOsave = config['idsTOsave']
        self.save_interval = config['save_interval']
        self.setup = config

        # TODO: CHECK HERE THE PATHS REFERRING TO YOUR FOLDERS
        folders = os.listdir(self.dir_dataset)
        folder_features = [f for f in folders if 'Feat' in f][0]
        folders.remove(folder_features)
        folder_images = folders[0]

        self.dir_features = self.dir_dataset + folder_features
        assert os.path.exists(self.dir_features), "Features directory doest not exist"

        self.dir_skeletons = self.dir_dataset + folder_features + '/gallery_sklt/'
        assert os.path.exists(self.dir_skeletons), "Skeletons directory doest not exist"

        self.dir_images = self.dir_dataset + folder_images + '/gallery'
        assert os.path.exists(self.dir_images), "Images directory doest not exist"
        
        if config['data_selection'] == 'CPR':
            self.use_of_distmat = True


def new_experiment(data, name_experiment):
    final_labels = get_labels(data)

    selected_data = [f for f in data if f[1] in final_labels]
    np.random.shuffle(selected_data)
    selected_data = np.array(selected_data, dtype = object)
    
    save_iteration(name_experiment, selected_data)

    
def get_labels(data):
    # Identities that exists in the dataset
    listIDS = [f[1] for f in data] 
    listIDS = list(set(listIDS))

    print('total number of identities {}'.format(len(listIDS)))
    return listIDS


def save_iteration(name, data):
    np.save('./Iterations/' + name + '.npy', data)
    print('Iteration save')
