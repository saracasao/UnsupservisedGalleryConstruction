from Settings import Configuration
from Dataset import DataLoader
from GalleryConstruction import IncrementalGalleryConstruction, UnsupervisedInitialization
from Evaluation import Evaluator
from metrics.gallery_metrics import Gallery_Evaluation, Query_ReIdentification
from DataInformation import extract_info_iteration
from ComponentsGallery import Gallery


"""
EVALUATION OF CLASSIFICATION TECHNIQUES IN THE ASSIGNMENT OF A NEW PATCH TO ITS CLUSTER
"""


def Gallery_Construction(config):
    print('Data selection', config.data_selection_algorithm)

    # Initialization
    gallery = Gallery()
    evaluation = Evaluator(config)

    print('Iteration {} and memory budget {}'.format(config.iteration, config.memory_budget))
    data = extract_info_iteration(config)

    # Gallery initialization with minimum number of models (min_models)
    gallery, data, evaluation, n, t = UnsupervisedInitialization(gallery, data, config, evaluation)
    print('{} models have been initialized'.format(len(gallery.models)))

    assert len(gallery.models) >= config.min_models
    IncrementalGalleryConstruction(gallery, data, config, evaluation, n, t)
    print('Done')


if __name__ == '__main__':
    mode = 'query_evaluation'  # gallery_construction, evaluation_gallery, query_evaluation

    setup = {'dir_dataset': './data/DukeDataset/',  # introduce the path of the folder that contains the data
             'iteration': './Duke/iter0.npy',  # iteration analysed
             'data_selection': 'CPR',  # ['IOM', 'Temporal', 'Random', 'ExStream']
             'min_models': 20,  # minimum number of models to finish the initilization gallery stage
             'memory_budget': 50,  # memory budget per class
             'min_size_cluster': 20,  # minimum size of the cluster to create a new class
             'confidence_threshold_clasif': 2,  # confidence threshold in classification
             'alpha': 0.6,  # weight assigned to the uncertainty of the sample in the optimization process
             'merge_threshold': 0.1,  # cosine distant threshold for merging new classes created in the unknown data manager with existing ones in the gallery
             'save_interval': 400,  # every save_interval tacklets analyzed -> save evaluation
             'idsTOsave': [],  # selected identities to save their evolution over time
             'name_test': 'DukeTest' # path to save the results
             }

    config = Configuration(setup)
    if mode == 'gallery_construction':
        Gallery_Construction(config)
    elif mode == 'evaluation_gallery':
        Gallery_Evaluation(config)
    elif mode == 'query_evaluation':
        Query_ReIdentification(config)
    else:
        raise AssertionError('Mode does not exist')

