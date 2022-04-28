import json 
import os 
import numpy as np
import cv2

import matplotlib.pyplot as plt
from metrics.distance import compute_distance_matrix
from metrics.rank import evaluate_rank
from EvaluationRank import evaluate_py
from DataInformation import parse_data_for_eval, parse_data_for_save, extract_info_query, extract_info_gallery, getDataList, parse_unknown_for_save
from Utils import TSNE_clusters, plot_clusters_frequency, plot_TSNE_unknown
from scipy import spatial
from DataInformation import load_identities, parse_unknown_for_eval, get_recall, get_precision, get_noise_eval
from eval_mode import eval_global

dir_saveMetric = '/home/scasao/pytorch/1_Clusters/0.1_Semi-SupervisedIncrementalGallery/Metrics/'
dir_saveModels = '/home/scasao/pytorch/1_Clusters/0.1_Semi-SupervisedIncrementalGallery/Final_modelsSub/'
dir_saveEval = '/home/scasao/pytorch/1_Clusters/0.1_Semi-SupervisedIncrementalGallery/DataEval/'
dir_saveEvol = '/home/scasao/pytorch/1_Clusters/0.1_Semi-SupervisedIncrementalGallery/ModelsEvolution/'

class Evaluator():
    def __init__(self):
        #PREDICTION
        #Phase 1 clasif
        self.pred_clasif_kept = []
        self.gt_clasif_kept = []
        self.id_clasif_kept = []
        self.discards = []
        #Phase 2 
        self.pred_merge = []
        self.gt_merge = []
        self.id_merge = []
        self.n_merge = 0
        #Phase 3
        self.pred_init = []
        self.gt_init = []
        self.id_init = []
        
        self.acc_Gallery = []
        self.id_gallery = []
        self.distractorsDiscarded_Clasf = []
        self.distractorsDiscarded_DS = []
        self.noiseName = []
        self.personName = []
        self.cluster_labels = []   
    
    def eval_query_selected(self, model, query):
        model.n_newModel += 1 
        keep = True
        if query.gt == model.label:
            self.acc_Gallery.append(1)
            self.id_gallery.append(int(model.identity))
        elif query.gt != model.label:
            self.acc_Gallery.append(0)
            self.id_gallery.append(int(model.identity))
        return model, keep
    
    def eval_model_merged(self, model, new_model_samples, final_samples):
        new_samples = [s for s in final_samples if s in new_model_samples]
        for s in new_samples:
            if s.gt == model.label:
                self.acc_Gallery.append(1)
                self.id_gallery.append(int(model.identity))
            elif s.gt != model.label:
                self.acc_Gallery.append(0)
                self.id_gallery.append(int(model.identity))
                
    def eval_new_model(self, model, init_samples, final_samples):
        for s in final_samples:
            #Gallery accuracy 
            if s.gt == model.label:
                self.acc_Gallery.append(1)
                self.id_gallery.append(int(model.identity))
            elif s.gt != model.label:
                self.acc_Gallery.append(0)
                self.id_gallery.append(int(model.identity))
            #Accuracy discarding noise
            if s.gt == 0:
                self.distractorsDiscarded_DS.append(0) 
                
        delete_samples = [s for s in init_samples if s not in final_samples]        
        #Accuracy discarding noise
        for d in delete_samples:
            if d.gt == 0:
                self.distractorsDiscarded_DS.append(1)
                
    def eval_noise_discarded_merged_process(self, new_samples, final_samples):
        #Correctly discarded
        discarded_samples = [s for s in new_samples if s not in final_samples]
        for s in discarded_samples:
            if s.gt == 0:
                self.distractorsDiscarded_DS.append(1)
       #Incorrectly save  
        kept_samples = [s for s in new_samples if s in final_samples]
        for s in kept_samples:
            if s.gt == 0:
                self.distractorsDiscarded_DS.append(0)
                
    def eval_noise_discarded(self, query, keep):
        if keep and query.gt == 0: 
            self.distractorsDiscarded_DS.append(0)
        elif not keep and query.gt == 0:
            self.distractorsDiscarded_DS.append(1)
            
    def eval_pseudolabeling(self, query, predicted_model):         
        #Evaluation of the distractors discarded
        if predicted_model is None and query.gt == 0:
            self.distractorsDiscarded_Clasf.append(1)
        elif predicted_model is not None and query.gt == 0:
            self.distractorsDiscarded_Clasf.append(0)
            
    def saveEvaluation(self, AppModels, TrackletIndex, config):
        info_precision = {'GalleryPrecision': self.acc_Gallery, 
                          'id_gallery': self.id_gallery,
                          'pred_clasif': self.pred_clasif_kept,
                          'gt_clasif': self.gt_clasif_kept,
                          'id_clasif': self.id_clasif_kept,
                          'pred_merge':self.pred_merge,
                          'gt_merge': self.gt_merge,
                          'id_merge': self.id_merge,
                          'num_merge': self.n_merge,
                          'pred_init': self.pred_init,
                          'gt_init': self.gt_init,
                          'id_init': self.id_init,
                          'discard_clasif': self.discards,
                          'Distractors_DiscardedClasf': self.distractorsDiscarded_Clasf,
                          'Distractors_Discarded_DS': self.distractorsDiscarded_DS,
                          'noiseName': self.noiseName,
                          'personName': self.personName,
                          'cluster_labels': self.cluster_labels
                        }

        name = str(TrackletIndex)
        name = name.zfill(6)
        
        name_test = config.name_test
        min_size = config.min_size 
        MEMORY_BUDGET = config.memory_budget
        iteration = config.iteration
        
        iteration_name  = iteration.split('/')[-1] 
        iteration_name  = iteration_name.split('.')[0]
        
        #Create folder if not exists
        if not os.path.exists(dir_saveMetric + name_test + '/' + str(min_size) + '_' + str(MEMORY_BUDGET) + '/' + iteration_name):
            os.makedirs(dir_saveMetric + name_test + '/' + str(min_size) + '_' + str(MEMORY_BUDGET)  + '/' + iteration_name)
        
        #Save metrics
        with open(dir_saveMetric + name_test + '/' + str(min_size) + '_' + str(MEMORY_BUDGET)  + '/' + iteration_name + '/' + name + '_metrics.json', 'w') as file:
            json.dump(info_precision,file)
            
    def saveEvolModels(self, AppModels, sample_index, config):
        idsTOsave = config.idsTOsave
        name_test = config.name_test
        
        identities = [m.identity for m in AppModels]
        models = [m for m in AppModels if m.label in idsTOsave]       

        for m in models:
            new_model = m.n_newModel
            last_update = m.last_modelUpate
            
            if last_update != new_model:
                m.last_modelUpate = new_model
                
                samples = m.samples
                samples_path = [s.get_img_path() for s in samples]
                
                model_id = str(m.identity)
                model_label = str(m.label)
                
                name_folder= model_label + '_' + model_id
                name_file = str(sample_index)
                name_file = name_file.zfill(8)
                
                if not os.path.exists(dir_saveEvol + name_test + '/' + name_folder):
                    os.makedirs(dir_saveEvol + name_test + '/' + name_folder)

                with open(dir_saveEvol + name_test + '/' + name_folder + '/' + name_file + '.json', 'w') as outfile:
                    json.dump(samples_path, outfile)        
                
                idx = identities.index(m.identity)
                assert AppModels[idx].identity == m.identity                

                AppModels[idx] = m

        return AppModels
    
    def save_models(self, AppModels, config):
        name_test = config.name_test
        min_size = config.min_size 
        MEMORY_BUDGET = config.memory_budget
        iteration = config.iteration
        
        iteration_name  = iteration.split('/')[-1] 
        iteration_name  = iteration_name.split('.')[0]
        
        #Create folder if not exists
        if not os.path.exists(dir_saveModels + name_test + '/' + str(min_size) + '_' + str(MEMORY_BUDGET) + '/' + iteration_name):
            os.makedirs(dir_saveModels + name_test + '/' + str(min_size) + '_' + str(MEMORY_BUDGET) + '/' + iteration_name)
            
        for model in AppModels:
            model_id = str(model.identity)
            name_file = model_id.zfill(4)
              
            data = parse_data_for_save(model)

            with open(dir_saveModels + name_test + '/' + str(min_size) + '_' + str(MEMORY_BUDGET)  + '/' + iteration_name + '/' + name_file + '.json', 'w') as outfile:
                json.dump(data, outfile)
    
    def save_unknown_samples(self, gallery, config):
        name_test = config.name_test
        min_size = config.min_size 
        MEMORY_BUDGET = config.memory_budget
        iteration = config.iteration
        
        iteration_name  = iteration.split('/')[-1] 
        iteration_name  = iteration_name.split('.')[0]
        
        #Create folder if not exists
        if not os.path.exists(dir_saveModels + name_test + '/' + str(min_size) + '_' + str(MEMORY_BUDGET) + '/' + iteration_name):
            os.makedirs(dir_saveModels + name_test + '/' + str(min_size) + '_' + str(MEMORY_BUDGET) + '/' + iteration_name)
              
        data = parse_unknown_for_save(gallery.unknown_samples)

        with open(dir_saveModels + name_test + '/' + str(min_size) + '_' + str(MEMORY_BUDGET)  + '/' + iteration_name + '/final_unknown_samples.json', 'w') as outfile:
            json.dump(data, outfile)

def Gallery_Evaluation(ALGORITHMS): 
    cluster_structure = {}
    # ALGORITHMS = ALGORITHMS[0:2]
    for a in ALGORITHMS:
        print('----------------------------')
        print('ALGORITHM ', a)
        cluster_structure[a] = []
        size_configuration = np.sort(os.listdir(dir_saveMetric + a + '/' ))
        for size in size_configuration:
            print('\n · Size configuration (numLabel_TotalMemory):',size)
            iterations = np.sort(os.listdir(dir_saveMetric + a + '/' + size + '/' ))
            # iterations = [iterations[0]]
            ids = {}
            for it in iterations:
                ids[it] = []
                print('** ITERATION **', it)
                x,y = eval_global(a, size, it, ids)
                # eval_itemize(a, size, it, ids)
                print('')            
                cluster_structure[a].append(y)
   
    # colors = {'IOM': 'tab:orange', 'Random': 'tab:red', 'Ours': 'tab:blue', 'Uniform': 'tab:green', 'ExStream': 'tab:purple'}        
  
    # X = [0,1,2,3,4]
    # bar = np.arange(len(X))
    # w = 0.15
    
    # plt.figure()  
    # for i,a in enumerate(ALGORITHMS):
    #     if i == 2:
    #         bar_ref = bar
    #     Y = np.mean(cluster_structure[a], axis = 0)
    #     std = np.std(cluster_structure[a], axis = 0)
        
    #     name_complete = a.split('/')[1]
    #     label = name_complete.split('_')[2]
    #     assert len(X) == len(Y) == len(std)
    #     if label == 'CPR':
    #         label = 'Ours'
    #     elif label == 'Temporal':
    #         label = 'Uniform'
    #     plt.bar(bar, Y, w, yerr=std, color = colors[label], label = label)
    #     bar = [i+w for i in bar]

        
    # plt.ylim(0,300)
    # values = ['0','1','2','3','>4']              
    # plt.xticks(bar_ref,values)
    # plt.rc('xtick',labelsize = 12)
    # plt.rc('ytick',labelsize = 12)
    # plt.xlabel('Num Clusters Initialized',size = 12)
    # plt.ylabel('Num Identities', size = 12)
    # plt.legend(bbox_to_anchor=(0,1.02,1,0.2), loc="lower left", mode="expand", borderaxespad=0, ncol=3)
    # plt.savefig('./GRAPHS/ClusterDistribution2_leyend.png', dpi = 1000, bbox_inches="tight")  
                
def Query_ReIdentification(ALGORITHMS, query_paths):
    """
    Traditional re-identification process 
    """
    print('QUERY EVALUATION:')
    #Number of batches
    n = 70
    for a in ALGORITHMS:
        print('----------------------------')
        print('ALGORITHM ', a)
        size_configuration = os.listdir(dir_saveModels + a)
        size_configuration = [size_configuration[0]]
        for size in size_configuration:
            iterations = np.sort(os.listdir(dir_saveModels + a + '/' + size +'/' ))
            for it in iterations:
                print('\n · Size configuration (numLabel_TotalMemory):', size, 'iteration',it)
                
                print('Extracting information from gallery...')
                gf, g_pids, g_camids = extract_info_gallery(dir_saveModels, a, size, it)
                print('Extracting information from query set...')
                qf, q_pids, q_camids = extract_info_query(query_paths)

                #Get batch_size
                batch_size = np.int0(len(qf)/n)    
                for i in range(n):
                    init = i*batch_size
                    if i < n:
                        end = (i+1)*batch_size
                    else:
                        end = len(qf)
                
                    qf_batch = qf[init:end]
                    q_pids_batch = q_pids[init:end]
                    q_camid_batch = q_camids[init:end]
                    print('Init:', init,'End:', end)
                
                    #Obtain distance matrix
                    distmat = compute_distance_matrix(qf_batch, gf, 'cosine')
                    distmat = distmat.numpy()
                    
                    # Delete patches with same ID same camera
                    cmc, AP, num_valid_q = evaluate_rank(
                        distmat,
                        q_pids_batch,
                        g_pids,
                        q_camid_batch,
                        g_camids,
                        video = True
                    )
                    
                    results = {'cmc': cmc,
                                'AP': AP,
                                'numValid': num_valid_q}
                
                    index = str(i)
                    index = index.zfill(4)
                    if not os.path.exists(dir_saveEval + a + '/' + size + '/' + it):
                        os.makedirs(dir_saveEval + a + '/' + size + '/' + it)
                        
                    name_file = dir_saveEval + a + '/' + size + '/' + it + '/' + index + '.json'
                    with open(name_file, 'w') as outfile:
                        json.dump(results,outfile)           
    ComputeFinalMetrics(ALGORITHMS)
    
def ComputeFinalMetrics(ALGORITHMS):
    """
    Compute the final metrics of the re-identification process
    """
    for a in ALGORITHMS:
        print('----------------------------')
        print('ALGORITHM ', a)
        size_configuration = os.listdir(dir_saveEval + a)
        for size in size_configuration:
            iterations = np.sort(os.listdir(dir_saveEval + a + '/' + size +'/' ))
            for it in iterations:
                print('\n · Size configuration (numLabel_TotalMemory):', size, 'iteration',it)
                all_cmc, all_AP = [],[]
                num_valid = 0.
                
                name_files = sorted([f for f in os.listdir(dir_saveEval + a + '/' + size + '/' + it)])
                for name_file in name_files:
                    with open(dir_saveEval + a + '/' + size + '/' + it +'/' + name_file) as f:
                        data = json.load(f)
                    #Unify all data in diferent files in one unique list 
                    all_cmc = getDataList(data['cmc'],all_cmc)
                    all_AP = getDataList(data['AP'],all_AP)
                    num_valid = num_valid + data['numValid']
                all_cmc = np.array(all_cmc).astype(np.float32)
                cmc = all_cmc.sum(0) / num_valid
                mAP = np.mean(all_AP)
                
                ranks=[1, 5]
                print('** Results **')
                print('mAP: {:.1%}'.format(mAP))
                print('CMC curve')
                for r in ranks:
                    print('Rank-{:<3}: {:.1%}'.format(r, cmc[r - 1]))            