import json 
import os 
import numpy as np
import matplotlib.pyplot as plt

from Dataset import getDataset
from sklearn import metrics
# from metrics.distance import compute_distance_matrix
# from metrics.rank import evaluate_rank
from EvaluationRank import evaluate_py
from DataInformation import extract_info_clusters, parse_data_for_eval, parse_data_for_save, extract_info_query, extract_info_gallery, parse_unknown_for_save, load_identities, parse_unknown_for_eval
from Utils import TSNE_clusters, plot_clusters_frequency, plot_TSNE_unknown

dir_saveMetric = '/home/scasao/pytorch/1_Clusters/0.1_Semi-SupervisedIncrementalGallery/Metrics/'
dir_saveModels = '/home/scasao/pytorch/1_Clusters/0.1_Semi-SupervisedIncrementalGallery/Final_modelsSub/'

def eval_global(a, size, it, ids, total = None):
    #Info dataset to evaluate
    # dir_features = '/home/scasao/Documents/0_DATASET/REID/People/Mars_normalize/'
    dir_features = '/home/scasao/Documents/0_DATASET/REID/People/DukeMTMC-VideoReID-Feat/'
   
    query_info, data_info, total_imgs = getDataset('DukeMTMC-Video', 'individualMetrics', dir_features, mode = 'tracks')
    # query_info, data_info, total_imgs = getDataset('Mars', 'individualMetrics', dir_features, mode = 'tracks')
    
    print('total imgs', total_imgs)
    
    data_id = [d[1] for d in data_info if d[1] != 0]
    data_id = list(set(data_id))
    
    #Metrics in the classification process
    all_GalleryPrecision, all_GalleryID, all_clusters = [],[], []
    all_pred_clasif, all_pred_merge, all_pred_init = [],[],[]
    all_gt_clasif, all_gt_merge, all_gt_init = [],[],[]
    all_predID_clasif, all_predID_merge, all_predID_init = [],[],[]
    
    #Unify all data in diferent files in one unique list evaluate_rank
    name_files = os.listdir(dir_saveMetric + a + '/' + size + '/' + it)
    name_files = np.sort(name_files)
    for name_file in name_files:
        with open(dir_saveMetric + a + '/' + size + '/' + it + '/' + name_file) as f:
            data = json.load(f)
        #Accuracy of class prediction and recall
        all_pred_clasif = getDataList(data['pred_clasif'], all_pred_clasif)
        all_gt_clasif = getDataList(data['gt_clasif'], all_gt_clasif)
        all_predID_clasif = getDataList(data['id_clasif'], all_predID_clasif)
        #Acc merge
        all_pred_merge = getDataList(data['pred_merge'], all_pred_merge)
        all_gt_merge = getDataList(data['gt_merge'], all_gt_merge)
        all_predID_merge = getDataList(data['id_merge'], all_predID_merge)
        #Acc init
        all_pred_init = getDataList(data['pred_init'], all_pred_init)
        all_gt_init = getDataList(data['gt_init'], all_gt_init)
        all_predID_init = getDataList(data['id_init'], all_predID_init)

        #Gallery Accuracy
        all_GalleryPrecision = getDataList(data['GalleryPrecision'], all_GalleryPrecision)
        all_GalleryID = getDataList(data['id_gallery'], all_GalleryID)
        
        #Clusters created
        all_clusters = getDataList(data['cluster_labels'], all_clusters)

    #PSEUDO-LABEL ASSIGNED
    # Precision 
    predictions = all_pred_clasif + all_pred_merge + all_pred_init
    pred_ident  = all_predID_clasif + all_predID_merge + all_predID_init
    groundTruth = all_gt_clasif + all_gt_merge + all_gt_init
    
    true_positives, false_positives, avg_precision, avg_recall, valid_ID, X, Y = get_Nprecision(all_clusters, data_id, predictions, pred_ident, groundTruth, a, size, it, data_info, N = 4)    
    valid_gallery_ID = [g_id in valid_ID for g_id in all_GalleryID]
    
    # valid_gallery_ID = np.array(valid_gallery_ID)
    # all_GalleryPrecision = np.array(all_GalleryPrecision)
    
    # gallery_precision = all_GalleryPrecision[valid_gallery_ID]  
    
    # Precision & Recall
    precision = np.sum(true_positives) / (np.sum(true_positives) + np.sum(false_positives))
    recall = np.sum(true_positives) / total_imgs
    F1_pred = (2*precision*recall) / (precision + recall)
    
    #GALLERY
    all_GalleryPrecision = np.array(all_GalleryPrecision)
    FinalGalleryPrecision = np.sum(all_GalleryPrecision) / len(all_GalleryPrecision)

    #CLUSTERS Analysis
    if len(ids[it]) > 0:
        labeled_clusters = load_identities(ids[it])
    else:
        labeled_clusters = []
    
    id_clusters_initialized = list(set(all_clusters))
    new_clusters_initialized = [i for i in id_clusters_initialized if i not in labeled_clusters]
    
    cluster_precision = len(new_clusters_initialized) / len(all_clusters)
    cluster_recall = len(new_clusters_initialized) / len(data_id)
    F1_clusters = (2*cluster_precision*cluster_recall) / (cluster_precision + cluster_recall)
    # plot_cluster_distribution(all_clusters, labeled_clusters, a, size, it)
    
    print('** Results **')
    print('Accuracy of {} images included in the gallery: {:.1%}'.format(len(all_GalleryPrecision),FinalGalleryPrecision))
    print('Analysis of the label assigned:')
    print('-> Precision, recall, F1 of predicted:{:.2%}, {:.2%},{:.2%}'.format(precision, recall, F1_pred)) #num_predicted,
    print('Cluster analysis:')
    print('-> Total of clusters initialized:', len(all_clusters))
    print('-> New clusters initialize correctly:', len(new_clusters_initialized))
    print('-> Precision and recall in classes identification:{:.2%}, {:.2%}, {:.2%}'.format(cluster_precision, cluster_recall, F1_clusters))
    
    return X,Y

def Final_Clusters_Analysis(a, size, it):
    print('----------------------------')
    print('ALGORITHM ', a)

    print('\n · Size configuration (numLabel_TotalMemory):', size, 'iteration',it)
    final_clusters = []
    
    name_files = os.listdir(dir_saveMetric + a + '/' + size + '/' + it)
    name_files = np.sort(name_files)
    for name_file in name_files:
        with open(dir_saveMetric + a + '/' + size + '/' + it + '/' + name_file) as f:
            data = json.load(f)
            
        final_clusters = getDataList(data['cluster_labels'], final_clusters)
    id_clusters, amount_init = np.unique(final_clusters, return_counts =  True)
    
    print('Extracting information from gallery...')
    gf, g_pids, gt, g_camids, g_dir = extract_info_gallery(dir_saveModels, a, size, it)                
    
    idxs_max_duplicate = np.argsort(amount_init)[-3:len(amount_init)]
    # idxs_max_duplicate = np.argsort(amount_init)[0:3]
    for idx_max_duplicate in idxs_max_duplicate:
        id_max_duplicate = id_clusters[idx_max_duplicate]
        # id_max_duplicate = 2
        print('label cluster', id_max_duplicate)
        idx_selected = np.where(gt == id_max_duplicate)

        gf_s = gf[idx_selected]
        gt_s = gt[idx_selected]
        g_pids_s = g_pids[idx_selected]
        g_camids_s = g_camids[idx_selected]
        g_dir_s = g_dir[idx_selected]

        TSNE_clusters(gf_s, g_pids_s, g_dir_s, a + '_' + size + '_' + it + '_clusters_' + str(id_max_duplicate) + '.png')
        
def getDataList(data, final_list):
    for i in range(len(data)):
        final_list.append(data[i])
    return final_list

def get_noise_eval(predictions, groundTruth):
    noise_predictions = [pred for pred in predictions if pred[0] == '0000']
    print('len noise prediction', len(noise_predictions))
    i = 0
    for pred in noise_predictions:
        if pred[1] in groundTruth:
            i += 1

    precision = i / len(noise_predictions)
    recall = i / len(groundTruth)
    return precision, recall

def get_Nprecision(clusters, gt_identities, predictions, pred_identity, groundTruth, a, size, it, data_info, N = 4):
    assert len(predictions) == len(pred_identity) == len(groundTruth)
    id_clusters_initialized, amount_clusters_same_id = np.unique(clusters, return_counts = True) 
    id_scenario, amount_clusters_same_id = get_info_scenario(id_clusters_initialized, gt_identities, amount_clusters_same_id)
    
    clusters_info = extract_info_clusters(dir_saveModels, a, size, it)
    clusters_label = [c[3] for c in clusters_info]
    clusters_id = [c[4] for c in clusters_info]

    assert len(clusters_label) == len(clusters_id)

    models_dict = {}
    for cl in id_scenario:
        cl_idx =  [i for i,l in enumerate(clusters_label) if l==cl]
        c_ids = sorted([c_id for i, c_id in enumerate(clusters_id) if i in cl_idx])    
        models_dict[cl] = c_ids
      
    predictions = np.array(predictions)
    groundTruth = np.array(groundTruth)
    pred_identity = np.array(pred_identity)
    
    precision, recall, gallery_valid_IDs = [], [], []
    true_positives, false_positives = [], []
    for i, cl in enumerate(id_scenario):
        if len(models_dict[cl]) > 0:
            #Get data
            predictions_cluster_idx = np.where(predictions == cl)
            predictions_cluster = predictions[predictions_cluster_idx]
            predictions_cluster_identity = pred_identity[predictions_cluster_idx]
            gt = groundTruth[predictions_cluster_idx]
            
            #Select valid clusters
            pred_id, freq_id = np.unique(predictions_cluster_identity, return_counts = True)
            idx_freq = np.argsort(freq_id)[::-1]
            modelID_valids = pred_id[idx_freq][:N]
            modelID_valids = list(modelID_valids)

            #Get matches 
            match_gt = (predictions_cluster == gt)
            match_id = [p_id in modelID_valids for p_id in predictions_cluster_identity]
            match_id = np.array(match_id)
            match = match_gt & match_id
            
            valid_predictions = match_gt[match_id]  
            failures = np.invert(valid_predictions)
            
            assert match_id.sum() >= match.sum()
            
            tp = match.sum()
            fp = failures.sum()   
            fn = get_FN(cl, data_info, tp)
            
            P = tp / (tp + fp)
            R = tp / (tp + fn)
            
            precision.append(P)
            recall.append(R)
            true_positives.append(tp)
            false_positives.append(fp)       
            
            gallery_valid_IDs = gallery_valid_IDs + modelID_valids
        else:
            precision.append(0)
            recall.append(0)

    avg_precision = sum(precision) / len(id_scenario)
    avg_recall = sum(recall) / len(id_scenario)
    
    X,Y = plot_cluster_distribution(models_dict, clusters, a,  size, it)
    return true_positives, false_positives, avg_precision, avg_recall, gallery_valid_IDs, X, Y

def plot_cluster_distribution(models_dict, clusters, a, size, it):
    total_clusters = len(clusters)
    
    id_not_identify = [i for i,k in enumerate(models_dict.keys()) if len(models_dict[k]) == 0]
    print('ids not identifies', len(id_not_identify))
    cluster_labels, freq = np.unique(clusters, return_counts = True)
    num_repeticiones, freq_rept = np.unique(freq, return_counts = True)
    
    max_rep = 4
    X,Y = [],[]
    freq_higher10 = 0
    for i, num_repet in enumerate(num_repeticiones):
        if num_repet ==  i+1 and num_repet < max_rep:
            X.append(num_repet)
            Y.append(freq_rept[i])
        elif num_repet !=  i+1 and num_repet < max_rep:
            X.append(i+1)
            Y.append(0)
        else: 
            freq_higher10 = freq_higher10 + freq_rept[i]
        
 
    X.append(max_rep)
    Y.append(freq_higher10)

    for i in range(max_rep + 1):
        if i == 0:
            X[i:i] = [i]
            Y[i:i] = [len(id_not_identify)]
        if i not in X:
            X[i:i] = [i]
            Y[i:i] = [0]                
    
    # colors = ['blue' for n in X]
    # colors[0] = 'red'
    
    # values = ['0','1','2','3','4','5','6','7','8','9','>10']  
    # assert len(X) == len(Y) == len(colors) == len(values), 'Different lengths X {}, Y {}, colors {}, values {}'.format(len(X),len(Y),len(colors),len(values))
    # plt.figure()
    # plt.bar(X,Y, color = colors)
    # plt.ylim(0,300)
    
    # alg = a.split('/')[1]
    # name_plot = alg.split('_')[2]
    
    # plt.title(name_plot)
    # plt.xticks(X,values)
    # plt.show()
    # plt.savefig('./' + name_plot + '.png')
    
    return X,Y


def plot_precision_recall(precision_ids, recall_ids, id_scenario):
    full_pred_idx = np.where(precision_ids == 1)

    x = id_scenario
    y1 = precision_ids
    y2 = recall_ids
    
    plt.figure()
    plt.title('Precision')
    plt.bar(x, y1)
    plt.show()
    
    plt.figure()
    plt.title('Recall')
    plt.bar(x, y2)
    plt.show()
    
def get_info_scenario(id_initialized, gt_identities, n_clusters_same_id):
    id_scenario, num_clusters_same_id = [], []
    for i in gt_identities:
        if i in id_initialized:
            i_idx = np.where(id_initialized == i)
            id_scenario.append(id_initialized[i_idx][0])
            num_clusters_same_id.append(n_clusters_same_id[i_idx][0])
        else:
            id_scenario.append(i)
            num_clusters_same_id.append(0)
    assert len(id_scenario) == len(num_clusters_same_id) == len(gt_identities)

    return id_scenario, num_clusters_same_id
    
def get_TP_FP(identity, predictions, groundTruth):
    predictions = np.array(predictions)
    groundTruth = np.array(groundTruth)
    
    #Get idx of the identity predictions
    pred_idx = np.where(predictions == identity)
    
    #If the system identifies the identity -> compute TP and FN
    if len(pred_idx) > 0:
        pred_ids = predictions[pred_idx]
        pred_gt  = groundTruth[pred_idx]
            
        match = (pred_ids == pred_gt)
        
        tp = match.sum()
        fp = len(pred_ids) - tp
    #If the system fails identifying the identity -> len(predictions) = 0
    else:
        tp, fp = 0, 0
    return tp, fp

def get_FN(identity, data_info, TP):
    data_id = [d for d in data_info if d[1] == identity]
    
    total_img_id = 0
    for d in data_id:
        tracklets = d[0]
        total_img_id = total_img_id + len(tracklets)
    fn = total_img_id - TP
    return fn

def get_precision(predictions, groundTruth):
    #Label Accuracy
    predictions = np.array(predictions)
    groundTruth = np.array(groundTruth)

    match = (predictions == groundTruth)
    precision = match.sum()/len(match)
    return precision, len(match)

def get_recall(predictions, groundTruth, total_imgs):
    predictions = np.array(predictions)
    groundTruth = np.array(groundTruth)
    
    match = (predictions == groundTruth)
    TP = match.sum()
    
    recall = TP /total_imgs 
    return recall

def get_recall_clasif(predictions, groundTruth, failures):
    predictions = np.array(predictions)
    groundTruth = np.array(groundTruth)
    
    match = (predictions == groundTruth)
    TP = match.sum()
    
    total_imgs = TP + len(failures)
    recall = TP / total_imgs 
    return recall

###########################################################################################
###########################################################################################

def get_V_measure(predictions, groundTruth, a, size, it):
    
    with open(dir_saveModels + a + '/' + size + '/' + it + '/final_unknown_samples.json') as f:
        unknown_data = json.load(f) 
        
    groundTruth_unknown, dir_data = parse_unknown_for_eval(unknown_data)
    gt_people = [gt for gt in groundTruth_unknown if gt != 0]  
    
    id_max = max(predictions)
    pred_unknown = [id_max + (i+1) for i in range(len(gt_people))]    
    # predictions = predictions + pred_unknown
    # groundTruth = groundTruth + gt_people
    
    assert len(predictions) == len(groundTruth)
    homogeneity, completness, v_measure = metrics.homogeneity_completeness_v_measure(groundTruth, predictions)
    contingency_matrix = metrics.cluster.contingency_matrix(groundTruth, predictions)
    purity = np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix) 
    
    return homogeneity, completness, v_measure, purity


def get_RandIndex(predictions, groundTruth, gt_unknown_data):
    id_max = max(predictions)
    pred_unknown = [id_max + (i+1) for i in range(len(gt_unknown_data))]    
    predictions = predictions + pred_unknown
    
    groundTruth = groundTruth + gt_unknown_data
    homogeneity, completness, v_measure = metrics.homogeneity_completeness_v_measure(groundTruth, predictions)
    
    return homogeneity, completness, v_measure

def precision_recall_normalized(clusters, gt_identities, predictions, groundTruth_prediction, data_info):
    assert len(predictions) == len(groundTruth_prediction), 'Error:size of predictions {} and groundtruth {} not match'.format(len(pids), len(groundTruth))
    
    id_clusters_initialized, amount_clusters_same_id = np.unique(clusters, return_counts = True) 
    id_scenario, amount_clusters_same_id = get_info_scenario(id_clusters_initialized, gt_identities, amount_clusters_same_id)
    n_idx = np.where(np.array(amount_clusters_same_id) > 1)
    print('num identities with more than 1 cluster', len(n_idx[0]))
    
    precision_ids, recall_ids = [], []
    for i, identity in enumerate(id_scenario):        
        TP, FP = get_TP_FP(identity, predictions, groundTruth_prediction)
        FN = get_FN(identity, data_info, TP)
        
        num_clusters_same_id = amount_clusters_same_id[i]
        
        if num_clusters_same_id == 0:
            precision_norm = 0
            recall_norm = 0
        else:            
            precision_norm = (1 / num_clusters_same_id) * (TP / (TP + FP)) 
            recall_norm = (1 / num_clusters_same_id) * (TP / (TP + FN))
        
        precision_ids.append(precision_norm)
        recall_ids.append(recall_norm)
    avg_precision = sum(precision_ids) / len(id_scenario)
    avg_recall = sum(recall_ids) / len(id_scenario)
    # plot_precision_recall(precision_ids, recall_ids, id_scenario)
    return avg_precision, avg_recall


def eval_itemize(a, size, it, ids):
    dir_people_noise = '/home/scasao/Documents/0_DATASET/REID/People/Mars/Noise/People_Noise'
    dir_real_noise = '/home/scasao/Documents/0_DATASET/REID/People/Mars/Noise/Real_Noise'
    
    people_noise = np.sort(os.listdir(dir_people_noise))
    real_noise = np.sort(os.listdir(dir_real_noise))
            
    people_noise_gt = [p.replace('.jpg','') for p in people_noise]
    real_noise_gt = [r.replace('.jpg','') for r in real_noise]
    
    #Metrics in the classification process
    all_clasif_discards, all_GalleryPrecision, all_clusters = [], [], []
    all_noiseClasif, all_noiseSel, all_noiseName, all_personName = [],[],[],[]     
    all_pred_clasif, all_gt_clasif, all_pred_merge, all_gt_merge, all_pred_init, all_gt_init = [],[],[],[],[],[]
    num_merge = 0
    
    #Unify all data in diferent files in one unique list evaluate_rank
    name_files = os.listdir(dir_saveMetric + a + '/' + size + '/' + it)
    name_files = np.sort(name_files)
    for name_file in name_files:
        with open(dir_saveMetric + a + '/' + size + '/' + it + '/' + name_file) as f:
            data = json.load(f)
        #Accuracy of class prediction an gallery accuracy
        all_pred_clasif = getDataList(data['pred_clasif'], all_pred_clasif)
        all_gt_clasif = getDataList(data['gt_clasif'], all_gt_clasif)
        #Acc merge
        all_pred_merge = getDataList(data['pred_merge'], all_pred_merge)
        all_gt_merge = getDataList(data['gt_merge'], all_gt_merge)
        #Acc init
        all_pred_init = getDataList(data['pred_init'], all_pred_init)
        all_gt_init = getDataList(data['gt_init'], all_gt_init)
        
        #Discards in classification
        all_clasif_discards = getDataList(data['discard_clasif'], all_clasif_discards)
        
        #Gallery Accuracy
        all_GalleryPrecision = getDataList(data['GalleryPrecision'], all_GalleryPrecision)
        
        #Noise Analysis
        all_noiseClasif = getDataList(data['Distractors_DiscardedClasf'], all_noiseClasif)
        all_noiseSel = getDataList(data['Distractors_Discarded_DS'], all_noiseSel)
        all_personName = getDataList(data['personName'], all_personName)      
        all_noiseName = getDataList(data['noiseName'], all_noiseName)
        
        #Clusters created
        all_clusters = getDataList(data['cluster_labels'], all_clusters)
    num_merge = data['num_merge']
    
    with open(dir_saveModels + a + '/' + size + '/' + it + '/final_unknown_samples.json') as f:
        unknown_data = json.load(f) 
    
    groundTruth, dir_data = parse_unknown_for_eval(unknown_data)
    gt_people = [gt for gt in groundTruth if gt != 0]
    
    # Precision 
    precision_clasif, n_clasif = get_precision(all_pred_clasif,all_gt_clasif)
    precision_merge, n_merge = get_precision(all_pred_merge,all_gt_merge)
    precision_init, n_init = get_precision(all_pred_init,all_gt_init)
    
    #Recall classification
    failures = [d for d in all_clasif_discards if d == 0]
    recall_clasif, n_true = get_recall(all_pred_clasif, all_gt_clasif, failures)
    
    total_pred_merge = all_pred_merge + all_pred_init
    total_gt_merge = all_gt_merge + all_gt_init
    recall_merge, n_true_m = get_recall(total_pred_merge, total_gt_merge, gt_people)
    
    #Gallery Precision
    all_GalleryPrecision = np.array(all_GalleryPrecision)
    FinalGalleryPrecision = np.sum(all_GalleryPrecision)/len(all_GalleryPrecision)
    
    #Noise Analysis
    DiscardNoiseClas = sum(all_noiseClasif)/len(all_noiseClasif)
    DiscardNoiseSel = sum(all_noiseSel)/len(all_noiseSel)
    
    people_noise_precision, people_noise_recall = get_noise_eval(all_personName, people_noise_gt)
    real_noise_precision, real_noise_recall = get_noise_eval(all_noiseName, real_noise_gt)
    
    #Clusters Analysis
    if len(ids[it]) > 0:
        labeled_clusters = load_identities(ids[it])
    else:
        labeled_clusters = []
    
    id_clusters_initialized = list(set(all_clusters))
    new_clusters_initialized = [i for i in id_clusters_initialized if i not in labeled_clusters]
    
    plot_clusters_frequency(all_clusters, labeled_clusters, a, size, it)
    Final_Clusters_Analysis(a, size, it)
    # plot_TSNE_unknown(groundTruth, dir_data, a + '_' + size + '_' + it + '_unknown2.png' )
    print('** Results **')
    print('Accuracy of {} images included in the gallery: {:.1%}'.format(len(all_GalleryPrecision),FinalGalleryPrecision))
    print('Analysis of the label assigned:')
    print('-> Precision in the classification process of {} images in total: {:.1%}'.format(n_clasif,precision_clasif))
    print('-> Precision in the merge process of {} images in total done {} times: {:.1%}'.format(n_merge,num_merge,precision_merge))
    print('-> Precision in the initializacion class process of {} images in total: {:.1%}'.format(n_init,precision_init))
    print('-> Recall in the classification process of {} true images: {:.1%}'.format(n_true, recall_clasif))
    print('-> Recall in the "repesca" processs of {} true images: {:.1%}'.format(n_true_m, recall_merge))
    print('Noise analysis:')
    print('-> % of noise samples (over all noise) discarded by the classification {:.2%}'.format(DiscardNoiseClas))
    print('-> % of noise samples (over the noise that arrive to the model) discarded by the data selection {:.2%}'.format(DiscardNoiseSel))
    print('-> {:.2%} precision and {:.2%} recall of skeletons filtering people'.format(people_noise_precision,people_noise_recall))
    print('-> {:.2%} precision and {:.2%} recall of skeletons filtering noise'.format(real_noise_precision,real_noise_recall))
    print('Cluster analysis:')
    print('-> Total of clusters initialized:', len(all_clusters))
    print('-> New clusters initialize correctly:', len(new_clusters_initialized))
    print('Unknown analysis:')
    print('-> Total of unknown data:', len(groundTruth))
    print('-> Labeled people in unknown data:', len(gt_people))
