import torch
import json 
import os
import numpy as np
import random

from ComponentsGallery import Gallery

# dir_features = '/home/scasao/Documents/0_DATASET/REID/People/Mars_normalize/gallery/'    

def select_known_identities(config, data = None, num_selected_ids =  None, name_file = None):
    selection_method = config.ids_initilized
    if selection_method == 'half':
        class_labels = get_half_IDS(data)
    elif selection_method == 'random':
        class_labels = get_random_IDS(data, num_selected_ids)
    elif selection_method == 'ids_preselected':
        class_labels = load_identities(name_file)
    elif selection_method == 'all':
        class_labels = get_all_identities(data)
    elif selection_method == 'unsupervised':
        class_labels = []
    return class_labels
   
def get_all_identities(data): 
    listIDS = [f[1] for f in data] 
    class_labels = list(set(listIDS))
    class_labels.remove(0)
    
    return class_labels

def load_identities(file):
    f = open('./Iterations/' + file, 'r')
    identities = f.readlines()
    identities = [int(i) for i in identities]

    return identities

def get_half_IDS(data):
    listIDS = [f[1] for f in data] 
    class_labels = list(set(listIDS))
    class_labels = [l for i,l in enumerate(class_labels) if i%2 == 0 and l != 0]
    
    return class_labels

def get_random_IDS(data, num_selected_ids):
    listIDS = [f[1] for f in data] 
    class_labels = list(set(listIDS))
    class_labels.remove(0)
    total_ids = len(class_labels)
    
    random_list = random.sample(range(total_ids), num_selected_ids)
    class_labels = [l for i,l in enumerate(class_labels) if i in random_list]

    assert len(class_labels) == num_selected_ids

    save_ids_selected('random_' + str(num_selected_ids), class_labels)
    return class_labels

def save_ids_selected(name_file, ids_selected):
    with open('./Iterations/' + name_file + '.txt', 'w') as file:
        for id_s in ids_selected:
            file.write('%s\n' % id_s)
    
def extract_info_iteration(iteration, dir_features):
    
    data = np.load('./Iterations/' + iteration, allow_pickle=True)
    data = data.tolist()
    data = [tuple(f) for f in data]
    
    tracklets = []
    for d in data:             
        name_features, id_gt, camid = d
        path_features = []
        for name_feat in name_features:
            path_feat = dir_features + '/gallery/' + name_feat 
            path_features.append(path_feat)
        path_features = tuple(path_features) 
        tracklets.append((path_features,id_gt,camid))    
    return tracklets

def extract_info_iteration_newModels(iteration, dir_features):
    data = np.load('./Iterations/' + iteration, allow_pickle=True)
    data = data.tolist()
    data = [tuple(f) for f in data]
    
    tracklets = []
    for d in data:             
        name_features, id_gt, camid = d
        path_features = []
        for path_feat in name_features:
            path_features.append(path_feat)
        path_features = tuple(path_features) 
        tracklets.append((path_features,id_gt,camid))    
    return tracklets

def extract_info_iteration_distributed(iteration, dir_features):
    distributed_data = {}
    with open('./Iterations/distributed_iterations/' + iteration) as json_file:
        data = json.load(json_file)
    
    for camid in data.keys():
        data_camid = data[camid]
        data_camid = data_camid.tolist()
        data_camid = [tuple(f) for f in data_camid]
    
        tracklets = []
        for d in data:             
            name_features, id_gt, camid = d
            path_features = []
            for path_feat in name_features:
                path_features.append(path_feat)
            path_features = tuple(path_features) 
            tracklets.append((path_features, id_gt, camid))   
        distributed_data[camid] = tracklets
    return distributed_data
    
def extract_info_query(data_loader):
    """
    Prepares query info to evaluation
    Return: tensor of features, array with IDS, array with camIDS
    """
    f, pids, camids = [], [], []
    for index, d in enumerate(data_loader):
        dir_features, pid, camid = d
        for dir_feat in dir_features:
            feature = np.load(dir_feat)
            feature = torch.from_numpy(feature)
            f.append(feature)
            pids.append(pid)
            camids.append(camid)
    f = torch.cat(f, 0)
    pids = np.array(pids)
    camids = np.array(camids)
    return f, pids, camids

def extract_info_gallery(dir_saveModels, algorithm, size, iteration):
    """
    Prepares gallery info to evaluation
    Return: tensor of features, array with IDS, array with camIDS
    """
    ids = sorted(os.listdir(dir_saveModels + algorithm + '/' + size + '/' + iteration + '/'))
    ids.remove('final_unknown_samples.json')
    f, pids, labels, groundTruth, camids, dir_feats = [], [], [], [], [], []
    for g_id in ids:
        with open(dir_saveModels + algorithm + '/' + size + '/' + iteration + '/' + g_id) as outfile:
            data = json.load(outfile)               
        gdir, label, pid, gt, camid = parse_data_for_eval(data)
        camids = camids + camid
        groundTruth = groundTruth + gt
        dir_feats = dir_feats + gdir
        for gd in gdir:
            feature = np.load(gd)
            feature = torch.from_numpy(feature)
            f.append(feature)
            pids.append(pid)
            labels.append(label)
    f = torch.cat(f, 0)
    pids = np.array(pids)
    labels = np.array(labels)
    camids = np.array(camids)
    groundTruth = np.array(groundTruth)
    dir_feats = np.array(dir_feats)
    return f, labels, pids, groundTruth, camids, dir_feats

def extract_info_clusters(dir_saveModels, algorithm, size, iteration):
    """
    Prepares gallery info to evaluation
    Return: tensor of features, array with IDS, array with camIDS
    """
    ids = sorted(os.listdir(dir_saveModels + algorithm + '/' + size + '/' + iteration + '/'))
    ids.remove('final_unknown_samples.json')
    final_clusters = []
    for g_id in ids:
        with open(dir_saveModels + algorithm + '/' + size + '/' + iteration + '/' + g_id) as outfile:
            data = json.load(outfile)               
        gdir, label, pid, gt, camid = parse_data_for_eval(data)
    
        final_clusters.append((tuple(gdir), tuple(gt), tuple(camid), label, pid))
    return final_clusters

def parse_data_for_eval(data):
    gdir = data['dir_model']
    pid = data['pid']
    gt = data['gt']
    camid = data['camid']
    label = data['model_label']
    return gdir, label, pid, gt, camid
                   
def parse_data_for_save(model):
    dir_feat = [c.path for c in model.samples]
    gt = [int(c.gt) for c in model.samples]
    camid = [int(c.camera) for c in model.samples]
    
    data = {'dir_model': dir_feat, 
            'pid': int(model.identity), 
            'model_label': int(model.label),
            'gt': gt, 
            'camid': camid}
    return data

def parse_unknown_for_save(unknown_samples):
    dir_feat = [c.path for c in unknown_samples]
    gt = [int(c.gt) for c in unknown_samples]
    
    data = {'dir_model': dir_feat, 
            'gt': gt}
    return data

def parse_unknown_for_eval(data):
    gt = data['gt']
    dir_feats = data['dir_model']
    
    return gt, dir_feats


def get_distributed_data(data):
    list_cameras = [f[2] for f in data] 
    id_cameras = list(set(list_cameras))
    
    distributed_data, distributed_gallery = {}, {}
    for id_c in id_cameras:
        data_camid = [f for f in data if f[2] == id_c]
        gallery = Gallery(id_c)
        
        distributed_data[id_c] = data_camid
        distributed_gallery[id_c] = gallery
    return distributed_gallery, distributed_data, id_cameras
             
def getDataList(data, final_list):
    for i in range(len(data)):
        final_list.append(data[i])
    return final_list

def get_precision(predictions, groundTruth):
    #Label Accuracy
    predictions = np.array(predictions)
    groundTruth = np.array(groundTruth)

    match = (predictions == groundTruth)
    precision = match.sum()/len(match)
    return precision, len(match)

def get_recall(predictions, groundTruth, errors):
    predictions = np.array(predictions)
    groundTruth = np.array(groundTruth)
    
    match = (predictions == groundTruth)
    
    TP = match.sum()
    FN = len(errors)
    
    recall = TP /(TP+FN) 
    return recall, TP+FN

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

def get_num_samples(data):
    n, t = 0, 0
    for d in data:                         
        path_features, id_gt, camid = d
        t = t + len(path_features)
        if id_gt != 0:
            n =  n + len(path_features)
    return n, t 
    

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    