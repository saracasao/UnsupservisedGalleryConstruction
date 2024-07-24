import torch
import json
import os
import numpy as np


def load_identities(file):
    f = open('./Iterations/' + file, 'r')
    identities = f.readlines()
    identities = [int(i) for i in identities]

    return identities


def get_local_path(path_iter_sample, path_dataset):
    split_path = path_iter_sample.split('/')
    index = split_path.index('People')
    selected_path = os.path.join(*split_path[index + 1::])
    complete_path = path_dataset + selected_path

    assert os.path.isfile(complete_path)
    return complete_path


def extract_info_iteration(config):
    iteration = config.iteration
    data = np.load('./Iterations/' + iteration, allow_pickle=True)
    data = data.tolist()
    data = [tuple(f) for f in data]
    
    tracklets = []
    for d in data:             
        name_features, id_gt, camid = d
        path_features = []
        for path_feat in name_features:
            path = get_local_path(path_feat, config.dir_dataset)
            path_features.append(path)
        path_features = tuple(path_features) 
        tracklets.append((path_features,id_gt,camid))    
    return tracklets

    
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


def extract_info_gallery(dir_save_models):
    """
    Prepares gallery info to evaluation
    Return: tensor of features, array with IDS, array with camIDS
    """
    ids = sorted(os.listdir(dir_save_models))
    ids.remove('final_unknown_samples.json')
    f, pids, labels, camids = [], [], [], []
    for g_id in ids:
        with open(dir_save_models + '/' + g_id) as outfile:
            data = json.load(outfile)               
        gdir, label, pid, gt, camid = parse_data_for_eval(data)
        camids = camids + camid
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
    return f, labels, camids


def extract_info_clusters(dir_saveModels):
    """
    Prepares gallery info to evaluation
    Return: tensor of features, array with IDS, array with camIDS
    """
    ids = sorted(os.listdir(dir_saveModels))
    ids.remove('final_unknown_samples.json')
    final_clusters = []
    for g_id in ids:
        with open(dir_saveModels + '/' + g_id) as outfile:
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

             
def getDataList(data, final_list):
    for i in range(len(data)):
        final_list.append(data[i])
    return final_list
    

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    