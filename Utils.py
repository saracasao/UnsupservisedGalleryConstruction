import numpy as np
import torch
import json 
import os
import pandas as pd
import matplotlib.pyplot as plt
import random

from scipy.spatial import distance
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from PIL import Image, ImageDraw

dir_images = '/home/scasao/Documents/0_DATASET/REID/People/Mars/bbox_test/'  
dir_features = '/home/scasao/Documents/0_DATASET/REID/People/Mars_normalize/gallery/'    

def JS_divergence(P,Q):
    matrix_js = []
    for i, p_i in enumerate(P):
        M = 0.5*(p_i+Q)

        kl_PM = KL_divergence(p_i,M)
        kl_QM = KL_divergence(Q,M)
        
        js = 0.5*kl_PM + 0.5*kl_QM 
        matrix_js.append(js)
    matrix_js = np.array(matrix_js)
    assert matrix_js.shape[0] == P.shape[0] and matrix_js.shape[1] == Q.shape[0]
    return matrix_js

def KL_divergence(P,Q):
    kl = np.sum(P*np.log2(P/Q), axis = 1)
    return kl

def update_inter_cluster_distance(app_models, model, inter_cluster_dist):
    samples = model.samples
    features = [s.feat for s in samples]
    
    n_updates = np.array([m.n_newModel for m in app_models])
    diff_updates = n_updates - model.ref_updateModels
    DistancesTOupdate = np.where(diff_updates != 0)[0]
    
    inter_cluster_dist = get_InterClusterDistances(app_models, features, inter_cluster_dist, DistancesTOupdate)
    
    return inter_cluster_dist, n_updates

def get_Centroid(samples):
    """
    Given a model of a class return the mean of the features (centroid)
    """
    features    = [s.feat for s in samples]
    feature_arr = np.array(features)
    class_mean  = np.mean(feature_arr, axis=0)
    return class_mean

def get_wCentroid(samples):
    """
    Given a model of a class return the mean of the features (centroid)
    """
    features    = [s.feat for s in samples]
    pct_joints  = [s.perct_key_points for s in samples]
    w_feats = [i*j for i,j in zip(pct_joints,features)]

    num = np.sum(w_feats, axis = 0)
    den = np.sum(pct_joints)
    centroid = num/den
    
    feature_arr = np.array(features) 
    assert centroid.shape == feature_arr[0].shape
    return centroid

def get_DiversityModel(samples):
    """
    Given a model of a class return the minimum distance within the samples (diversity)
    """
    distmat = get_IntraClustersDistances(samples)
    np.fill_diagonal(distmat,0)
    diversity = np.amin(np.where(distmat>0, distmat, np.inf))

    return diversity

def get_DiversityQuery(query, samples):
    features = [s.feat for s in samples]
    query_distmat  = distance.cdist([query.feat], features,'cosine')[0]
    query_diversity = np.min(query_distmat)

    return query_diversity

def get_DiversitySamples(samples):
    distmat = get_IntraClustersDistances(samples)
    np.fill_diagonal(distmat,0)  
    distmat = np.where(distmat>0, distmat, np.inf)
    
    samples_diversity = distmat.min(axis=1)  
    return samples_diversity

def get_DistCentroids(app_models, query):
    centroids = [m.centroid for m in app_models]
    distmat = distance.cdist([query.feat], centroids,'cosine')
    return distmat

def get_InterClusterDistances(app_models, features, clustersDistance, indexTOupdate):   
    if len(indexTOupdate) > 0:    
        modelsTOupdate = [m for i, m in enumerate(app_models) if i in indexTOupdate]
        centroids = [m.centroid for m in modelsTOupdate]
        distmat = distance.cdist(features, centroids,'cosine')   
        for i in range(len((indexTOupdate))):
            clustersDistance[:,indexTOupdate[i]] = distmat[:,i]      
    return clustersDistance

def get_IntraClustersDistances(samples):
    features = [s.feat for s in samples] 
    distmat  = distance.cdist(features, features,'cosine')
    
    return distmat

def get_Hmodel(model_index, features, samples, int_clusterD, num_models = None):     
    #Update values
    new_centroid = get_wCentroid(samples) 
    dist_new_centroid = distance.cdist([new_centroid], features,'cosine')
    int_clusterD[:,model_index] = dist_new_centroid
    
    P = get_Pmodel(int_clusterD, num_models)
    P = np.where(P == 0, 1, P)

    H_i = (-1)*np.sum(P*np.log2(P), axis = 1)

    assert len(H_i) == len(samples), 'Error in entrophy dimensions: len entrophy {} but len samples {}'.format(len(H_i), len(samples))
    
    idx_labeled = [i for i, s in enumerate(samples) if s.labeled == 1]
    H_i[idx_labeled] = 0
    
    return H_i

def get_Pmodel(D, num_models = None):    
    #Get Probabilies
    D = D/0.1 
    num = np.exp((-1)*D)
    den = np.sum(num, axis = 1) 
    den.shape = (D.shape[0], 1)
    P = num/den
    # P = P_normalization(P, num_models)

    assert round(np.sum(P[0]), 0) == 1
    return P

def P_normalization(P, num_models):
    P_maxH = 1 / num_models
    thr_negligible = P_maxH * 10**(-2)
    P_thr = np.where(P < thr_negligible, 0, P)

    if P_thr.ndim == 1:
        P_norm = P_thr / np.sum(P_thr)
        assert round(np.sum(P_norm),0) == 1.
    elif P_thr.ndim == 2:
        den = np.sum(P_thr, axis = 1)
        den.shape = (den.shape[0], 1)
        P_norm = P_thr / den
        assert round(np.sum(P_norm[0]), 0) == 1.

    assert P_norm.shape == P.shape 
    return P_norm
    
def plot_clusters_frequency(all_clusters, labeled_clusters, a, size, it):
    _, name_plot = a.split('/')
    total_cluster = len(all_clusters)
    
    id_clusters_initialized, amount_clusters_same_id = np.unique(all_clusters, return_counts = True)
    
    if 'unsupervised' not in it:
        #Labeled clusters
        info_label = [[i,a] for i,a in zip(id_clusters_initialized, amount_clusters_same_id) if i in labeled_clusters]
        info_label = np.array(info_label)
        _, amount_clusters_labeled_id = info_label[:,0],info_label[:,1]
    
        #New clusters
        info_new = [[i,a] for i,a in zip(id_clusters_initialized,amount_clusters_same_id) if i not in labeled_clusters]
        info_new = np.array(info_new)
        _, amount_clusters_new = info_new[:,0],info_new[:,1]
        
        #Frequency of clusters
        freq_labeled, num_clusters_labeled = np.unique(amount_clusters_labeled_id, return_counts = True)
        freq_new, num_clusters_new = np.unique(amount_clusters_new, return_counts = True)
        max_freq = max(max(freq_labeled), max(freq_new))
    
        X = list(range(1,max_freq))
        num_l = []
        for f in X:
            if f in freq_labeled:
                idx = np.where(freq_labeled== f)
                num_l.append(int(num_clusters_labeled[idx]))
            else:
                num_l.append(0)
                
        num_n = []
        for f in X:
            if f in freq_new:
                idx = np.where(freq_new== f)
                num_n.append(int(num_clusters_new[idx]))
            else:
                num_n.append(0)
                
        max_num_models = [i+j for i,j in zip(num_l, num_n)]
        max_num_models = max(max_num_models)
        
        plt.figure()
        x_ticks = np.arange(0,max_freq, 1)
        plt.xticks(x_ticks)
        y_ticks = np.arange(0, max_num_models + 5, 20)
        plt.yticks(y_ticks)
        plt.ylim(0,max_num_models + 1)
        plt.bar(X, num_l, label = 'labeled')
        plt.bar(X, num_n, bottom=num_l, label = 'new')
        plt.legend()
    
        plt.savefig('./Custer_Distributions/' + name_plot + '_' + size + '_' + it + 'newVSlabeled.png')
    
    frequency, amount_clusters = np.unique(amount_clusters_same_id, return_counts = True)
    pct_total = [num_clusters_sameID/total_cluster for num_clusters_sameID in amount_clusters]
    
    plt.figure()
    y_ticks = np.arange(0, 0.5, 0.05)
    plt.yticks(y_ticks)
    plt.ylim(0, 0.5)
    plt.bar(frequency, pct_total)
    plt.legend()

    plt.savefig('./Custer_Distributions/' + name_plot + '_' + size + '_' + it + 'percentage.png')
    
def plot_TSNE_unknown(groundTruth, dir_data, name_file):
    people_dir_data = []
    for gt, df in zip(groundTruth, dir_data):
        if gt != 0:
            people_dir_data.append(df)
        
    features, f_dir, label = [], [], []
    for i, df in enumerate(people_dir_data):
        if i%2 == 0:
            feat = np.load(df)[0]
            features.append(feat)
            f_dir.append(df)
            label.append(0)
    
    TSNE_clusters(features, label, f_dir, name_file)
    

def TSNE_clusters(gf, g_pids, g_dir, name_file):
    X = gf
    Y = g_pids
    
    dir_imgs = []
    for gd in g_dir:
        dir_img = featTOimage(gd)
        dir_imgs.append(dir_img)  
       
    print('len data', len(X))
    X = np.array(X)    
    feat_cols = ['pixel'+str(i) for i in range(X.shape[1])]
    
    df = pd.DataFrame(X,columns=feat_cols)
    print('Size of the dataframe: {}'.format(df.shape))
    
    pca_50 = PCA(n_components=50)
    pca_result_50 = pca_50.fit_transform(df[feat_cols].values)
    print('Cumulative explained variation for 50 principal components: {}'.format(np.sum(pca_50.explained_variance_ratio_)))
    
    tsne = TSNE(n_components=2, verbose=0, perplexity=100, n_iter=1000)
    tsne_pca_results = tsne.fit_transform(pca_result_50)
    
    #DIBUJAR IMÁGENES
    n = list(set(g_pids))
    color_dict = get_color_clusters(n)
    tx, ty = tsne_pca_results[:,0], tsne_pca_results[:,1]
    tx = (tx-np.min(tx)) / (np.max(tx) - np.min(tx))
    ty = (ty-np.min(ty)) / (np.max(ty) - np.min(ty))
    
    width = 4000
    height = 3000
    max_dim = 100
    
    j = 0
    full_image = Image.new('RGBA', (width, height))
    for img, x, y in zip(dir_imgs, tx, ty):
        tile = Image.open(img)
         
        identity = Y[j]
        color = color_dict[identity]
        tile = preprocess_img(tile, color)
        
        rs = max(1, tile.width/max_dim, tile.height/max_dim)
        tile = tile.resize((int(tile.width/rs), int(tile.height/rs)), Image.ANTIALIAS)
        full_image.paste(tile, (int((width-max_dim)*x), int((height-max_dim)*y)), mask=tile.convert('RGBA'))
        j += 1

    full_image.save('./TSNE_RESULTS/' +  name_file)    

def preprocess_img(img, color):
    coord = [(0,0),(img.size[0],0),(img.size[0],img.size[1]),(0,img.size[1])]
    draw = ImageDraw.Draw(img)
    for i in range(len(coord)):
        if i < len(coord)-1:
            p = (coord[i], coord[i+1])
        else:
            p = (coord[i], coord[0])
        draw.line(p, fill = color, width = 10)
    return img

def get_color():
    color_dict = {'CPR_%':'red', 'CPR_%+sharpness':'green', 'CPR_%+intensity': 'blue'}
    return color_dict

def get_color_clusters(n):
    rand_colors = {}
    for i in n:
        rand_colors[i] = ("#"+''.join([random.choice('ABCDEF0123456789') for i in range(6)]))
    return rand_colors
    
def featTOimage(dir_feat):
    name_file = dir_feat.split('/')[-1]
    name_image = name_file.split('.')[0] + '.jpg'
    dir_folder = name_file[0:4]

    dir_img = dir_images + dir_folder + '/' + name_image
    
    return dir_img

def imageTOfeat(dir_img):
    name_file = dir_img.split('/')[-1]
    name_feat = name_file.split('.')[0] + '.npy'

    dir_feat = dir_features + name_feat
    
    return dir_feat





########################################################################################################################
def TSNE_images(data, name_file, algorithm_global, algorithm_sub1, algorithm_sub2, algorithm_sub3):
    X = []
    Y = []
    dir_imgs = []
    
    if 'GTperson_PREDnoise' in name_file:
        n = 1
    else:
        n = 10
        
    name_images_sub1 = data[algorithm_sub1]       
    name_images_sub2 = data[algorithm_sub2]
    name_images_sub3 = data[algorithm_sub3]
    name_images_global = data[algorithm_global]
    print(len(name_images_global))
    for i, name in enumerate(name_images_global):
        if i % n == 0:
            dir_feat = imageTOfeat(name)
            dir_img = featTOimage(dir_feat)
            
            feat = np.load(dir_feat)[0]
            
            X.append(feat)
            dir_imgs.append(dir_img)  
        if name in name_images_sub1:
            Y.append(algorithm_sub1)
        elif name in name_images_sub2:
            Y.append(algorithm_sub2)
        elif name in name_images_sub3:
            Y.append(algorithm_sub3)

    print(len(X))
    X = np.array(X)    
    feat_cols = ['pixel'+str(i) for i in range(X.shape[1])]
    
    df = pd.DataFrame(X,columns=feat_cols)
    print('Size of the dataframe: {}'.format(df.shape))
    
    pca_50 = PCA(n_components=50)
    pca_result_50 = pca_50.fit_transform(df[feat_cols].values)
    print('Cumulative explained variation for 50 principal components: {}'.format(np.sum(pca_50.explained_variance_ratio_)))
    
    tsne = TSNE(n_components=2, verbose=0, perplexity=100, n_iter=1000)
    tsne_pca_results = tsne.fit_transform(pca_result_50)
    
    #DIBUJAR IMÁGENES
    color_dict = get_color()
    tx, ty = tsne_pca_results[:,0], tsne_pca_results[:,1]
    tx = (tx-np.min(tx)) / (np.max(tx) - np.min(tx))
    ty = (ty-np.min(ty)) / (np.max(ty) - np.min(ty))
    
    width = 4000
    height = 3000
    max_dim = 100
    
    j = 0
    full_image = Image.new('RGBA', (width, height))
    for img, x, y in zip(dir_imgs, tx, ty):
        tile = Image.open(img)
         
        # algorithm = Y[j]
        # color = color_dict[algorithm]
        # tile = preprocess_img(tile, color)
        
        rs = max(1, tile.width/max_dim, tile.height/max_dim)
        tile = tile.resize((int(tile.width/rs), int(tile.height/rs)), Image.ANTIALIAS)
        full_image.paste(tile, (int((width-max_dim)*x), int((height-max_dim)*y)), mask=tile.convert('RGBA'))
        j += 1
        
    full_image.save('./TSNE_RESULTS/' +  name_file)


def analysis_discardsGT(all_discards, all_predictions):                    
    #DISCARDS ANALYSIS WITH GROUNDTRUTH
    #Clean data selected to be included in the model
    all_discards = np.array(all_discards)

    predID_Discards = [f for i,f in enumerate(all_predictions) if all_discards[i] != -1]
    gt_discards = [f for f in all_discards if f != -1]
    
    predID_Discards = np.array(predID_Discards)
    gt_discards = np.array(gt_discards)
    
    #Amount of data discarded that is noise 
    noise_discard = np.where(gt_discards == 0)                
    predID_Discards_clean = np.delete(predID_Discards, noise_discard)
    gt_discards_clean = np.delete(gt_discards, noise_discard)

    #Amount of data discarded that the clasification is correct but it is discarded
    error_discards = (predID_Discards_clean == gt_discards_clean)
    error_discards = error_discards.sum()
                 
    #Amount of data discarded that the clasification is incorrect and it is discarded
    correct_discards = len(gt_discards_clean) - error_discards
                    
    print('Total Data Discarded:', len(gt_discards))
    print('-> Noise: {} equal to {:.2%}'.format(len(noise_discard[0]),  len(noise_discard[0])/len(gt_discards)))
    print('-> Correct clasification but discarded: {} equal to {:.2%}'.format(error_discards,error_discards/len(gt_discards)))
    print('-> Incorrect clasification and discarded: {} equal to {:.2%}'.format(correct_discards,correct_discards/len(gt_discards)))
    
    print('ANALYSIS OF DATA CORRECTLY PREDICTED BUT DICARDED')
    info_images = extract_info_images(all_discards, all_predictions)
    percentage, frequency = np.unique(info_images, return_counts = True)
    
    none_info_index = np.where(percentage == 0)
    none_info_freq = frequency[none_info_index][0]
    
    Info_error_discards = error_discards - none_info_freq
    
    print('-> Zero info in the image: {} equal to {:.2%}'.format(none_info_freq, none_info_freq/error_discards))
    print('-> Info in the image greather than zero: {} equal to {:.2%}'.format(Info_error_discards, Info_error_discards/error_discards))


def extract_images(noise, people):
    for p in people:
        path_img = dir_images + p[0:4] + '/' + p + '.jpg'
        img = cv2.imread(path_img)
        print(path_img)
        cv2.imshow('personFailed', img)
        cv2.waitKey(0)  
        
    for n in noise:
        path_img = dir_images + n[0:4] + '/' + n + '.jpg'
        img = cv2.imread(path_img)
        print(path_img)
        cv2.imshow('noiseFailed', img)
        cv2.waitKey(0)  
        
def extract_info_images(discards, predictions):
    dir_masks = '/home/scasao/Documents/0_DATASET/REID/People/Mars/bbox_test_mask/'
    dir_skeleton = '/home/scasao/Documents/0_DATASET/REID/People/Mars/bbox_test_skeleton/'
    
    data = extract_info_iteration('0.npy','/home/scasao/Documents/0_DATASET/REID/People/Mars_normalize')
    path_masks, path_skeletons = [], []
    i = 0
    for d in data:
        paths_features, _, _ = d
        for path_feat in paths_features:
            name = path_feat.split('/')[-1]
            name_mask, _ = name.split('.')
            if discards[i] != -1:
                assert discards[i] == int(name_mask[0:4]), 'FAIL INDEX {} DISCARD {} PATH ID {}'.format(i,  discards[i], int(name_mask[0:4]))
            path_masks.append(dir_masks + name_mask[0:4] + '/' + name_mask + '.jpg')
            path_skeletons.append(dir_skeleton + name_mask[0:4] + '/' + name_mask + '.npy')
            i += 1
    path_skeletons = np.array(path_skeletons)
    path_masks = np.array(path_masks)
    predictions = np.array(predictions)
    discards = np.array(discards)

    error_discards = ~(discards == predictions)
    
    path_skeletons = np.delete(path_skeletons, error_discards)
    path_masks = np.delete(path_masks, error_discards)    

    percentage_infoM, percentage_infoS = [], []
    for m in path_masks:
        mask = cv2.imread(m,0)
        index_info = np.where(mask == 255)
        
        #Percentage of info included in the image
        info_area = len(index_info[0])
        total_area = mask.shape[0] * mask.shape[1]
        if round((info_area / total_area)*100,0) == 0:
            print(m)
            name = m.split('/')[-1]
            name = name.split('.')[0]
            final_name = name + '.jpg'
            
            folder = final_name[0:4]
            img = cv2.imread('/home/scasao/Documents/0_DATASET/REID/People/Mars/bbox_test/' + folder + '/' + final_name)
            cv2.imshow('MASK', img)
            cv2.waitKey(0)
        percentage_infoM.append(round((info_area / total_area)*100,0))
    N = 18 
    for s in path_skeletons:
        sklt = np.load(s, allow_pickle=True)
        if len(sklt) == 0: 
            print(s)
            name = s.split('/')[-1]
            name = name.split('.')[0]
            final_name = name + '.jpg'
            
            folder = final_name[0:4]
            img = cv2.imread('/home/scasao/Documents/0_DATASET/REID/People/Mars/bbox_test/' + folder + '/' + final_name)
            cv2.imshow('Image', img)
            cv2.waitKey(0)
        percentage_infoS.append(round((len(sklt) / N)*100,0))
    print('END')  
    return percentage_infoS








        