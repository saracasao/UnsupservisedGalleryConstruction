import cv2
import re
import numpy as np
import matplotlib
import json
import random
import os
from os import listdir

import time
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image, ImageDraw 
from matplotlib.pyplot import imshow
from matplotlib.colors import ListedColormap

dir_images = '/home/scasao/Documents/DATASET/Re-Id/Mars/bbox_test/'
dir_save_models = '/home/scasao/pytorch/1_Clusters/Mars/ModelsEvolution/'

color_dict = dict({2:'red', 4:'red',52:'red',68:'green',202:'green',196:'green',34:'blue',106:'blue',146:'blue',
         94:'k',156:'k',176:'k',110:'yellow',112:'yellow',142:'yellow',
         308:'magenta',390:'magenta',258:'magenta',18:'darkviolet',24:'darkviolet',26:'darkviolet'})

style_dict = dict({2:0, 4:1,52:2,68:0,202:1,196:2,34:0,106:1,146:2,
         94:0,156:1,176:2,110:0,112:1,142:2,
         308:0,390:1,258:2,18:0,24:1,26:2})

t_sample = 50

class TSNE_model():
    def __init__(self):
        name_file = 'tSNE-points.json'
        with open(name_file) as json_file:
            complete_cluster = json.load(json_file)
            
        self.tsne_pca_results = np.array(complete_cluster['pca'])
        self.dir_imgs = complete_cluster['paths']
        self.ids = np.array(complete_cluster['ids'])

class Visualization():
    def show_model(self, AppModels, ind,  mode, algorithm, metric, model = None, ids_tosave = None):
        if mode =='global':
            for model in AppModels:
                if model.label in ids_tosave:
                    gt = []
                    images = []
                    pid = model.label
                    folder = str(pid)
                    folder = folder.zfill(4)
                    if not os.path.exists(dir_save_models + algorithm + '/' + metric + '/' + folder):
                        os.makedirs(dir_save_models + algorithm + '/' + metric + '/' + folder)
                    
                    total_width = 0
                    max_height = 0
                    for c in model.gallery:
                        dir_feat = c.path
                        name_file = dir_feat.split('/')[-1]
                        name_image = name_file.split('.')[0] + '.jpg'
                        
                        dir_folder = name_file[0:4]
                        dir_img = dir_images + dir_folder + '/' + name_image
    
                        images.append(cv2.imread(dir_img))
                        gt.append(c.gt)
                        if images[-1].shape[0] > max_height: 
                            max_height = images[-1].shape[0]
                        total_width += images[-1].shape[1]
                    final_image = np.zeros((max_height,total_width,3), dtype=np.uint8)
                    current_x = 0
                    for img in images: 
                        final_image[0: max_height, current_x:img.shape[1]+current_x] = img
                        current_x += img.shape[1]
                    
                    name = str(ind)
                    name = name.zfill(8)
                    update = str(model.n_updated)
                    update = update.zfill(4)                                  
                    dir_save_img = dir_save_models + algorithm + '/' + metric + '/' + folder + '/' + name + '_' + update + '.jpg'
                    cv2.imwrite(dir_save_img, final_image)
                    np.save(dir_save_models + algorithm + '/' + metric + '/' + folder + '/' + name + '_' + update + '.npy', np.array(gt))
                
        elif mode == 'local':
            pid = model.label
            folder = str(pid)
            folder = folder.zfill(4)
            if not os.path.exists(dir_save_models + algorithm + '/' + metric + '/' + folder):
                os.makedirs(dir_save_models + algorithm + '/' + metric + '/' + folder)
                    
            gt = []
            images = []
            total_width = 0
            max_height = 0
            for c in model.gallery:
                dir_feat = c.path
                name_file = dir_feat.split('/')[-1]
                name_image = name_file.split('.')[0] + '.jpg'
                
                dir_folder = name_file[0:4]
                dir_img = dir_images + dir_folder + '/' + name_image
                img = cv2.imread(dir_img)

                images.append(cv2.imread(dir_img))
                gt.append(c.gt)
                if images[-1].shape[0] > max_height: 
                    max_height = images[-1].shape[0]
                total_width += images[-1].shape[1]
            final_image = np.zeros((max_height,total_width,3), dtype=np.uint8)
            current_x = 0
            for img in images: 
                final_image[0: max_height, current_x:img.shape[1]+current_x] = img
                current_x += img.shape[1]
  
            update = str(model.n_updated)
            update = update.zfill(4)
            dir_save_img = dir_save_models + algorithm + '/' + metric  + '/' + folder + '/' + update + '.jpg'
            cv2.imwrite(dir_save_img, final_image)
            np.save(dir_save_models + algorithm + '/' + metric  + '/' + folder + '/' + update + '.npy', np.array(gt))
            
    def TSNE(self, AppModels, ind, mode, algorithm, gids, metric, sid = None, n_updated= None):          
        X = []
        y = []  
        dir_imgs = []
        for model in AppModels:
            if model.label in gids:
                pid = model.label
                for c in model.gallery:
                    feat = c.feat
                    dir_feat = c.path
                    
                    name_file = dir_feat.split('/')[-1]
                    name_image = name_file.split('.')[0] + '.jpg'
                    
                    dir_folder = name_file[0:4]
                    dir_imgs.append(dir_images + dir_folder + '/' + name_image)
                   
                    X.append(feat)
                    y.append(pid)
        X = np.array(X)
        y = np.array(y)
        imgs_ids = y

        feat_cols = ['pixel'+str(i) for i in range(X.shape[1])]
        
        df = pd.DataFrame(X,columns=feat_cols)
        df['y'] = y
        df['label'] = df['y'].apply(lambda i: str(i))
        
        pca_50 = PCA(n_components=50)
        pca_result_50 = pca_50.fit_transform(df[feat_cols].values)
  
        tsne = TSNE(n_components=2, verbose=0, perplexity=50, n_iter=1000)
        tsne_pca_results = tsne.fit_transform(pca_result_50)

        #TSNE_points(tsne_pca_results, y, ind, mode, sid, n_updated, algorithm, metric)
        TSNE_images(tsne_pca_results, dir_imgs, imgs_ids, ind, mode, gids, sid, n_updated, algorithm, metric)

    def tsneTOmodel(self, AppModels, tsne_model, ind, mode, algorithm, gids, metric, sid = None, n_updated = None):
        dir_gimgs = []
        g_pids = []
        for model in AppModels:
            if model.label in gids:
                for c in model.gallery:
                    dir_feat = c.path
                    name_file = dir_feat.split('/')[-1]
                    name_image = name_file.split('.')[0] + '.jpg'
                    
                    dir_folder = name_file[0:4]
                    dir_gimgs.append(dir_images + dir_folder + '/' + name_image)
                    g_pids.append(model.label)
                
        dir_imgs = tsne_model.dir_imgs
        tsne_results = tsne_model.tsne_pca_results

        tsne_pca = []
        dir_img_pca = []
        gids_pca = []
        for pca, dir_img in zip(tsne_results,dir_imgs):
            if dir_img in dir_gimgs:
                tsne_pca.append(pca)
                dir_img_pca.append(dir_img)
                
                g_id = g_pids[dir_gimgs.index(dir_img)]
                gids_pca.append(g_id)
        tsne_pca = np.array(tsne_pca)
        gids_pca = np.array(gids_pca)
        
        # TSNE_points(tsne_pca, gids_pca, ind, mode, sid, n_updated, algorithm, metric)
        TSNE_images(tsne_pca, dir_img_pca, gids_pca, ind,  mode, gids, sid, n_updated, algorithm, metric)
        
def TSNE_points(tsne_pca_results, all_ids, ind, mode, sid, n_updated, algorithm, metric):
    plt.figure(str(ind)) 
    
    x = tsne_pca_results[:,0]
    y = tsne_pca_results[:,1]
    
    x_n, y_n, ids_n, marker = [], [], [], []
    for q_id, x_i, y_i in zip(all_ids, x, y): 
        x_n.append(x_i)
        y_n.append((-1)*y_i)
        ids_n.append(q_id)
        marker.append(style_dict[q_id])
    cluster = {'pos_x': x_n, 'pos_y': y_n}
    cluster['y'] = ids_n
    cluster['style'] = marker

    df = pd.DataFrame(data=cluster)
    g = sns.scatterplot(x='pos_x', y='pos_y',
                    hue='y',
                    style = 'style',
                    palette=color_dict,
                    data=df,
                    legend=False,
                    alpha = 0.7)
    name = str(ind)
    name = name.zfill(8)
    
    if mode == 'global':
        # dir_save = dir_save_models + algorithm + '/' + metric + '/' + 'P_'+ name +'.jpg'
        update = str(n_updated)
        update = update.zfill(4)
        dir_save = dir_save_models + algorithm + '/' +  metric + '/' + 'P_'+ name + '_' + update + '.jpg'
        plt.savefig(dir_save, bbox_inches="tight")
    elif mode == 'local':
        update = str(n_updated)
        update = update.zfill(4)
        folder = str(sid)
        folder = folder.zfill(4)
        dir_save = dir_save_models + algorithm + '/' +  metric + '/' + folder + '/' + 'P_'+ name + '_' + update + '.jpg'
        plt.savefig(dir_save, bbox_inches="tight")
    plt.close('all')
    
def TSNE_images(tsne_pca_results, dir_images, all_ids, ind, mode, gids, sid, n_updated, algorithm, metric):
    colors = get_colors(gids)    
    tx, ty = tsne_pca_results[:,0], tsne_pca_results[:,1]
    tx = (tx-np.min(tx)) / (np.max(tx) - np.min(tx))
    ty = (ty-np.min(ty)) / (np.max(ty) - np.min(ty))
    
    width = 4000
    height = 3000
    max_dim = 100
    
    full_image = Image.new('RGBA', (width, height))
    i = 0
    for img, x, y in zip(dir_images, tx, ty):
        tile = Image.open(img)
        pid = all_ids[i]
        color = colors[pid]
        tile = preprocess_img(tile, color)
        
        rs = max(1, tile.width/max_dim, tile.height/max_dim)
        tile = tile.resize((int(tile.width/rs), int(tile.height/rs)), Image.ANTIALIAS)
        full_image.paste(tile, (int((width-max_dim)*x), int((height-max_dim)*y)), mask=tile.convert('RGBA'))
        i += 1
        
    name = str(ind)
    name = name.zfill(8)
    if mode == 'global':
        dir_save = dir_save_models + algorithm + '/' + metric + '/' + 'I_'+ name +'.png'
        # update = str(n_updated)
        # update = update.zfill(4)
        # dir_save = dir_save_models + algorithm + '/' +  metric + '/' + 'I_'+ name + '_' + update + '.png'
        full_image.save(dir_save)
    elif mode == 'local':
        update = str(n_updated)
        update = update.zfill(4)
        folder = str(sid)
        folder = folder.zfill(4)
        dir_save = dir_save_models + algorithm + '/' + metric + '/' + folder + '/' + 'I_'+ name + '_' + update +'.png'
        full_image.save(dir_save)

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

def get_colors(ids):
    color_dict = dict({2:'red',4:'firebrick',52:'tomato',68:'green',202:'lawngreen',196:'lightgreen',34:'blue',106:'darkblue',146:'aquamarine',
         94:'black',156:'gray',176:'silver',110:'yellow',112:'khaki',142:'gold',
         308:'magenta',390:'pink',258:'deeppink',18:'darkviolet',24:'blueviolet',26:'rebeccapurple'})
    return color_dict

def plot_pseudolabels(cmc_info, mAP_info,acc_info,algorithms, path_saveMetric):
    print('Creating plot...')
    plt.close('all')
    # plt.figure()
    # for a in algorithms:
    #     for m in cmc_info[a].keys():  
    #         x, y = [],[]
    #         all_cmc = cmc_info[a][m]
    #         n_samples = int((len(all_cmc) / t_sample)) + 1
    #         for i in range(n_samples):          
    #             cmc_cum = all_cmc[:((i*t_sample) + 1)] 
    #             cmc_cum = np.array(cmc_cum).astype(np.float32)
    #             cmc = cmc_cum.sum(0) / len(cmc_cum)
            
    #             x.append(100*(i*t_sample)/len(all_cmc))
    #             y.append(cmc[0])
    #         plt.plot(x, y, label = a +'_'+ m)
    # # plt.ylim(ymin=0)
    # plt.legend(bbox_to_anchor=(1.05, 1))
    # plt.title('RANK1')
    # plt.savefig(path_saveMetric + 'cmc.eps', format = 'eps', bbox_inches="tight")
    # plt.close('all')
    # print('rank1 save')
    
    # plt.figure()
    # for a in algorithms:
    #     for m in mAP_info[a].keys():   
    #         x, y = [],[]
    #         all_AP = mAP_info[a][m]     
    #         n_samples = int((len(all_AP) / t_sample)) + 1 
    #         for i in range(n_samples):
    #             AP_cum = all_AP[:((i*t_sample) + 1)]
    #             mAP = np.mean(AP_cum)
            
    #             # x.append(i*t_sample)
    #             x.append(100*(i*t_sample)/len(all_cmc))
    #             y.append(mAP)
       
    #         plt.plot(x, y, label = a + '_'+ m)
    # # plt.ylim(ymin=0)
    # plt.legend(bbox_to_anchor=(1.05, 1))
    # plt.title('mAP')
    # plt.savefig(path_saveMetric + 'mAP.eps', format = 'eps', bbox_inches="tight")
    # plt.close('all')
    # print('mAP save')
    
    plt.figure()
    for a in algorithms:
        print(a)
        for m in acc_info[a].keys():  
            x, y = [],[]
            all_acc = acc_info[a][m]
            n_samples = int((len(all_acc) / t_sample)) + 1 
            for i in range(n_samples):
                acc_cum = all_acc[:((i*t_sample) + 1)]
                acc = acc_cum.sum()/len(acc_cum)
                
                # x.append(i*t_sample)
                x.append(100*(i*t_sample)/len(all_acc))
                y.append(100*acc)
            x.pop(0)
            y.pop(0)
            print(x[0:20])
            print(y[0:20])
            plt.plot(x, y, label = a)
    # plt.ylim(ymin=0)
    # plt.legend(bbox_to_anchor=(1.05, 1))
    # plt.title('acc')
    plot.tick_params(axis='x', labelsize=14)
    plot.tick_params(axis='y', labelsize=14)
    plt.ylabel('Pseudo-Label Precision (%)')
    plt.xlabel('Data Analyzed Ratio (%)')
    plt.savefig(path_saveMetric + 'Pseudo-Label_Precision.eps',format = 'eps', bbox_inches="tight")
    plt.close('all')
    print('acc pseudo label save')
    
def plot_accClasify(accConfidence,algorithms, path_saveMetric):
    print('Creating plot...')
    plt.close('all')
    plt.figure()
    for a in algorithms:
        for m in accConfidence[a].keys():    
            x, y = [],[]
            all_accClasif = accConfidence[a][m]
            n_samples = int((len(all_accClasif) / t_sample)) + 1 
            for i in range(n_samples):
                acc_cum = all_accClasif[:((i*t_sample) + 1)]
                acc = acc_cum.sum()/len(acc_cum)
                
                # x.append(i*t_sample)
                x.append(100*(i*t_sample)/len(all_accClasif))
                y.append(acc)
            plt.plot(x, y, label = a)
    # plt.ylim(ymin=0)
    plt.legend(bbox_to_anchor=(1.05, 1))
    plt.title('Accuracy confidence score')
    plt.savefig(path_saveMetric + 'accClasifFilter.eps', format = 'eps', bbox_inches="tight")
    plt.close('all')
    print('accuracy threshold clasificator save')
    
def plot_accModel(acc_model,algorithms, path_saveMetric, l):
    print('Creating plot...')
    plt.close('all')
    plt.figure()
    x_aux = None
    for a in algorithms:
        for m in acc_model[a].keys():   
            x, y = [],[]
            all_accModel = acc_model[a][m]
            n_samples = int((len(all_accModel) / t_sample)) + 1 
            if a != 'Fixed':
                for i in range(n_samples):
                    acc_cum = all_accModel[:((i*t_sample) + 1)]
                    acc = acc_cum.sum()/len(acc_cum)
                    
                    # x.append(i*t_sample)
                    x.append(100*(i*t_sample)/len(all_accModel))
                    y.append(100*acc)        
                    x_aux = x
            else: 
                x = x_aux
                y = 100*np.ones(len(x))
            plt.plot(x, y, label = a)
    # plt.ylim(ymin=35)
    # plt.legend(bbox_to_anchor=(1.05, 1))
    # plt.title('Gallery Accuracy')
    plot.tick_params(axis='x', labelsize=14)
    plot.tick_params(axis='y', labelsize=14)
    plt.ylabel('Gallery Precision (%)')
    plt.xlabel('Data Analyzed Ratio (%)')
    plt.savefig(path_saveMetric + 'accGallery.eps', format = 'eps', bbox_inches="tight")
    plt.close('all')
    print('Accuracy model save')
    
def plot_modelEvolution(model_evol, algorithms, path_saveMetric):
    print('Creating plot...')
    plt.close('all')
    plt.figure()
    for a in algorithms:
        for m in model_evol[a].keys():    
            x, y = [],[]
            all_evolModel = model_evol[a][m]
            num_updates = all_evolModel.sum()
            n_samples = int((len(all_evolModel) / t_sample)) + 1 
            for i in range(n_samples):
                n_cum = all_evolModel[:((i*t_sample) + 1)]
                n = n_cum.sum()
                
                # x.append(i*t_sample)
                x.append(100*(i*t_sample)/len(all_evolModel))
                y.append(n)
       
            plt.plot(x, y, label = a)
    plt.legend(bbox_to_anchor=(1.05, 1))
    plt.title('Updates Model')
    plt.savefig(path_saveMetric + 'evolModelNumber.eps', format = 'eps', bbox_inches="tight")
    plt.close('all')
    print('Evol model save')
        
def plot_noise(groundTruth, path_saveMetric):
    y = []
    x = []
    for i,j in enumerate(groundTruth):
        if j == 0:
            y.append(1)
        else:
            y.append(0)
        x.append(100*(i/len(groundTruth)))

    plt.plot(x, y, 'r,')
    plt.legend(bbox_to_anchor=(1.05, 1))
    plt.title('Noise Distribution')
    plt.savefig(path_saveMetric + 'NoiseDistr.eps', format = 'eps', bbox_inches="tight")
    plt.close('all')
    print('Noise save')
    
def plot_indexDelete(index_delete, algorithms, path_saveMetric):
    for a in algorithms:
        for m in index_delete[a].keys():   
            x, y = [],[]
            all_indDelete = index_delete[a][m]
            for i,j in enumerate(all_indDelete):
                y.append(j)
                x.append(100*(i/len(all_indDelete)))

            plt.plot(x, y, ',', label = a)
    plt.legend(bbox_to_anchor=(1.05, 1))
    plt.title('IndexDelete')
    plt.savefig(path_saveMetric + 'IndexDelete.eps', format = 'eps', bbox_inches="tight")
    plt.close('all')
    print('Noise save')
# =============================================================================
#         
# def plot_noise(noiseF1,noiseF2,algorithms, path_saveMetric):
#     print('Creating plot...') 
#     t_sample = 1
#     plt.close('all')
#     plt.figure()
#     for a in algorithms:
#         print(a)
#         for m in noiseF1[a].keys(): 
#             print(m)
#             x, y = [],[]
#             all_nf1 = noiseF1[a][m]
#             # print('len noise en F1', len(all_nf1))
#             n_samples = int((len(all_nf1) / t_sample)) + 1 
#             for i in range(n_samples):
#                 nf1_cum = all_nf1[:((i*t_sample) + 1)]
#                 nf1 = nf1_cum.sum()/len(nf1_cum)
#                 
#                 # x.append(i*t_sample)
#                 x.append(100*(i*t_sample)/len(all_nf1))
#                 y.append(nf1)
#            
#             plt.plot(x, y, label = a + '_' + m)
#     plt.ylim(ymin=0)
#     plt.legend(bbox_to_anchor=(1.05, 1))
#     plt.title('Noise Detected Clasificator')
#     plt.savefig(path_saveMetric + 'noiseF1.jpg', bbox_inches="tight")
#     plt.close('all')
#     print('noiseF1 sav')
#     
#     plt.close('all')
#     plt.figure()
#     for a in algorithms:
#         print(a)
#         for m in noiseF2[a].keys():     
#             print(m)
#             x, y = [],[]
#             all_nf2 = noiseF2[a][m]
#             # print('len noise en F2', len(all_nf2))
#             n_samples = int((len(all_nf2) / t_sample)) + 1 
# 
#             for i in range(n_samples):
#                 if i == n_samples-1:
#                     nf2_cum = all_nf2
#                     x.append(100*len(nf2_cum)/len(all_nf2))
#                 else:
#                     nf2_cum = all_nf2[:((i*t_sample) + 1)]
#                     x.append(100*(i*t_sample)/len(all_nf2))
#                 nf2 = nf2_cum.sum()/len(nf2_cum)
#                 
#                 # x.append(i*t_sample)
#                 y.append(nf2)
#             plt.plot(x, y, label = a + '_' + m)
#     plt.ylim(ymin=0)
#     plt.legend(bbox_to_anchor=(1.05, 1))
#     plt.title('Noise Detected Entropy')
#     plt.savefig(path_saveMetric + 'noiseF2.jpg', bbox_inches="tight")
#     plt.close('all')
#     print('noiseF2 sav')
# =============================================================================
