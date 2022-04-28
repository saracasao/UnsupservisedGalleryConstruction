import cv2
import re
import numpy as np
import matplotlib
import json
import random
import os
import glob
import matplotlib.pyplot as plt
import seaborn as sns
from os import listdir

import time
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from PIL import Image, ImageDraw
from matplotlib.pyplot import imshow

# path_evolution = '/home/scasao/pytorch/1_Clusters/Mars/ModelsEvolution/'

# dir_features = '/home/scasao/Documents/DATASET/Re-Id/Mars_normalize/gallery/'
# dir_images = '/home/scasao/Documents/0_DATASET/REID/People/Mars/bbox_test/'

# path_save_tsne = '/home/scasao/pytorch/1_Clusters/Mars/TSNE_images_video/'
# path_save_tsne_points = '/home/scasao/pytorch/1_Clusters/Mars/TSNE_images_points/'
# path_save_gallery = '/home/scasao/pytorch/1_Clusters/Mars/Complete_Gallery/'

#%%
"""
FROM ALL DATA -> TNSE DISTRIBUTION
"""
# IDS_selected = [2,4,52,68,202,196,34,106,146,94,156,176,110,112,142,308,390,258,18,24,26]

ids = sorted(os.listdir(path_evolution))
path_features = []
for i in ids: 
    name_files = sorted(os.listdir(path_evolution + i))
    for name in name_files:
        with open(path_evolution + i + '/' + name) as f:
            global_paths = json.load(f)
            
        for global_path in global_paths:
            individual_name = global_path.split('/')[-1]
            path_features.append(individual_name)

feat_names = np.unique(path_features) 
print(len(feat_names))
ids_str = []
for i in ids:
    ids = str(i)
    ids = ids.zfill(4)
    ids_str.append(ids)
print(ids_str) 

# feat_names = sorted([f for f in os.listdir(dir_features)])

X = []
ids_all = []  
dir_imgs = []
for name_file in feat_names:
    feat = np.load(dir_features + name_file)[0]
    
    name_image = name_file.split('.')[0] + '.jpg'
    dir_folder = name_file[0:4]
    
    if dir_folder in ids_str:
        dir_imgs.append(dir_images + dir_folder + '/' + name_image)
        
        X.append(feat)
        ids_all.append(int(dir_folder))
X = np.array(X)
print(len(X))
ids_all = np.array(ids_all)   

feat_cols = ['pixel'+str(i) for i in range(X.shape[1])]

df = pd.DataFrame(X,columns=feat_cols)
df['y'] = ids_all
df['label'] = df['y'].apply(lambda i: str(i))
print('Size of the dataframe: {}'.format(df.shape))

pca_50 = PCA(n_components=50)
pca_result_50 = pca_50.fit_transform(df[feat_cols].values)
print('Cumulative explained variation for 50 principal components: {}'.format(np.sum(pca_50.explained_variance_ratio_)))

tsne = TSNE(n_components=2, verbose=0, perplexity=100, n_iter=1000)
tsne_pca_results = tsne.fit_transform(pca_result_50)

#DIBUJAR IMÁGENES
# tx, ty = tsne_pca_results[:,0], tsne_pca_results[:,1]
# tx = (tx-np.min(tx)) / (np.max(tx) - np.min(tx))
# ty = (ty-np.min(ty)) / (np.max(ty) - np.min(ty))

# width = 4000
# height = 3000
# max_dim = 100

# full_image = Image.new('RGBA', (width, height))
# for img, x, y in zip(dir_imgs, tx, ty):
#     tile = Image.open(img)
#     rs = max(1, tile.width/max_dim, tile.height/max_dim)
#     tile = tile.resize((int(tile.width/rs), int(tile.height/rs)), Image.ANTIALIAS)
#     full_image.paste(tile, (int((width-max_dim)*x), int((height-max_dim)*y)), mask=tile.convert('RGBA'))
    
# full_image.show('images')
# full_image.save('images.png')
tsne_path = 'tSNE-points.json'
ids_all = ids_all.tolist()
tsne_pca_results = tsne_pca_results.tolist()

data = {'paths':dir_imgs, 'ids':ids_all, 'pca':tsne_pca_results}
with open(tsne_path, 'w') as outfile:
    json.dump(data, outfile)
    
#%%

"""
TSNE + CLASS MODELS
"""
color_dict_points = dict({2:'red',4:'firebrick',52:'tomato',68:'green',202:'limegreen',196:'lightgreen',34:'blue',106:'darkblue',146:'aquamarine',
                   94:'black',156:'gray',176:'silver',110:'darkgoldenrod',112:'orange',142:'gold',
                   308:'magenta',390:'fuchsia',258:'deeppink',18:'darkviolet',24:'blueviolet',26:'rebeccapurple'})

color_dict = dict({2:'red',4:'firebrick',52:'tomato',68:'green',202:'lawngreen',196:'lightgreen',34:'blue',106:'darkblue',146:'aquamarine',
                   94:'black',156:'gray',176:'silver',110:'yellow',112:'khaki',142:'gold',
                   308:'magenta',390:'pink',258:'deeppink',18:'darkviolet',24:'blueviolet',26:'rebeccapurple'})
# style_dict = dict({2:0, 4:1,52:2,68:0,202:1,196:2,34:0,106:1,146:2, 94:0,156:1,176:2,110:0,112:1,142:2,308:0,390:1,258:2,18:0,24:1,26:2})

# style_sizes = dict({2:1, 4:1,52:1,68:1,202:1,196:1,34:1,106:1,146:1,
#                     94:1,156:1,176:1,110:1,112:1,142:1,308:1,390:1,258:1,18:1,24:1,26:1})

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

def featTOimage(dir_feat):
    name_file = dir_feat.split('/')[-1]
    name_image = name_file.split('.')[0] + '.jpg'
    dir_folder = name_file[0:4]

    dir_img = dir_images + dir_folder + '/' + name_image
    
    return dir_img

def get_currenData(models):
    dir_imgs = []
    g_ids = []
    for i in ids:
        assert len(models[i]) < 31
        for path in models[i]:
            dir_feat = path
            dir_img = featTOimage(dir_feat)
            
            dir_imgs.append(dir_img)
            g_ids.append(i)
    return g_ids, dir_imgs

def tsne_points(tsne_pca_results,all_ids, global_index):
    plt.figure(str(global_index)) 
    
    x = tsne_pca_results[:,0]
    y = tsne_pca_results[:,1]
    
    x_n, y_n, ids_n, marker, sizes = [], [], [], [], []
    for q_id, x_i, y_i in zip(all_ids, x, y): 
        x_n.append(x_i)
        y_n.append((-1)*y_i)
        ids_n.append(q_id)
        marker.append(style_dict[q_id])
        sizes.append(style_sizes[q_id])
    cluster = {'pos_x': x_n, 'pos_y': y_n}
    cluster['y'] = ids_n
    cluster['style'] = marker
    cluster['size'] = sizes

    df = pd.DataFrame(data=cluster)
    g = sns.scatterplot(x='pos_x', y='pos_y',
                    hue='y',
                    palette=color_dict_points,
                    data=df,
                    legend=False,
                    alpha = 0.7)
    # style = 'style'
    name = str(global_index)
    name = name.zfill(8)

    dir_save = path_save_tsne_points + name + '.png'
    plt.savefig(dir_save,dpi=300)
    plt.close('all')

def save_models(current_models, global_index,out_model, height, width, sample):
    ids_toUpdate = current_models.keys()
    for i in ids_toUpdate:  
        total_width = 0
        max_height = 0
        
        images = []
        for path in current_models[i]:
            dir_feat = path
            dir_img = featTOimage(dir_feat)
            
            images.append(cv2.imread(dir_img))
            if images[-1].shape[0] > max_height: 
                max_height = images[-1].shape[0]
            total_width += images[-1].shape[1]
        final_image = 255*np.ones((height,width,3), dtype=np.uint8)
        current_x = 0
        for img in images: 
            final_image[0: max_height, current_x:img.shape[1]+current_x] = img
            current_x += img.shape[1]
        
        name = str(global_index)
        name = name.zfill(4)
        if not os.path.exists(path_save_gallery + i):
            os.makedirs(path_save_gallery + i)                   
        dir_save_img = path_save_gallery + i + '/'  + name + '.jpg'

        # cv2.imwrite(dir_save_img, final_image)

        if global_index%sample == 0:
            new_img = cv2.cvtColor(final_image, cv2.COLOR_BGRA2BGR)
            out_model[i].write(new_img)
            
    return  out_model


#TSNE VIDEO
dir_imagesSaved = sorted(glob.glob('/home/scasao/pytorch/1_Clusters/Mars/TSNE_images_newOrder/*.png'))

dir_img0 = dir_imagesSaved[0]
img0 = cv2.imread(dir_img0)

height_tsne, width_tsne, layers = img0.shape
size = (width_tsne,height_tsne)

out_tsne = cv2.VideoWriter('./videos/tsne_orderF.avi',cv2.VideoWriter_fourcc(*'DIVX'), 30, size)

#MODEL VIDEO
dir_gallery = '/home/scasao/pytorch/1_Clusters/Mars/Complete_Gallery/'
ids = os.listdir(dir_gallery)

out_model = {}
for i in ids:
    dir_imagesSaved = sorted(glob.glob(dir_gallery + i + '/*.jpg' ))
    dir_img0 = dir_imagesSaved[-1]
    img0 = cv2.imread(dir_img0)
    
    height_model, width_model, layers = img0.shape
    size = (width_model,height_model)
    
    out_model[i] = cv2.VideoWriter('./videos/complete_gallery_fast/ID'+ i + '.avi',cv2.VideoWriter_fourcc(*'DIVX'), 30, size)
            
#LOAD DATA PER NEW SAMPLE IN CLASS MODEL
ids = sorted(os.listdir(path_evolution))
print(ids)
global_paths = []
global_index = []
current_models = {}
for i in ids: 
    name_files = sorted(os.listdir(path_evolution + i))
    for name in name_files:
        g_index, l_index = name.split('_')
        if g_index == '000000':
            with open(path_evolution + i + '/' + name) as f:
                file = json.load(f)
            current_models[i] = file
        else:
            with open(path_evolution + i + '/' + name) as f:
                file = json.load(f)
            global_paths.append(file)
            global_index.append(g_index)
sort = np.argsort(global_index)  
print('first save')      
# save_models(current_models, 0, out_model, height_model, width_model)

print('total updates to show', len(sort))
#LOAD TSNE DATA
tsne_path = 'tSNE-points.json'
with open(tsne_path) as f:
    data = json.load(f)
    
dir_imgs = data['paths']
tsne_results = np.array(data['pca'])        
for i, index in enumerate(sort):
    i = i + 1
    if i%200 == 0:
        print(i)
    g_ids, dir_gimgs = get_currenData(current_models)
    assert len(dir_gimgs) < 361
    #SELECT CURRENT MODELS
    tsne_pca = []
    dir_img_pca = []
    gids_pca = []
    for j, dir_gimg in enumerate(dir_gimgs):
        for pca, dir_img in zip(tsne_results, dir_imgs):
            if dir_img == dir_gimg:
                tsne_pca.append(pca)
                dir_img_pca.append(dir_img)
            
                g_id = g_ids[j]
                gids_pca.append(int(g_id))
    tsne_pca = np.array(tsne_pca)
    gids_pca = np.array(gids_pca)
    
    assert len(tsne_pca) == len(gids_pca) == len(dir_img_pca)
    
    # tsne_points(tsne_pca,gids_pca, i)
    
    #DIBUJAR IMÁGENES
    tx, ty = tsne_pca[:,0], tsne_pca[:,1]
    tx = (tx-np.min(tx)) / (np.max(tx) - np.min(tx))
    ty = (ty-np.min(ty)) / (np.max(ty) - np.min(ty))

    width = 700#4000
    height = 600#3000
    max_dim = 30#100
    full_image = Image.new('RGBA', (width, height))
    j = 0
    for img, x, y in zip(dir_img_pca, tx, ty):
        tile = Image.open(img)
        pid = gids_pca[j]
        
        color = color_dict[pid]
        tile = preprocess_img(tile, color)

        rs = max(1, tile.width/max_dim, tile.height/max_dim)
        tile = tile.resize((int(tile.width/rs), int(tile.height/rs)), Image.ANTIALIAS)
        full_image.paste(tile, (int((width-max_dim)*x), int((height-max_dim)*y)), mask=tile.convert('RGBA'))
        j += 1
    name_image = str(i)
    name_image = name_image.zfill(4)
    full_image.save(path_save_tsne + name_image + '.png')
    
    updates = global_paths[index]
    id_updated = updates[0].split('/')[-1][0:4]
    current_models[id_updated] = updates
    # out_model = save_models(current_models, i,out_model, height_model, width_model)
    
    #create video tsne
    # img = cv2.imread(path_save_tsne + name_image + '.png', cv2.IMREAD_UNCHANGED)
    # trans_mask = img[:,:,3] == 0
    # img[trans_mask] = [255, 255, 255, 255]
    # new_img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    # out_tsne.write(new_img)

# out_tsne.release()
# for i in out_model.keys():
#     out_model[i].release()
    
#%%
"""
QUERY + CLASS MODELS
"""
path_evolution = '/home/scasao/pytorch/1_Clusters/0.1_Semi-SupervisedIncrementalGallery/ModelsEvolution/SelectedModelEvolMars/'

def get_queries():
    queries = np.load('/home/scasao/pytorch/1_Clusters/0.1_Semi-SupervisedIncrementalGallery/Iterations/Mars/unsupervised/unsupervised_iter0.npy',  allow_pickle=True)
    queries = queries.tolist()
    queries = [tuple(f) for f in queries]
    
    path_queries = []
    gt_queries = []
    for q in queries:
        path = q[0]
        gt_id = q[1]
        for p in path:
            individual_name = p.split('/')[-1]
            path_queries.append(individual_name)
            gt_queries.append(int(gt_id))
    
    ids = sorted(os.listdir(path_evolution))
    path_features = []
    for i in ids: 
        name_files = sorted(os.listdir(path_evolution + i))
        for name in name_files:            
            with open(path_evolution + i + '/' + name) as f:
                global_paths = json.load(f)
            for global_path in global_paths:
                individual_name = global_path.split('/')[-1]
                individual_name = individual_name.split('.')[0] + '.npy'
                path_features.append(individual_name)
    feat_names = np.unique(path_features) 
    ids_int = [int(i) for i in ids]
    
    n = 0   
    path_qSelected, g_idSelected, intoGallery, idx_queries = [], [], [], []
    for path, gt_id in zip(path_queries,gt_queries):
        if (gt_id in ids_int) or (path in feat_names):
            path_qSelected.append(path)
            g_idSelected.append(gt_id)
            idx_queries.append(n)

            if path in feat_names:
                intoGallery.append(1)
            else:
                intoGallery.append(0)
        n += 1
    return path_qSelected,g_idSelected, intoGallery, idx_queries

def featTOimage(dir_feat):
    name_file = dir_feat.split('/')[-1]
    name_image = name_file.split('.')[0] + '.jpg'
    dir_folder = name_file[0:4]

    dir_img = dir_images + dir_folder + '/' + name_image
    
    return dir_img

def get_currenData(models):
    dir_imgs = []
    g_ids = []
    for i in ids:
        assert len(models[i]) < 31
        for path in models[i]:
            dir_feat = path
            dir_img = featTOimage(dir_feat)
            
            dir_imgs.append(dir_img)
            g_ids.append(i)
    return g_ids, dir_imgs

def hconcat_resize_min(im_list, video_heigth, interpolation=cv2.INTER_CUBIC):
    h_min = video_heigth
    im_list_resize = [cv2.resize(im, (int(im.shape[1] * h_min / im.shape[0]), h_min), interpolation=interpolation)
                      for im in im_list]
    return cv2.hconcat(im_list_resize)

def get_model(dir_images, video_height, video_width): 
    final_image = 255*np.ones((video_height,video_width,3), dtype=np.uint8)
    if dir_images is not None:
        images = []
        for i,dir_img in enumerate(dir_images):
            # if i not in idx_to_delete:
            img = cv2.imread(dir_img)
            images.append(img)
        app_model = hconcat_resize_min(images,video_height)
        final_image[0: app_model.shape[0], 0:app_model.shape[1]] = app_model 
    return final_image

def get_model_ref(dir_model, height_q, width_q):
    total_width = 0
    max_height = 0
        
    images = []
    for dir_img in dir_model:     
        img = cv2.imread(dir_img)
        img = cv2.resize(img, (width_q, height_q), interpolation = cv2.INTER_AREA)
        images.append(img)
        if images[-1].shape[0] > max_height: 
            max_height = images[-1].shape[0]
        total_width += images[-1].shape[1]
    final_image = 255*np.ones((max_height,total_width,3), dtype=np.uint8)
    current_x = 0
    for img in images: 
        final_image[0: max_height, current_x:img.shape[1]+current_x] = img
        current_x += img.shape[1]
    return final_image

def save_modelsV0(current_models, global_index, out_model, height, width, sample):
    ids_toUpdate = current_models.keys()
    for i in ids_toUpdate:  
        total_width = 0
        max_height = 0
        
        images = []
        for path in current_models[i]:
            dir_feat = path
            dir_img = featTOimage(dir_feat)
            
            images.append(cv2.imread(dir_img))
            if images[-1].shape[0] > max_height: 
                max_height = images[-1].shape[0]
            total_width += images[-1].shape[1]
        final_image = 255*np.ones((height,width,3), dtype=np.uint8)
        current_x = 0
        for img in images: 
            final_image[0: max_height, current_x:img.shape[1]+current_x] = img
            current_x += img.shape[1]
        
        name = str(global_index)
        name = name.zfill(4)
        if not os.path.exists(path_save_gallery + i):
            os.makedirs(path_save_gallery + i)                   
        dir_save_img = path_save_gallery + i + '/'  + name + '.jpg'

        # cv2.imwrite(dir_save_img, final_image)
        if global_index%sample==0:
            new_img = cv2.cvtColor(final_image, cv2.COLOR_BGRA2BGR)
            out_model[i].write(new_img)
    return  out_model

def get_imgQuery(q_img, intoGallery, tick, cross):
    gap = 5
    width = q_img.shape[1] + tick.shape[1] + gap
    height = q_img.shape[0]
    final_image = 255*np.ones((height,width,3), dtype=np.uint8)

    final_image[0: q_img.shape[0], 0:q_img.shape[1]] = q_img
    
    if intoGallery == 1:
        final_image[int(q_img.shape[0]/3):int(q_img.shape[0]/3) + tick.shape[0], gap + q_img.shape[1]: gap + q_img.shape[1] + tick.shape[1]] = tick 
    else:
        final_image[int(q_img.shape[0]/3):int(q_img.shape[0]/3) + cross.shape[0], gap + q_img.shape[1]: gap + q_img.shape[1] + cross.shape[1]] = cross

    return final_image

path_queries, q_id, intoGallery, idx_queries = get_queries()

#QUERIES VIDEO
dir_img = featTOimage(path_queries[0])
img_q0 = cv2.imread(dir_img)
height_q, width_q, layers = img_q0.shape


#Load tick 
tick = cv2.imread('./videos/tick.png')
#Load corss
cross = cv2.imread('./videos/cross.png')
new_size = int(height_q/5)
size = (new_size,new_size)
tick = cv2.resize(tick, size,interpolation = cv2.INTER_AREA)
cross = cv2.resize(cross, size,interpolation = cv2.INTER_AREA)

size_q = (width_q+new_size +5 ,height_q)
out_query = cv2.VideoWriter('./videos/test/query_fastSample.avi',cv2.VideoWriter_fourcc(*'DIVX'), 37, size_q)


# MODEL VIDEO
dir_gallery = path_evolution #'/home/scasao/pytorch/1_Clusters/Mars/Complete_Gallery/'
ids = sorted(os.listdir(path_evolution))

out_model = {}
for i in ids:
    print('id', i)
    dir_imagesSaved = sorted(glob.glob(dir_gallery + i + '/*.json' ))
    path_file = dir_imagesSaved[-1]
    with open(path_file) as f:
        dir_model = json.load(f)
    name_file = path_file.split('/')[-1]
    global_index, _ = name_file.split('.')

    img0 = get_model_ref(dir_model, height_q, width_q)
    height_model, width_model, layers = img0.shape
    size_model = (width_model,height_model)
    
    out_model[i] = cv2.VideoWriter('./videos/test/ID'+ i + '.avi',cv2.VideoWriter_fourcc(*'DIVX'), 37, size_model)
            
#LOAD DATA PER NEW SAMPLE IN CLASS MODEL
ids = sorted(os.listdir(path_evolution))
global_paths = []
global_index = []
current_models = {}
for i in ids: 
    name_files = sorted(os.listdir(path_evolution + i))
    for name in name_files:
        g_index, l_index = name.split('.')
        if g_index == '000000':
            with open(path_evolution + i + '/' + name) as f:
                file = json.load(f)
            current_models[i] = file
        else:
            with open(path_evolution + i + '/' + name) as f:
                file = json.load(f)
            global_paths.append(file)
            global_index.append(g_index)
sort = np.argsort(global_index)  
sample = 4 

for identity in ids:
    print(identity)
    name_files_models = sorted(os.listdir(path_model_evol + identity))
    for name_file in name_files_models:
        with open(path_model_evol + identity + '/' + name_file) as f:
            dir_model = json.load(f)

        global_index, _ = name_file.split('.')
        save_models(dir_model, identity, global_index, height, width)

save_models(current_models, 0, out_model, height_model, width_model, sample)
n = 0
print('total updates to show', len(path_queries))
for i, q_path in enumerate(path_queries):
    if i%500 == 0:
        print(i)
    dir_q = featTOimage(q_path)
    img_q = cv2.imread(dir_q)
    img_q = cv2.resize(img_q, (img_q0.shape[1],img_q0.shape[0]),interpolation = cv2.INTER_AREA)
    
    img_q = get_imgQuery(img_q, intoGallery[i], tick, cross)
    if i%sample==0:
        out_query.write(img_q)
    
    #Get individual models 
    index = sort[n]
    updates = global_paths[index]
    id_updated = updates[0].split('/')[-1][0:4]
    current_models[id_updated] = updates
   
    out_model = save_models(current_models, i, out_model, height_model, width_model, sample)
    if intoGallery[i] == 1:
        n += 1

        
out_query.release()
for i in out_model.keys():
    out_model[i].release()
    
#%%

"VISUALIZATION OF MODEL EVOLUTION"

import os
import json
import cv2
import numpy as np

path_save_imgs = '/home/scasao/pytorch/1_Clusters/0.1_Semi-SupervisedIncrementalGallery/ModelsEvolution/EvolMarsImages/'
path_model_evol = '/home/scasao/pytorch/1_Clusters/0.1_Semi-SupervisedIncrementalGallery/ModelsEvolution/SelectedModelEvolMars/'

def save_models(current_models, identity, global_index, height, width): 
    total_width = 0
    max_height = 0
    
    images = []
    for dir_img in current_models:   
        img = cv2.imread(dir_img)
        img = cv2.resize(img, (width, height), interpolation = cv2.INTER_AREA)
        images.append(img)
        if images[-1].shape[0] > max_height: 
            max_height = images[-1].shape[0]
        total_width += images[-1].shape[1]
    final_image = 255*np.ones((height,total_width,3), dtype=np.uint8)
    current_x = 0
    for img in images:
        final_image[0: max_height, current_x:img.shape[1]+current_x] = img
        current_x += img.shape[1]
    
    name = str(global_index)
    name = name.zfill(4)
    if not os.path.exists(path_save_imgs + identity):
        os.makedirs(path_save_imgs + identity)                   
    dir_save_img = path_save_imgs + identity + '/'  + name + '.jpg'
    
    cv2.imwrite(dir_save_img, final_image)

height = 256
width = 128

ids = sorted(os.listdir(path_model_evol))


for identity in ids:
    print(identity)
    name_files_models = sorted(os.listdir(path_model_evol + identity))
    for name_file in name_files_models:
        with open(path_model_evol + identity + '/' + name_file) as f:
            dir_model = json.load(f)

        global_index, _ = name_file.split('.')
        save_models(dir_model, identity, global_index, height, width)


#%%

"VIDEO OF GALLERY EVOLUTION SELECTED IDENTITIES"
idx_to_delete = [2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32,34,38,40,42]

def hconcat_resize_min(im_list, video_heigth, img_width, interpolation=cv2.INTER_CUBIC):
    h_min = video_heigth
    im_list_resize = [cv2.resize(im, (img_width, h_min), interpolation=interpolation)
                      for im in im_list]
    return cv2.hconcat(im_list_resize)

def get_model(dir_images, video_height, video_width, img_width): 
    final_image = 255*np.ones((video_height,video_width,3), dtype=np.uint8)
    if dir_images is not None:
        images = []
        for i,dir_img in enumerate(dir_images):
            if i not in idx_to_delete:
                img = cv2.imread(dir_img)
                images.append(img)
        app_model = hconcat_resize_min(images,video_height, img_width)
        final_image[0: app_model.shape[0], 0:app_model.shape[1]] = app_model 
    return final_image

def get_model_ref(dir_model, height_q, width_q):
    total_width = 0
    max_height = 0
        
    images = []
    for i, dir_img in enumerate(dir_model):
        if i not in idx_to_delete:
            img = cv2.imread(dir_img)
            img = cv2.resize(img, (width_q, height_q), interpolation = cv2.INTER_AREA)
            images.append(img)
            if images[-1].shape[0] > max_height: 
                max_height = images[-1].shape[0]
            total_width += images[-1].shape[1]
    final_image = 255*np.ones((max_height,total_width,3), dtype=np.uint8)
    current_x = 0
    for img in images: 
        final_image[0: max_height, current_x:img.shape[1]+current_x] = img
        current_x += img.shape[1]
    return final_image


dir_models = '/home/scasao/pytorch/1_Clusters/0.1_Semi-SupervisedIncrementalGallery/ModelsEvolution/SelectedModelEvolDuke/'

total_imgs = 30
height_img, width_img = 256, 128
ids = sorted(os.listdir(dir_models))

#GET SIZE REF
dir_imagesSaved = sorted(glob.glob(dir_models + ids[0] + '/*.json' ))
path_file = dir_imagesSaved[-1]
with open(path_file) as f:
    dir_model = json.load(f)
name_file = path_file.split('/')[-1]
global_index, _ = name_file.split('.')

img0 = get_model_ref(dir_model, height_img, width_img)
height_model, width_model, layers = img0.shape
size_model = (width_model,height_model)

widht_img_model = int(width_model/total_imgs)


#INIT VIDEOS MODEL
out_model, app_model_size = {},{}
for identity in ids:
    out_model[identity] = cv2.VideoWriter('./videos/Duke_30patch/ID'+ identity + '.avi',cv2.VideoWriter_fourcc(*'DIVX'), 30, size_model)

#RANGE OF DATA
global_index = []
model_per_id = {identity: [] for identity in ids}
for identity in ids: 
    name_files = sorted(os.listdir(dir_models + identity))
    name_files = [n.split('.')[0] for n in name_files]
    name_files.sort(key = int)
    
    model_per_id[identity] = name_files
    global_index = global_index + name_files
global_index = list(set(global_index))
global_index = np.sort(global_index)  
print('total frames with info', len(global_index))


#DEFINING MODEL PER EACH FRAME -> SYNCHRONIZATION OF VIDEOS
model_per_frame = {identity: [] for identity in ids}
for identity in ids:
    app_model = None
    for frame_id in global_index:
        if frame_id in model_per_id[identity]:
            app_model = frame_id
        model_per_frame[identity].append(app_model)
    assert len(model_per_frame[identity]) == len(global_index), 'Error in lenght, model len {}, total data len {}'.format(len(model_per_frame[identity]),len(global_index))

for i, frame_id in enumerate(global_index):
    if i%100 == 0:
        print(i)
    #Get individual models 
    for identity in ids:
        model_in_frame = model_per_frame[identity][i]
        if model_in_frame is not None:
            dir_file = dir_models + identity + '/' + model_in_frame + '.json'
            with open(dir_file) as f:
                path_model = json.load(f)
            final_image = get_model(path_model, height_model, width_model, widht_img_model)
            out_model[identity].write(final_image)
        else:
            final_image = get_model(None, height_model, width_model, widht_img_model)
            out_model[identity].write(final_image)

for i in out_model.keys():
    out_model[i].release()


