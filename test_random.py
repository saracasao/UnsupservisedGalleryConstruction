from ComponentsGallery import Gallery

cameras = []
for i in range(5):
    g = Gallery()
    cameras.append(g)
    
g0 = cameras[0]
g0.next_id = 25

for c in cameras:
    print(c.next_id)
#%%

run = True

while run:
    print('inside while')
    a = list(range(20))
    for i in a:
        print(i)
        print('inside for')
        input('')
        if (i+1)%5 == 0:
            print(i)
            run = False
            break

#%%
import os
import glob
import shutil

dir_gallery_origin = '/home/scasao/Documents/0_DATASET/REID/People/DukeMTMC-VideoReID/gallery_sklt/' 
dir_gallery_new = '/home/scasao/Documents/0_DATASET/REID/People/DukeMTMC-VideoReID-Feat/gallery_sklt/' 
dir_query = '/home/scasao/Documents/0_DATASET/REID/People/DukeMTMC-VideoReID-Feat/query/' 

dir_query_ids = glob.glob(os.path.join(dir_query, '*'))

query_data = []
for d_q in dir_query_ids:
    dir_query_tracks =  glob.glob(os.path.join(d_q, '*'))
    for d_t in dir_query_tracks:
        q_data = d_t.split('/')[-2::]
        q_data = q_data[0] + '/' + q_data[1] 
        query_data.append(q_data)
        
# global_path = '/home/scasao/Documents/0_DATASET/REID/People/DukeMTMC-VideoReID/gallery/'
num_tracklets = 0
identities = sorted([f for f in os.listdir(dir_gallery_origin)])
for identity in identities: 
    identity_path = dir_gallery_origin + identity
    # tracklets = glob.glob(os.path.join(identity_path, '*'))
    tracklets = sorted([f for f in os.listdir(identity_path)])
    for track in tracklets: 
        t_data = identity + '/' + track
        if t_data not in query_data:
            img_path = identity_path + '/' + track
            descriptors = glob.glob(os.path.join(img_path, '*'))
            
            path_save = dir_gallery_new + identity + '/' + track
            if not os.path.exists(path_save):
                os.makedirs(path_save)
            
            for d in descriptors: 
                shutil.copy(d, path_save)
                
#%%
import os
import glob
import shutil

dir_gallery = '/home/scasao/Documents/0_DATASET/REID/People/DukeMTMC-VideoReID-Feat/gallery_sklt/' 
dir_query = '/home/scasao/Documents/0_DATASET/REID/People/DukeMTMC-VideoReID-Feat/query/' 

num_query_tracklets, num_query_img = 0,0
dir_query_ids = glob.glob(os.path.join(dir_query, '*'))

query_data = []
for d_q in dir_query_ids:
    dir_query_tracks =  glob.glob(os.path.join(d_q, '*'))
    num_query_tracklets = num_query_tracklets + len(dir_query_tracks)
    for d_t in dir_query_tracks:
        descriptors = glob.glob(os.path.join(d_t, '*'))
        num_query_img = num_query_img + len(descriptors)
        

num_gal_tracklets, num_gal_img = 0,0
identities = sorted([f for f in os.listdir(dir_gallery)])
for identity in identities: 
    identity_path = dir_gallery + identity
    tracklets = sorted([f for f in os.listdir(identity_path)])
    num_gal_tracklets = num_gal_tracklets + len(tracklets)
    for track in tracklets: 
        t_data = identity + '/' + track
        img_path = identity_path + '/' + track
        descriptors = glob.glob(os.path.join(img_path, '*'))
        num_gal_img = num_gal_img + len(descriptors)


    