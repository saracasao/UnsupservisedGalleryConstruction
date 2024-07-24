import glob
import re
import os.path as osp
import warnings

from scipy.io import loadmat


def DataLoader(name, step = None, path_saveFeat = None):
    if name == 'Market1501':
        data_dir = '/home/scasao/Documents/0_DATASET/REID/People/Market-1501-v15.09.15'
        query_dir = osp.join(data_dir, 'query')
        gallery_dir = osp.join(data_dir, 'bounding_box_test')
        
        query = dirMarket1501(query_dir)
        gallery = dirMarket1501(gallery_dir)
        
    elif name == 'DukeMTMC':
        data_dir = '/home/scasao/Documents/0_DATASET/REID/People/DukeMTMC-reID/DukeMTMC-reID'
        query_dir = osp.join(data_dir, 'query')
        gallery_dir = osp.join(data_dir, 'bounding_box_test')
        
        query = dirDuke(query_dir)
        gallery = dirDuke(gallery_dir)
        
    elif name == 'Mars':
        data_dir = '/home/scasao/Documents/0_DATASET/REID/People/Mars'
        if step == 'extractFeat':
            query, gallery = dirMars(data_dir)
        elif step == 'individualMetrics':
            query, gallery, total_imgs = dirMarsFeat(data_dir, path_saveFeat)
    
    elif name == 'DukeMTMC-Video':
        gallery, gallery_size = dirDukeFeat('gallery', path_saveFeat)
        query, query_size = dirDukeFeat('query', path_saveFeat)
        total_imgs = gallery_size
    else:
        raise RuntimeError("Dataset not defined")
    return query, gallery, total_imgs


def dirMarket1501(dir_path):
    img_paths = glob.glob(osp.join(dir_path, '*.jpg'))
    pattern = re.compile(r'([-\d]+)_c(\d)')

    data = []
    for img_path in img_paths:
        pid, camid = map(int, pattern.search(img_path).groups())
        if pid == -1:
            continue # junk images are just ignored
        assert 0 <= pid <= 1501 # pid == 0 means background
        assert 1 <= camid <= 6
        camid -= 1 # index starts from 0
        data.append((img_path, pid, camid))
    
    return data


def dirDuke(dir_path):
    img_paths = glob.glob(osp.join(dir_path, '*.jpg'))
    pattern = re.compile(r'([-\d]+)_c(\d)')

    data = []
    for img_path in img_paths:
        pid, camid = map(int, pattern.search(img_path).groups())
        assert 1 <= camid <= 8
        camid -= 1 # index starts from 0
        data.append((img_path, pid, camid))

    return data

    
def dirMars(dataset_dir):
    test_name_path = osp.join(dataset_dir, 'test_name.txt')
    track_test_info_path = osp.join(dataset_dir, 'tracks_test_info.mat')
    query_IDX_path = osp.join(dataset_dir, 'query_IDX.mat')
    
    test_names = get_names(test_name_path) #list of gallery names
    track_test = loadmat(track_test_info_path)['track_test_info'] # numpy.ndarray (12180, 4) #[init_track, end_track, id, cam]
    query_IDX = loadmat(query_IDX_path)['query_IDX'].squeeze() # numpy.ndarray (1980,)
    
    query_IDX -= 1 # index from 0
    track_query = track_test[query_IDX, :] #track info 
    gallery_IDX = [i for i in range(track_test.shape[0]) if i not in query_IDX] #remove the query tracks in the gallery
    track_gallery = track_test[gallery_IDX, :] #track info 
    
    query = process_data(test_names, track_query, 'bbox_test', dataset_dir) #query = ((img_tracklet_path, pid, camid), (..))
    gallery = process_data(test_names, track_gallery, 'bbox_test', dataset_dir)# gallery = ((img_tracklet_path, pid, camid), (..))
    return query, gallery


def dirMarsFeat(dataset_dir, dataset_dirFeat):
    test_name_path = osp.join(dataset_dir, 'test_name.txt')
    track_test_info_path = osp.join(dataset_dir, 'tracks_test_info.mat')
    query_IDX_path = osp.join(dataset_dir, 'query_IDX.mat')
    
    test_names = get_names(test_name_path) #list of gallery names
    track_test = loadmat(track_test_info_path)['track_test_info'] # numpy.ndarray (12180, 4) #[init_track, end_track, id, cam]
    query_IDX = loadmat(query_IDX_path)['query_IDX'].squeeze() # numpy.ndarray (1980,)
    
    query_IDX -= 1 # index from 0
    track_query = track_test[query_IDX, :] #track info 
    gallery_IDX = [i for i in range(track_test.shape[0]) if i not in query_IDX] #remove the query tracks in the gallery
    track_gallery = track_test[gallery_IDX, :] #track info 
    
    query, n = process_dataFeat(test_names, track_query, 'query', dataset_dirFeat) #query = ((img_tracklet_path, pid, camid), (..))
    gallery, n = process_dataFeat(test_names, track_gallery, 'gallery', dataset_dirFeat)# gallery = ((img_tracklet_path, pid, camid), (..))
    return query, gallery, n


def get_names(fpath):
    names = []
    with open(fpath, 'r') as f:
        for line in f:
            new_line = line.rstrip()
            names.append(new_line)
    return names


def process_data(names, meta_data, home_dir, dataset_dir):
    assert home_dir in ['bbox_train', 'bbox_test']
    min_seq_len = 1
    num_tracklets = meta_data.shape[0]
    n = 0
    tracklets = []
    for tracklet_idx in range(num_tracklets):
        data = meta_data[tracklet_idx, ...]
        start_index, end_index, pid, camid = data
        if pid == -1:
            continue # junk images are just ignored
        assert 1 <= camid <= 6
        camid -= 1 # index starts from 0
        img_names = names[start_index - 1:end_index]
        n = n + len(img_names)
        # make sure image names correspond to the same person
        pnames = [img_name[:4] for img_name in img_names]
        assert len(set(pnames)) == 1, 'Error: a single tracklet contains different person images'

        # make sure all images are captured under the same camera
        camnames = [img_name[5] for img_name in img_names]
        assert len(set(camnames)) == 1, 'Error: images are captured under different cameras!'

        # append image names with directory information
        img_paths = [
            osp.join(dataset_dir, home_dir, img_name[:4], img_name)
            for img_name in img_names
        ]

        if len(img_paths) >= min_seq_len:
            img_paths = tuple(img_paths)
            tracklets.append((img_paths, pid, camid))
    return tracklets


def process_dataFeat(names, meta_data, home_dir, dataset_dirFeat):
    assert home_dir in ['query', 'gallery'] #check
    min_seq_len = 1
    num_tracklets = meta_data.shape[0]
    n = 0
    tracklets = []
    for tracklet_idx in range(num_tracklets):
        data = meta_data[tracklet_idx, ...]
        start_index, end_index, pid, camid = data
        if pid == -1:
            continue # junk images are just ignored
        assert 1 <= camid <= 6
        camid -= 1 # index starts from 0
        img_names = names[start_index - 1:end_index]
        img_names = [f[:15]+'.npy' for f in img_names]
        
        nimages = [img_name for img_name in img_names if img_name[0:4] != '0000']
        n = n + len(nimages)
        
        # make sure image names correspond to the same person
        pnames = [img_name[:4] for img_name in img_names]
        assert len(set(pnames)) == 1, 'Error: a single tracklet contains different person images'

        # make sure all images are captured under the same camera
        camnames = [img_name[5] for img_name in img_names]
        assert len(set(camnames)) == 1, 'Error: images are captured under different cameras!'

        # append image names with directory information
        img_paths = [
            osp.join(dataset_dirFeat, home_dir, img_name)
            for img_name in img_names
        ]

        if len(img_paths) >= min_seq_len:
            img_paths = tuple(img_paths)
            tracklets.append((img_paths, pid, camid))
    return tracklets, n


def dirDukeFeat(folder, dir_path):
    dir_data = dir_path + folder
    
    min_seq_len = 1    

    print('=> Generating split json file (** this might take a while **)')
    pdirs = glob.glob(osp.join(dir_data, '*')) # avoid .DS_Store
    print(
        'Processing "{}" with {} person identities'.format(
            dir_data, len(pdirs)
        )
    )

    pid_container = set()
    for pdir in pdirs:
        pid = int(osp.basename(pdir))
        pid_container.add(pid)
    pid2label = {pid: label for label, pid in enumerate(pid_container)}
    
    total_imgs = 0
    tracklets = []
    for pdir in pdirs:
        pid = int(osp.basename(pdir))
        tdirs = glob.glob(osp.join(pdir, '*'))
        for tdir in tdirs:
            raw_img_paths = glob.glob(osp.join(tdir, '*.npy'))
            num_imgs = len(raw_img_paths)

            if num_imgs < min_seq_len:
                continue

            img_paths = []
            for img_idx in range(num_imgs):
                # some tracklet starts from 0002 instead of 0001
                img_idx_name = 'F' + str(img_idx + 1).zfill(4)
                res = glob.glob(
                    osp.join(tdir, '*' + img_idx_name + '*.npy')
                )
                if len(res) == 0:
                    warnings.warn(
                        'Index name {} in {} is missing, skip'.format(
                            img_idx_name, tdir
                        )
                    )
                    continue
                img_paths.append(res[0])
            img_name = osp.basename(img_paths[0])
            if img_name.find('_') == -1:
                # old naming format: 0001C6F0099X30823.jpg
                camid = int(img_name[5]) - 1
            else:
                # new naming format: 0001_C6_F0099_X30823.jpg
                camid = int(img_name[6]) - 1
            img_paths = tuple(img_paths)
            total_imgs = total_imgs + len(img_paths)
            tracklets.append((img_paths, pid, camid))
            
    return tracklets, total_imgs
