import numpy as np
import cv2
from Utils import get_wCentroid, get_DiversityModel
from scipy.spatial import distance


class Gallery:
    next_id = 0

    def __init__(self, camera = None):
        self.models = []
        self.candidatesTOmodel = []
        self.camera = camera
        self.N = 0

        self.unknown_samples = []
        self.batch_unknown_samples = 0        
        self.size_to_cluster = 100 
        self.clustering = False
        self.flag_sample = None
    
    def unsupervisedInitialization(self, models):
        num_models = len(models)
        for model in models:
            assert model.identity is not None
            model.centroid = get_wCentroid(model.samples)
            
            model.diversity = get_DiversityModel(model.samples)
            model.ref_updateModels = (-1)*np.ones(num_models)
            model.cluster_distances = np.zeros((len(model.samples), num_models))
        self.models = models
        models_not_initialized = [m for m in self.candidatesTOmodel if m not in models]
        
        samples_unused = []
        for m in models_not_initialized:
            samples_unused = samples_unused + m.samples

        self.unknown_samples = samples_unused
        
    def get_next_id(self):
        identity = self.next_id
        self.next_id = self.next_id + 1
        return identity
    
    def set_models(self, app_models):
        self.models = app_models
    
    def add_unknown_sample(self, sample):
        self.unknown_samples.append(sample)
        self.batch_unknown_samples += 1      
   
    def add_unknown_sampleV2(self, sample):
        if self.flag_sample is not None:
            self.unknown_samples.append(self.flag_sample)
            
        self.unknown_samples.append(sample)
        self.batch_unknown_samples += 1      
        if self.batch_unknown_samples >= self.size_to_cluster:
            self.clustering = self.anomaly_detection()
        else:
            self.clustering = False
            self.flag_sample = None
             
    def add_new_model(self, new_model, config):
        new_model.identity = self.get_next_id()
        unknown_s = self.unknown_samples 
        
        samples = new_model.samples
        update_unknown_s = [u for u in unknown_s if u not in samples]
        self.unknown_samples = update_unknown_s

        if config.use_of_distmat:
            for model in self.models:
                model.ref_updateModels = np.append(model.ref_updateModels, -1)
                new_dist_model = np.zeros(len(model.samples))
                new_dist_model.shape = (new_dist_model.shape[0], 1)
                model.cluster_distances = np.append(model.cluster_distances,new_dist_model , axis = 1)
                assert model.cluster_distances.shape[1] == len(self.models)  + 1
        self.models.append(new_model)

    def add_new_labeled_model(self, model):
        for s in model.samples:
            s.labeled = 1
        model.identity = self.get_next_id()
        self.models.append(model)    
    
    def anomaly_detection(self):
        last_data = self.unknown_samples[-2]
        new_data = self.unknown_samples[-1]
        
        d = distance.cosine(last_data.feat, new_data.feat)
        
        if d < 0.1:
            clustering = False
            self.flag_sample = None
        else:
            clustering = True            
            self.flag_sample = new_data
            self.unknown_samples.pop(-1)

        return clustering
    

class AppearanceModel:
    def __init__(self, gt):
        self.label = gt
        self.identity = None
        self.samples = []
        self.centroid_w = None
        self.centroid = None
        self.class_mean = None
        self.diversity = None
        # Update data
        self.diameter = None
        self.cluster_distances = None
        self.ref_updateModels = []
        self.P = None
        # Evaluation data
        self.n_newModel = 0
        self.last_modelUpate = None

    def get_samples_gt(self):
        gt = [int(s.gt) for s in self.samples]
        return gt


class Sample:
    def __init__(self, feat, path, camera, gt, config):
        self.dir_skeleton = config.dir_skeletons
        self.dir_images = config.dir_images

        self.feat = feat
        self.path = path
        self.camera = camera
        self.gt = gt
        self.H = 0
        self.score = 0     
        self.labeled = 0
        self.P = []
        self.distmat = []
        self.key_points, self.perct_key_points = self.get_key_points()
        self.ratio = self.get_ratio()


    def get_key_points(self):
        N = 18 
        folder, name_sklt = self.get_name()
        people = np.load(self.dir_skeleton + folder + '/' + name_sklt + '.npy', allow_pickle = True)

        max_joints = 0
        person_selected = None
        for person in people:
            num_joints = len(person)
            if num_joints > max_joints:
                max_joints = num_joints
                person_selected = person
                
        if person_selected is not None:
            joints = person_selected
            percentage_joints = round(len(joints)/N, 2)
        else:
            joints = []
            percentage_joints = 0
        return joints, percentage_joints
    
    def get_ratio(self):
        if len(self.key_points) > 0:
            xmin = np.min(self.key_points[:,1])
            xmax = np.max(self.key_points[:,1])
            ymin = np.min(self.key_points[:,2])
            ymax = np.max(self.key_points[:,2])
            
            height = ymax - ymin
            width = xmax - xmin
        
            if width > 0:
                ratio = round((height/width),1)
            else:
                ratio = 0
        else:
            ratio = 0
        return ratio 
    
    def load_image(self):
        img_path = self.get_img_path()
        img = cv2.imread(img_path)
        return img
    
    def get_img_path(self):
        folder, name_image = self.get_name()
        img_path = self.dir_images + folder + '/' + name_image + '.jpg'
        
        return img_path
        
    def get_name(self):
        if 'Mars' in self.dir_images:
            nameFile = self.path.split('/')[-1]
            nameFile, _ = nameFile.split('.')
            folder = nameFile[0:4]
        elif 'Duke' in self.dir_images:
            nameFile = self.path.split('/')[-1]
            nameFile, _ = nameFile.split('.')
            
            folder1, folder2 = self.path.split('/')[-3:-1]
            folder = folder1 + '/' + folder2
        else:
            raise RuntimeError("Dataset not defined")
        return folder, nameFile
    

            