import json 
import os
import pathlib
from DataInformation import parse_data_for_save, parse_unknown_for_save


class Evaluator:
    def __init__(self, config):
        # PREDICTION
        # Phase 1 classification stage
        self.pred_clasif_kept = []
        self.gt_clasif_kept = []
        self.id_clasif_kept = []
        # Phase 2 new classes merging process
        self.pred_merge = []
        self.gt_merge = []
        self.id_merge = []
        # Phase 3 classes initialization
        self.pred_init = []
        self.gt_init = []
        self.id_init = []

        # Gallery metrics
        self.acc_Gallery = []
        self.id_gallery = []
        self.cluster_labels = []

        # Define and create evaluation directories
        self.dir_save_evaluation = (str(pathlib.Path().absolute()) + '/evaluation/' + config.name_test +
                                    '/' + str(pathlib.Path(config.iteration).stem))

        self.dir_save_metric_gallery   = self.dir_save_evaluation + '/metrics_gallery'
        self.dir_save_final_models     = self.dir_save_evaluation + '/final_appearance_models'
        self.dir_save_models_evolution = self.dir_save_evaluation + '/models_evolution'

        self.check_folder(self.dir_save_metric_gallery)
        self.check_folder(self.dir_save_final_models)
        self.check_folder(self.dir_save_models_evolution)

        self.save_config(config)

    @staticmethod
    def check_folder(path_to_check):
        # Create folder if not exists
        if not os.path.exists(path_to_check):
            os.makedirs(path_to_check)

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
                
    def eval_new_model(self, model, final_samples):
        for s in final_samples:
            # Gallery accuracy
            if s.gt == model.label:
                self.acc_Gallery.append(1)
                self.id_gallery.append(int(model.identity))
            elif s.gt != model.label:
                self.acc_Gallery.append(0)
                self.id_gallery.append(int(model.identity))
            
    def saveEvaluation(self, index_tacklet):
        info_precision = {'GalleryPrecision': self.acc_Gallery, 
                          'id_gallery': self.id_gallery,
                          'pred_clasif': self.pred_clasif_kept,
                          'gt_clasif': self.gt_clasif_kept,
                          'id_clasif': self.id_clasif_kept,
                          'pred_merge':self.pred_merge,
                          'gt_merge': self.gt_merge,
                          'id_merge': self.id_merge,
                          'pred_init': self.pred_init,
                          'gt_init': self.gt_init,
                          'id_init': self.id_init,
                          'cluster_labels': self.cluster_labels
                        }

        name = str(index_tacklet)
        name = name.zfill(6)

        # Save metrics
        with open(self.dir_save_metric_gallery + '/' + name + '_metrics.json', 'w') as file:
            json.dump(info_precision,file)
            
    def saveEvolModels(self, AppModels, sample_index, config):
        """
        Save the appearance model evolution of the selected identities
        save a .json file including the path of the images that define the appearance model at sample_index (= name_file)
        the name of the folder is:
            model_label + '_' + model_id where model_label = ground truth assigned, and model_id = identity assigned in the gallery
        """
        idsTOsave = config.idsTOsave
        
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

                name_folder = model_label + '_' + model_id
                name_file = str(sample_index)
                name_file = name_file.zfill(8)

                if not os.path.exists(self.dir_save_models_evolution + '/' + name_folder):
                    os.makedirs(self.dir_save_models_evolution + '/' + name_folder)

                with open(self.dir_save_models_evolution + '/' + name_folder + '/' + name_file + '.json', 'w') as outfile:
                    json.dump(samples_path, outfile)        
                
                idx = identities.index(m.identity)
                assert AppModels[idx].identity == m.identity                

                AppModels[idx] = m

        return AppModels
    
    def save_models(self, AppModels):
        for model in AppModels:
            model_id = str(model.identity)
            name_file = model_id.zfill(4)
              
            data = parse_data_for_save(model)

            with open(self.dir_save_final_models + '/' + name_file + '.json', 'w') as outfile:
                json.dump(data, outfile)
    
    def save_unknown_samples(self, gallery):
        data = parse_unknown_for_save(gallery.unknown_samples)

        with open(self.dir_save_final_models + '/final_unknown_samples.json', 'w') as outfile:
            json.dump(data, outfile)

    def save_config(self, config):
        with open(self.dir_save_evaluation + '/config.json', 'w') as outfile:
            json.dump(config.setup, outfile,  indent = 6)

