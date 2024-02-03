import os
import torch
import torch.nn as nn
import numpy as np
import torchvision
from torchvision.models import feature_extraction
import torch.nn.functional as F
from tqdm import tqdm
from PIL import Image

class FeatureExtractor():
    def __init__(self, model_name, weights = None, num_of_classes = None):
        self.model = None
        self.feature_list = []
        self.device = "cpu"
        self.last_layer_name = None
        models = torchvision.models.list_models()

        assert model_name in models, f"model_name must be in {models}"
        self.model_name = model_name


        self.model = torchvision.models.get_model(model_name, weights = "DEFAULT")
        graph_node_names = list(feature_extraction.get_graph_node_names(self.model))

        if weights is not None and num_of_classes is None:
            raise ValueError("num_of_classes must be exist when weights is not None")
        if weights is None and num_of_classes is not None:
            raise ValueError("weights must be exist when num_of_classes is not None")
        
        if num_of_classes is not None:
            print("graph_node_names[-1][-1]", graph_node_names[-1][-1])
            if graph_node_names[-1][-1] == "heads.head":
                
                in_features = getattr(self.model, "heads")[-1].in_features
                self.model.heads[-1] = nn.Linear(in_features, num_of_classes)
                print(self.model)
            else:
                in_features = getattr(self.model, graph_node_names[-1][-1]).in_features
                setattr(self.model, graph_node_names[-1][-1], nn.Linear(in_features, num_of_classes))
            
        if weights is not None:
            weights = torch.load(weights)["state_dict"]
            self.model.load_state_dict(weights)
        
        # create feature extractor
            
        self.last_layer_name = graph_node_names[-1][-2]
        self.model = feature_extraction.create_feature_extractor(self.model, return_nodes = [graph_node_names[-1][-2]])

    def to(self, device):
        self.device = device
        if device == "cuda":
            self.model.to(self.device)
        
    def _preprocess(self, image):
        pass
    
    def _postprocess(self, output):
        return output
    
    def extract_features(self, dataset):
        for i, data in enumerate(tqdm(dataset, desc="Extracting features", unit="images")):
            preprocessed_data = self._preprocess(data["image"])
            self.model.eval()
            with torch.no_grad():
                output = self.model(preprocessed_data)
            
            output = self._postprocess(output)  
            if self.device == "cuda":
                output = output.cpu().numpy()
            output_dict = {
                "image_id": i,
                "label_id" : data["label_id"],
                "label_name" : data["label_name"],
                "feature" : output
            }    
            self.feature_list.append(output_dict)
        return self.feature_list
    
    def save(self, name, path):
        shape = self.feature_list[0]["feature"].shape
        
        saved_path = os.path.join(path , f"{name}{shape}.ft")
        
        torch.save(self.feature_list, saved_path)    

class VitFeatureExtractor(FeatureExtractor):
    def __init__(self,model_name:str = "vit_b_16", weights=None, num_of_classes = None):
        super().__init__(model_name, weights, num_of_classes)
        
    def _preprocess(self, image):
        # if image size is not 224,244 or 518,518, resize it

        if self.model_name == "vit_h_14":
            if image.width <518 or image.height < 518:
                raise ValueError("image size must be larger than 518,518")
            image = image.resize((518,518))
        else:
            if image.width <224 or image.height < 224:
                raise ValueError("image size must be larger than 224,224")
            image = image.resize((224,224))

        transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],std = [0.229, 0.224, 0.225])])
        
        transformed_image = transform(image).unsqueeze(0)
        return transformed_image.to(self.device)

    def _postprocess(self, output):
        output = output[self.last_layer_name]   
        return output
        
class ResnetFeatureExtractor(FeatureExtractor):
    def __init__(self, model_name:str = "resnet50", weights=None, num_of_classes = None):
        super().__init__(model_name, weights, num_of_classes)

    def _preprocess(self, image):
        transform = torchvision.transforms.Compose([torchvision.transforms.Resize((224,224)),
                                                    torchvision.transforms.ToTensor(),
                                                    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],std = [0.229, 0.224, 0.225])])
        
        transformed_image = transform(image).unsqueeze(0)
        return transformed_image.to(self.device)
    
    def _postprocess(self, output):
        output = output[self.last_layer_name]
        return output

