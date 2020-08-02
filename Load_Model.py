from vision.torchvision.models.detection import fasterrcnn_resnet50_fpn
from vision.torchvision.models.detection.faster_rcnn import FastRCNNPredictor

import torch
import os

def load_model(save_path, num_classes, device):
    
    model = fasterrcnn_resnet50_fpn(pretrained=True)

    #Tuning the model
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor.box_predictor = FastRCNNPredictor(in_features, num_classes=num_classes)
    
    # Moving model to CPU or GPU
    model.to(device)
    
    model.load_state_dict(torch.load(os.path.join(save_path)))
    
    return model