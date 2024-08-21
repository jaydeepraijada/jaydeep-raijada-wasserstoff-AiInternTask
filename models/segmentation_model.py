import torch
from torchvision.models.detection import maskrcnn_resnet50_fpn
import os



input_images_dir = 'data/input_images/'
segmented_objects_dir = 'data/segmented_objects/'
os.makedirs(input_images_dir, exist_ok=True)
os.makedirs(segmented_objects_dir, exist_ok=True)

#Loading the model

def load_model():
    model = maskrcnn_resnet50_fpn(pretrained=True)
    # Using a different backbone
    #model = maskrcnn_resnet50_fpn(pretrained=False, pretrained_backbone=False, backbone_name='resnext50_32x4d')
    model.eval()  
    """
    We have set this to evaluation mode, 
    because we have loaded a pretrained model 
    so we must deactivate dropout layers and other 
    training-specific behaviors.
    """
    return model

model = load_model() #model initialization


def run_inference(model,image_tensor):
    with torch.no_grad():
        outputs = model(image_tensor)
    return outputs



