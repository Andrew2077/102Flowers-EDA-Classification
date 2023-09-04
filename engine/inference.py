from gradcam import GradCAM
from models import Resnet50Flower102
import torch

def load_cam_model(path):
    # * load the model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Resnet50Flower102(device)

    # * load gradcam, update the model with the best weights
    gradcam = GradCAM(model, device)
    gradcam.model.load_state_dict(torch.load(path))
    gradcam.model.eval()
    print("model & gradcam loaded")
    
    return gradcam


def load_model(path):
    # * load the model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Resnet50Flower102(device)

    # * load the best weights
    model.load_state_dict(torch.load(path))
    model.eval()
    print("model loaded")
    
    return model