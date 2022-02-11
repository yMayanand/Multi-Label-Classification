import torch.nn as nn
from torchvision import models

def get_model(model_name, pretrained=False):
    model = getattr(models, model_name, None)
    assert model != None, 'model not found'
    model = model(pretrained=pretrained)
    model.fc = nn.Linear(model.fc.in_features, 20)
    return model
