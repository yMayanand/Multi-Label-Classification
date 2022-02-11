import fastbook
from torchvision import models, transforms
from fastai.vision.all import *
from data import Dataset, get_dl
from model import get_model

imagenet_stats = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

model = get_model('resnet18', pretrained=True)

train_tfms = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.4, 1)),
    transforms.ToTensor(),
    transforms.Normalize(*imagenet_stats)
])

val_tfms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(*imagenet_stats)
])

train_dl = get_dl('train', transform=train_tfms, shuffle=True)
val_dl = get_dl('val', transform=val_tfms)


