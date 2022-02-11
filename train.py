import fastbook # only for google colab else comment out this
from fastai.vision.all import *
from torch import optim, nn
from torchvision import models, transforms
from data import Dataset, get_dl
from model import get_model
from functools import partial

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

dls = DataLoaders(train_dl, val_dl)

opt_func = partial(OptimWrapper, opt=optim.Adam)

learn = Learner(dls, model, opt_func=opt_func, loss_func=nn.BCEWithLogitsLoss(), metrics=accuracy_multi)


