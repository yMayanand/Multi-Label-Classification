import torchvision

classes = [ 'aeroplane',
'bicycle,'
'bird',
'boat',
'bottle',
'bus,'
'car',
'cat',
'chair',
'cow',
'diningtable',
'dog',
'horse',
'motorbike',
'person',
'pottedplant',
'sheep',
'sofa',
'train',
'tvmonitor']


class Dataset:
    def __init__(self, image_set):
        self.ds = torchvision.datasets.VOCDetection('./', image_set=image_set, download=True)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
