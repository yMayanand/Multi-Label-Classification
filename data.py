<<<<<<< HEAD
import torch
from torchvision import transforms, datasets


classes = [
    "aeroplane",
    "bicycle", 
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
=======
import torchvision

classes = [
    "aeroplane",
    "bicycle," "bird",
    "boat",
    "bottle",
    "bus," "car",
>>>>>>> f241b3c6d34e36373b6527ac736b043cf33c8b7d
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
]

cls2idx = {key: value for value, key in enumerate(classes)}


class Dataset:
<<<<<<< HEAD
    def __init__(self, image_set, transform=None):
        self.image_set = image_set
        self.transform = transform
        self.ds = datasets.VOCDetection("./", image_set=image_set, download=True)
=======
    def __init__(self, image_set):
        self.ds = torchvision.datasets.VOCDetection(
            "./", image_set=image_set, download=True
        )
>>>>>>> f241b3c6d34e36373b6527ac736b043cf33c8b7d

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        image = self.ds[idx][0]
<<<<<<< HEAD
        if self.transform is not None:
            image = self.transform(image)

        label = set()
        label.update([i["name"] for i in self.ds[idx][1]["annotation"]["object"]])
        label = list(map(lambda x: cls2idx[x], label))
        label = torch.tensor(label)
        return image, label


def get_dl(set_type, transform=None,  bs=32, shuffle=False):
    ds = Dataset(set_type, transform=transform)
    dl = torch.utils.data.DataLoader(ds, bs, shuffle=shuffle)
    return dl
=======
        label = set()
        label.update([i["name"] for i in self.ds[idx][1]["annotation"]["object"]])
        return image, label
>>>>>>> f241b3c6d34e36373b6527ac736b043cf33c8b7d
