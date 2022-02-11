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
    def __init__(self, image_set, transform=None):
        self.image_set = image_set
        self.transform = transform
        self.ds = datasets.VOCDetection("./", image_set=image_set, download=True)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        image = self.ds[idx][0]
        if self.transform is not None:
            image = self.transform(image)

        label = set()
        label.update([i["name"] for i in self.ds[idx][1]["annotation"]["object"]])
        label = list(map(lambda x: cls2idx[x], label))
        temp = torch.zeros(20)
        for i in label:
            temp[i] = 1
        label = temp
        return image, label


def get_dl(set_type, transform=None,  bs=32, shuffle=False):
    ds = Dataset(set_type, transform=transform)
    dl = torch.utils.data.DataLoader(ds, bs, shuffle=shuffle)
    return dl
