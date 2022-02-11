import torchvision

classes = [
    "aeroplane",
    "bicycle," "bird",
    "boat",
    "bottle",
    "bus," "car",
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
    def __init__(self, image_set):
        self.ds = torchvision.datasets.VOCDetection(
            "./", image_set=image_set, download=True
        )

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        image = self.ds[idx][0]
        label = set()
        label.update([i["name"] for i in self.ds[idx][1]["annotation"]["object"]])
        return image, label
