# This implementation is inspired from
# https://www.kaggle.com/itsahmad/starter-mit-indoor-scenes-pytorch
# and https://github.com/jiaxue1993/pytorch-material-classification

from PIL import Image
import os

from torchvision.datasets.folder import accimage_loader
from torchvision import datasets

import os.path

def find_classes(dir):
    classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)


def make_dataset(txtnames, datadir, class_to_idx):
    images = []
    for txtname in txtnames:
        with open(txtname, 'r') as lines:
            for line in lines:
                classname = line.split('/')[0]
                _img = os.path.join(datadir, 'images', line.strip())
                assert os.path.isfile(_img)
                item = (_img, class_to_idx[classname])
                images.append(item)

    return images


class DTD(datasets.ImageFolder):
    def __init__(self, root, transform=None, target_transform=None, train=True):

        classes, class_to_idx = find_classes(root)

        if train:
            filename = [os.path.join(root, "labels/train1.txt"),
                        os.path.join(root, "labels/val1.txt")]
        else:
            filename = [os.path.join(root, "labels/test1.txt")]

        images = make_dataset(filename, root, class_to_idx)

        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = images

        self.images = self.samples
        self.loader = default_loader
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.samples)