# This implementation is inspired from
# https://www.kaggle.com/itsahmad/starter-mit-indoor-scenes-pytorch

from PIL import Image
import os

from torchvision.datasets.folder import accimage_loader
from torchvision import datasets


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def find_classes(dataset_dir):
    classes = [d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx


def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)


class Scenes(datasets.ImageFolder):
    def __init__(self, root, transform=None, target_transform=None, train=True):
        images_names_file = 'Images.txt'
        if train:
            images_names_file = 'Train' + images_names_file
        else:
            images_names_file = 'Test' + images_names_file
        file_names = open(os.path.join(root, images_names_file)).read().splitlines()
        self.root = os.path.join(root, 'indoorCVPR_09/Images')
        classes, class_to_idx = find_classes(self.root)

        images = []

        for filename in list(set(file_names)):
            target = filename.split('/')[0]
            path = os.path.join(root, 'indoorCVPR_09/Images/' + filename)
            item = (path, class_to_idx[target])
            images.append(item)

        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = images

        self.images = self.samples
        self.loader = default_loader
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.samples)
