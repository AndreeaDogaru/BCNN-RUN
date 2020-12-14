from bcnn_helper import BCNNManager
import os
from models import BCNN
from datasets.mit_dataset import Scenes

root_dir = ""

path = {
    'dataset': os.path.join(root_dir, "MIT"),
    'model': os.path.join(root_dir, 'checkpoints', "mit_fc"),
    # 'pretrained': os.path.join(root_dir, 'checkpoints', "mit_fc.pth")
}

options_fc = {
    'base_lr': 1.0,
    'batch_size': 64,
    'epochs': 55,
    'weight_decay': 1e-8,
}
options_all = {
    'base_lr': 1e-2,
    'batch_size': 32,
    'epochs': 25,
    'weight_decay': 1e-5,
}

config = {
    'device': 'cuda:0',
    'train_options': options_fc,
    'path': path,
    'model': BCNN,
    'dataset': Scenes,
    'freeze_features': True
}

bcnn_manager = BCNNManager(config)
bcnn_manager.train()

