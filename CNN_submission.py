import timeit
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import random_split
from torchvision import transforms, datasets
import numpy as np
import random

from torchvision.models import *

#Function for reproducibilty. You can check out: https://pytorch.org/docs/stable/notes/randomness.html
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
set_seed(100)

#TODO: Populate the dictionary with your hyperparameters for training
def get_config_dict(pretrain):
    """
    pretrain: 0 or 1. Can be used if you need different configs for part 1 and 2.
    """
    
    config = {
        "batch_size": 128 if not pretrain else 256,
        "lr": 1e-3 if not pretrain else 1.5e-4,
        "num_epochs": 10 if not pretrain else 5,
        "weight_decay": 0 if not pretrain else 1e-4,   #set to 0 if you do not want L2 regularization
        "save_criteria": None if not pretrain else 'accuracy',     #Str. Can be 'accuracy'/'loss'/'last'. (Only for part 2)

    }
    
    return config
    

#TODO: Part 1 - Complete this with your CNN architecture. Make sure to complete the architecture requirements.
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 5)
        self.conv2 = nn.Conv2d(16, 32, 5)
        self.fc1 = nn.Linear(32*5*5, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 10)
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


#TODO: Part 2 - Complete this with your Pretrained CNN architecture. 
class PretrainedNet(nn.Module):
    def __init__(self):
        super(PretrainedNet, self).__init__()
        # TODO: Load a pretrained model
        self.model = resnet18(ResNet18_Weights.DEFAULT)
        print("Model summary:",self.model)

    def forward(self, x):
        x = self.model(x)
        return x


#Feel free to edit this with your custom train/validation splits, transformations and augmentations for CIFAR-10, if needed.
def load_dataset(pretrain):
    """
    pretrain: 0 or 1. Can be used if you need to define different dataset splits/transformations/augmentations for part 2.

    returns:
    train_dataset, valid_dataset: Dataset for training your model
    test_transforms: Default is None. Edit if you would like transformations applied to the test set. 

    """
    weights = ResNet18_Weights.DEFAULT
    preprocess = weights.transforms()

    full_dataset = datasets.CIFAR10(root='./data', train=True, download=True,
                                    transform=preprocess if pretrain == 1 else transforms.Compose([
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]))

    train_dataset, valid_dataset = random_split(full_dataset, [38000, 12000])

    test_transforms = preprocess if pretrain == 1 else None

    
    return train_dataset, valid_dataset, test_transforms

