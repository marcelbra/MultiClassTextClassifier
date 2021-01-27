from __future__ import print_function, division
from torch.utils.data import DataLoader
import os
import torch
import pandas as pd
#from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")


class Dataset(Dataset):
    """Dataset desccription."""

    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def set_data(self, other):
        self.data = pd.concat([self.data, other.data], axis=1)

    def __getitem__(self, idx):
        text = self.data["text"][idx]
        author = self.data["author"][idx]
        return text, author
