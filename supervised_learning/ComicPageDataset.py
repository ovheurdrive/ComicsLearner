from __future__ import print_function, division
import os
import torch
from skimage import io, transform
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

class ComicPageDataset(Dataset):
    """
    Comic Page Dataset.
    """
    def __init__(self, root_dir, all_comic_images, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            all_comic_images (list): List of all the existing images
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.all_comic_images = all_comic_images
        self.transform = transform

    def __len__(self):
        return len(self.all_comic_images)
    
    def __get__item(self, idx):
        img_name = self.all_comic_images[idx]["filename"]
        image = io.imread(img_name)
        # comic_name = self.all_comic_images[idx]["comic_name"]
        
        publication_year = self.all_comic_images[idx]["label"]
        if(publication_year < 1954):
            label = "Golden Age"
        elif(publication_year < 1970):
            label = "Silver Age"
        elif(publication_year < 1986):
            label = "Bronze Age"
        else:
            label = "Modern Age"

        sample = { "image": image, "label": label }

        if self.transform:
            sample = self.transform(sample["image"])
        
        return sample