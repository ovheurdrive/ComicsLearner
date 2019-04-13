from __future__ import print_function, division
import os, sys
import torch
from PIL import Image
# Allow truncated images
Image.LOAD_TRUNCATED_IMAGES = True

from skimage import transform
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
    def __init__(self, root_dir, all_comic_images):
        """
        Args:
            root_dir (string): Directory with all the images.
            all_comic_images (list): List of all the existing images
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.all_comic_images = all_comic_images

    def __len__(self):
        return len(self.all_comic_images)
    
    def __getitem__(self, idx):
        img_name = self.all_comic_images[idx]["filename"]
        with open(os.path.join("..", img_name), 'rb') as f:
            image = Image.open(f)
            image = image.convert('RGB')
        # comic_name = self.all_comic_images[idx]["comic_name"]
        
        publication_year = self.all_comic_images[idx]["label"]
        if(int(publication_year) < 1954):
            label = "Golden Age"
        elif(int(publication_year) < 1970):
            label = "Silver Age"
        elif(int(publication_year) < 1986):
            label = "Bronze Age"
        else:
            label = "Modern Age"

        sample = { "image": image, "label": label }

        return sample
    """
    def __transform__(self):
        if self.transform:
            for i in range(self.__len__()):
                sample = self.__getitem__(i)
                sample = self.transform(sample["image"])
    """