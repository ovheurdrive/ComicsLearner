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
    def __init__(self, root_dir, all_comic_images, transform):
        """
        Args:
            root_dir (string): Directory with all the images.
            all_comic_images (list): List of all the existing images
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.to_tensor = transforms.ToTensor()
        self.root_dir = root_dir
        self.all_comic_images = all_comic_images
        self.transform = transform

    def __len__(self):
        return len(self.all_comic_images)
    
    def __getitem__(self, idx):
        img_name = self.all_comic_images[idx]["filename"]
        with open(os.path.join("..", img_name), 'rb') as f:
            image = Image.open(f)
            if self.transform:
                image = self.transform(image)
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

        # img_as_tensor = self.to_tensor(image)

        return (image, label)