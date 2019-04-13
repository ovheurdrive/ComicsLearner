from __future__ import print_function, division
import os, sys
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")


sys.path.insert(0, os.path.join(".."))
import pythonscraper.db as db
sys.path.insert(0, os.path.join("supervised_learning"))


import ComicPageDataset

def load_dataset(root_dir, data_transforms):
    files = db.query("SELECT * from files", ())
    all_comic_images = []
    for file in files:
        new_page = {}
        new_page["filename"] = file[1]
        new_page["label"] = file[3]
        new_page["comic_name"] = new_page["filename"].split("/")[1]
        all_comic_images.append(new_page)

    comic_page_dataset = ComicPageDataset(root_dir=root_dir, all_comic_images=all_comic_images, transform=data_transforms)

    for i in range(len(comic_page_dataset)):
        sample = comic_page_dataset[i]
