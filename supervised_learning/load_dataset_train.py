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


from supervised_learning.dataset_model import ComicPageDataset

def load_dataset(root_dir, data_transforms, labels_to_idx):
    files = db.query("SELECT * from files ORDER BY RANDOM() LIMIT 50", ())
    all_comic_images = []
    for file in files:
        new_page = {}
        new_page["filename"] = file[1]
        new_page["label"] = file[3]
        new_page["comic_name"] = new_page["filename"].split("/")[1]
        all_comic_images.append(new_page)

    comic_page_dataset = ComicPageDataset(root_dir=root_dir, all_comic_images=all_comic_images, transform=data_transforms, labels_to_idx=labels_to_idx)

    return comic_page_dataset

def transform_dataset(dataset, transform):
    n = len(dataset)
    for i in range(n):
        dataset[i]["image"] = transform(dataset[i]["image"])
    return dataset
