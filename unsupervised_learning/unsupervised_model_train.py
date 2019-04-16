from __future__ import print_function, division
import os, sys
import torch
import numpy as np
import torchvision as tv
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
from torchvision.utils import save_image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torch.optim as optim

from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, normalized_mutual_info_score

sys.path.insert(0, os.path.join(".."))
import pythonscraper.db as db
sys.path.insert(0, os.path.join("supervised_learning"))
from supervised_learning.load_dataset_train import load_dataset, transform_dataset


# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.join(".."))
import pythonscraper.db as db
sys.path.insert(0, os.path.join("unsupervised_learning"))

from autoencoder import AutoEncoder
input_size = 224

trainTransforms = tv.transforms.Compose([
                 tv.transforms.Resize(input_size),
                 tv.transforms.CenterCrop(input_size),
                 tv.transforms.ToTensor(), 
                 tv.transforms.Normalize(
                     (0.4914, 0.4822, 0.4466), 
                     (0.247, 0.243, 0.261)
                    )
                 ])


# Here load data
# TODO: use supervised DataLoader to load images and pass trainTransforms in arg
data_dir = "../comics"
labels_to_idx = {
        "Golden Age" : 0,
        "Silver Age" : 1,
        "Bronze Age" : 2,
        "Modern Age" : 3
    }
image_datasets = {}
dataset_full = load_dataset(data_dir, trainTransforms, labels_to_idx)

# Split in train, val and test from the image list
np.random.seed(42)
image_datasets["train"], image_datasets["test"] = train_test_split(dataset_full)
image_datasets["train"], image_datasets["val"] = train_test_split(image_datasets["train"])

# Training
num_epochs = 5 #you can go for more epochs, I am using a mac
batch_size = 8

# Create training, validation and test dataloaders
dataloaders_dict = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=4) for x in ["train", "val", "test"]}

ae = AutoEncoder().cuda()
distance = nn.MSELoss()
optimizer = optim.Adam(ae.parameters(), lr=0.001)

ae.train()

for epoch in range(num_epochs):
    for phase in [ "train", "val", "test" ]:

        if phase == "train":
            ae.train()   # Set ae to training mode
        elif phase == "val":
            ae.eval()    # Set ae to evaluate mode
        else:
            ae.eval()   

        for data in dataloaders_dict[phase]:
            inputs, labels = data
            inputs = Variable(inputs).cuda()
            output = ae(inputs)
            loss = distance(output, inputs)
            optimizer.zero_grad()
            if phase == "train":
                loss.backward()
                optimizer.step()
        print('{} epoch [{}/{}], loss:{:.4f}'.format(phase, epoch+1, num_epochs, loss.item()))

train_x = []
train_y = []
val_x = []
val_y = []
for data in dataloaders_dict["train"]:
    inputs, labels = data
    train_x.append(Variable(inputs).cuda())
    train_y.append(labels)

for data in dataloaders_dict["val"]:
    inputs, labels = data
    val_x.append(Variable(inputs).cuda())
    val_y.append(labels)

pred_auto_train = [ ae.encoder(inp) for inp in train_x ]
pred_auto = [ ae.encoder(inp) for inp in val_x ]

km = KMeans(n_jobs=-1, n_clusters=4, n_init=20)

km.fit(pred_auto_train)
pred = km.predict(pred_auto)

normalized_mutual_info_score(val_y, pred)
print(pred)