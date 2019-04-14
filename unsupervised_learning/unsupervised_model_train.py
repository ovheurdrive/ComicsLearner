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


# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.join(".."))
import pythonscraper.db as db
sys.path.insert(0, os.path.join("unsupervised_learning"))

from autoencoder import AutoEncoder
import pythonscraper.db as db

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4466), (0.247, 0.243, 0.261))])

trainTransforms = tv.transforms.Compose([
                 tv.transforms.ToTensor(), 
                 tv.transforms.Normalize(
                     (0.4914, 0.4822, 0.4466), 
                     (0.247, 0.243, 0.261)
                    )
                 ])


# Here load data
# TODO: use supervised DataLoader to load images and pass trainTransforms in arg
dataloader = DataLoader([])

# Training
num_epochs = 5 #you can go for more epochs, I am using a mac
batch_size = 128

ae = AutoEncoder().cuda()
distance = nn.MSELoss()
optimizer = optim.Adam(ae.parameters(), lr=0.001)

for epoch in range(num_epochs):
    for data in dataloader:
        inputs, labels = data
        inputs = Variable(inputs).cuda()
        output = ae(inputs)
        loss = distance(output, inputs)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print('epoch [{}/{}], loss:{:.4f}'.format(epoch+1, num_epochs, loss.data()))