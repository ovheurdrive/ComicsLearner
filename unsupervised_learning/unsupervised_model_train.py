from __future__ import print_function, division
# Ignore warnings
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import os, sys
import torch
import numpy as np
import torchvision as tv
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torch.optim as optim

from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, normalized_mutual_info_score, confusion_matrix

sys.path.insert(0, os.path.join(".."))
import pythonscraper.db as db
sys.path.insert(0, os.path.join("supervised_learning"))
from supervised_learning.load_dataset_train import load_dataset, transform_dataset
sys.path.insert(0, os.path.join("unsupervised_learning"))
from autoencoder import AutoEncoder



def train_ae(dataloaders_dict, device=0, num_epochs=5):

    ae = AutoEncoder()
    ae = ae.to(device)
    distance = nn.MSELoss()
    optimizer = optim.Adam(ae.parameters(), lr=0.001)

    for epoch in range(num_epochs):
        print('\nEpoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        for phase in [ "train", "val", "test" ]:

            if phase == "train":
                ae.train()   # Set ae to training mode
            else:
                ae.eval()

            for data in dataloaders_dict[phase]:
                inputs, labels = data
                inputs = inputs.to(device)
                with torch.set_grad_enabled(phase == "train"):
                    output = ae(inputs)
                    loss = distance(output, inputs)
                    optimizer.zero_grad()
                    if phase == "train":
                        loss.backward()
                        optimizer.step()
            print("{} Loss: {:.4f}".format(phase,loss.item()))
    return ae



def cluster(dataloaders_dict, ae, device=0):
    train_x = []
    train_y = np.array([])
    val_x = []
    val_y = np.array([])
    for data in dataloaders_dict["train"]:
        inputs, labels = data
        inputs = inputs.to(device)
        train_x.append(inputs)
        train_y = np.concatenate(( train_y, labels.numpy()))

    for data in dataloaders_dict["val"]:
        inputs, labels = data
        inputs = inputs.to(device)
        val_x.append(inputs)
        val_y = np.concatenate(( val_y, labels.numpy()))

    pred_auto_train = None
    pred_auto = None

    for inp in train_x:
        out = ae.encoder(inp)
        size = out.size()
        out = out.view(size[0], -1)
        np_out = out.detach().cpu().numpy()
        pred_auto_train = np.concatenate( (pred_auto_train, np_out) , axis=0) if pred_auto_train is not None else np_out

    for inp in val_x:
        out = ae.encoder(inp)
        size = out.size()
        out = out.view(size[0], -1)
        np_out = out.detach().cpu().numpy()
        pred_auto = np.concatenate( (pred_auto, np_out) , axis=0) if pred_auto is not None else np_out

    km = KMeans(n_jobs=-1, n_clusters=4, n_init=20)

    km.fit(pred_auto_train)
    pred = km.predict(pred_auto)

    build_plots(pred_auto, pred, val_y)


def build_plots(pred_auto, pred, val_y):
    plt.scatter(np.array(pred_auto)[:, 0], np.array(pred_auto)[:, 1], c=pred, s=50, cmap='viridis')
    plt.savefig("scatter_plot_{}dp.png".format(len(pred)))

    mat = confusion_matrix(val_y, pred)
    sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False)
    plt.xlabel('true label')
    plt.ylabel('predicted label')
    plt.savefig("confusion_matrix_{}dp.png".format(len(pred)))

    nmi = normalized_mutual_info_score(val_y, pred)
    acc = accuracy_score(val_y, pred)

    print("NMI Score: {}, Acc Score: {}".format(nmi, acc))


def build_dataloaders(data_dir, trainTransforms, labels_to_idx, batch_size, test=None):
     # Here load data
    image_datasets = {}
    phase = ["train", "val", "test"]
    dataset_full = load_dataset(data_dir, trainTransforms, labels_to_idx)

    # Split in train, val and test from the image list
    np.random.seed(42)
    image_datasets["train"], image_datasets["val"] = train_test_split(dataset_full)
    image_datasets["train"], image_datasets["test"] = train_test_split(image_datasets["train"])

    # Create training, validation and test dataloaders
    dataloaders_dict = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=4) for x in phase}
    return dataloaders_dict


def main():
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
    data_dir = "../comics"
    labels_to_idx = {
        "Golden Age" : 0,
        "Silver Age" : 1,
        "Bronze Age" : 2,
        "Modern Age" : 3
    }
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dataloaders = build_dataloaders(data_dir, trainTransforms, labels_to_idx, 8, True)
    ae = train_ae(dataloaders, device, 3)
    cluster(dataloaders, ae, device)


if __name__ == "__main__":
    main()