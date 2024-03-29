import torch
import torchvision as tv
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder,self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 6, kernel_size=5),
            nn.ReLU(True),
            nn.Conv2d(6, 16, kernel_size=5),
            nn.ReLU(True))
 
        self.decoder = nn.Sequential(             
            nn.ConvTranspose2d(16, 6, kernel_size=5),
            nn.ReLU(True),
            nn.ConvTranspose2d(6, 3, kernel_size=5),
            nn.ReLU(True),
            nn.Sigmoid())

    def forward(self,x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

class AutoEncoder2(nn.Module):
    def __init__(self):
        super(AutoEncoder2,self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2), 
            nn.Conv2d(16, 8,kernel_size=5), 
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=1)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(8, 16, 8, stride=2),
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 8, kernel_size=5), 
            nn.ReLU(True),
            nn.ConvTranspose2d(8, 3, kernel_size=5), 
            nn.Sigmoid()
        )
    def forward(self,x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x