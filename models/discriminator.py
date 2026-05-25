import numpy as np
import torch
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, output_chanels = 3, fn = torch.sigmoid):
        super(self, Discriminator).__init__()

        nn.conv = nn.Sequential(
            nn.Conv2d(3,128,3,padding=1), #(,128,96,96)
            nn.Conv2d(128,256,3,padding=1,stride=3), #(,256,48,48)
            nn.Conv2d(256,512,3,padding=1,stride=2), #(,512,16,16)
        )

        nn.ff = nn.Linear(512*16*16, 1)


    def forward(self,x):
        y = self.conv(x)
        y = y.view([-1,512*16*16])
        y = torch.softmax(y) 

        return y