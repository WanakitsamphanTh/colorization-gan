import numpy as np
import torch
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(3,64,3,padding=1), #(,64,96,96)
            nn.Conv2d(64,64,3,padding=1), #(,64,96,96)
            nn.Conv2d(64,128,3,padding=1,stride=2), #(,128,48,48)
            nn.Conv2d(128,128,3,padding=1), #(,128,48,48)
            nn.Conv2d(128,256,3,padding=1,stride=3), #(,256,16,16)
            nn.Flatten()
        )

        self.ff = nn.Linear(256*16*16, 1)


    def forward(self,x):
        y = self.conv(x)
        y = self.ff(y)
        y = torch.sigmoid(y) 

        return y