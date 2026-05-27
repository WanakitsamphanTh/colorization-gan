import numpy as np
import torch
import torch.nn as nn

"""
Input: (X,C) where X is a collection of grayscale images with dimension (1,96,96)
 and C is a collection of color palettes in (3,2,2)
"""
        

class ColorGenerator(nn.Module):
    def __init__(self, output_chanels = 3, n_cell = 2, fn = torch.sigmoid):
        super(ColorGenerator, self).__init__()

        self.xconv1 = nn.Sequential(
            nn.Conv2d(1,32,3,padding=1), #(,32,96,96)
            nn.LeakyReLU(inplace=True)
        )

        self.xconv2 = nn.Sequential(
            nn.Conv2d(32,64,3,padding=1,stride=2), #(,64,48,48)
            nn.LeakyReLU(inplace=True)
        )

        self.xconv3 = nn.Sequential(
            nn.Conv2d(64,256,3,padding=1,stride=3), #(,256,16,16)
            nn.LeakyReLU(inplace=True)
        )

        self.xconv4 = nn.Sequential(
            nn.Conv2d(256,16,3,padding=1,stride=2), #(,16,8,8)
            nn.LeakyReLU(inplace=True)
        )

        self.clinear = nn.Sequential(
            nn.Linear(16*8*8 + 3 * n_cell * n_cell, 256*16*16),
            nn.LeakyReLU(inplace=True)
        )


        self.cconvt1 = nn.Sequential(
            nn.ConvTranspose2d(256,64,3,padding=0,stride=3), # (,64,48,48)
            nn.LeakyReLU(inplace=True)
        )

        self.cconvt2 = nn.Sequential(
            nn.ConvTranspose2d(64,32,3,padding=1,output_padding=1,stride=2), #(,32,96,96)
            nn.LeakyReLU(inplace=True)
        )
        
        self.xcconv = nn.Sequential(
            nn.Conv2d(32,output_chanels,3,padding=1), #(,output_chanels,96,96)
        )

        self.fn = fn


    def forward(self, x, c):
        # Convolution of greyscale image
        x1 = self.xconv1(x) #(,32,96,96)
        x2 = self.xconv2(x1) #(,64,48,48)
        x3 = self.xconv3(x2) #(,128,16,16)
        x4 = self.xconv4(x3) #(,16,8,8)
        x4 = torch.flatten(x4, start_dim=1)


        # Generating colors from c
        c = torch.flatten(c, start_dim=1)
        c = torch.concat([x4,c], dim=1)
        c = self.clinear(c)
        c = c.view([-1,256,16,16])

        c = self.cconvt1(c) # (,64,48,48)
        c = self.cconvt2(c) # (,32,96,96)
        c = self.xcconv(c) # (,output_chanels,96,96)
        y = self.fn(c)
        
        return y


        
        