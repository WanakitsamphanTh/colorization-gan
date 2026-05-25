import numpy as np
import torch
import torch.nn as nn

"""
Input: (X,C) where X is a collection of grayscale images with dimension (1,96,96)
 and C is a collection of color palettes in (3,2,2)
"""

class ColorGenerator(nn.Module):
    def __init__(self, output_chanels = 3, fn = torch.sigmoid):
        super(self, ColorGenerator).__init__()

        self.xconv1 = nn.Sequential(
            nn.Conv2d(1,32,3,padding=1), #(,32,96,96)
            nn.Conv2d(32,64,3,padding=1) #(,64,96,96)
        )

        self.xconv2 = nn.Sequential(
            nn.Conv2d(64,64,3,padding=1,stride=2), #(,64,48,48)
            nn.Conv2d(64,64,3,padding=1,stride=2) #(,64,24,24)
        )

        self.xconv3 = nn.Sequential(
            nn.Conv2d(64,128,3,padding=1) #(,128,96,96)
        )

        self.xconv4 = nn.Sequential(
            nn.Conv2d(128,128,3,padding=1,stride=2) #(,128,48,48)
        )

        self.xconv5 = nn.Sequential(
            nn.Conv2d(128,256,3,padding=1), #(,128,96,96)
        )

        self.clinear = nn.Linear(3*2*2, 8*8*8)

        self.cconvt = nn.Conv2dTranspose(8,64,3,padding=0,stride=3) # (,64,24,24)

        self.xcconvt1 = nn.Conv2d(128,128,3,padding=0,output_padding=1,stride=2) #(,128,48,48)
        
        self.xcconvt2 = nn.Conv2d(256,256,3,padding=0,output_padding=1,stride=2) #(,256,96,96)

        self.final_conv = nn.Conv2d(512,output_chanels,3,padding=1) #(,3,96,96)

        self.fn = fn


    def forward(self, x, c):
        # Convolution of greyscale image
        x = self.xconv1(x) #(,64,96,96)
        _xc = self.xconv2(x) #(,64,24,24)
        
        # Generating colors from c
        c = c.view([-1,3*2*2])
        c = self.clinear(c)
        c = c.view([-1,8,8,8])
        c = self.cconvt(c) # (,64,24,24)
        
        # Concat colors to scaled-downn image and feed forward
        xc = torch.concat([_xc,c], dim=1)
        xc = self.xcconvt1(xc) #(,128,48,48)

        # Further convolution
        x = self.xconv3(x) #(,128,96,96)

         # Concat colors to scaled-downn image and feed forward
        _xc = self.xconv4(x) #(,128,48,48)
        xc = torch.concat([_xc,xc], dim=1) #(,256,48,48)
        xc = self.xcconvt2(xc) #(,256,96,96)

        # Further convolution
        x = self.xconv3(4) #(,256,96,96)

        # Concat generated colors to real image
        xc = torch.concat([x,xc], dim=1)
        y = torch.final_conv(xc)

        y = self.fn(y)
        
        return y


        
        