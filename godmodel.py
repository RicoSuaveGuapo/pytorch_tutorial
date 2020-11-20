import os
import cv2
import math
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
# pip install pretrainedmodels
import pretrainedmodels
# https://github.com/Cadene/pretrained-models.pytorch

# just for calculating the output size after the convolution
def outSize(input_size, kernal, stride, padding):
    # recall the formula (W−F+2P)/S+1
    outsize = math.floor((input_size - kernal + 2 * padding)/stride +1)
    return outsize

# =======================
# || Self define model ||
# =======================
# all model should inherit nn.Module
# implement the very old and very simple LeNet
class GodModelSelf(nn.Module):
    def __init__(self, args, hidden_dim=64, dropout=0.5, classes=2):
        # must inherit nn.Module class
        super().__init__()

        # let the image size as 256x256
        self.c1 = nn.Conv2d(in_channels=3,out_channels=6,kernel_size=5,stride=1) # (6, 252)
        self.s2 = nn.AvgPool2d(kernel_size=2) # (6, 126)
        self.c3 = nn.Conv2d(in_channels=6,out_channels=16,kernel_size=5,stride=1) # (16, 122)
        self.s4 = nn.AvgPool2d(kernel_size=2) # (16, 61)
        self.c5 = nn.Conv2d(in_channels=16,out_channels=120,kernel_size=5,stride=1) # (120, 57)
        self.ac = nn.Tanh()

        # calculate how may neurons needed for the 1st FC
        c1_size = outSize(args.image_size,5,1,0)
        s2_size = outSize(c1_size,2,2,0)
        c3_size = outSize(s2_size,5,1,0)
        s4_size = outSize(c3_size,2,2,0)
        c5_size = outSize(s4_size,5,1,0)
        feat_size = c5_size**2*120
        # print(feat_size)

        # you can pack your own sequential block
        self.classifier = nn.Sequential(
            nn.Linear(in_features=feat_size, out_features=hidden_dim),
            nn.Tanh(),
            nn.Dropout(p=dropout),
            nn.Linear(in_features=hidden_dim, out_features=classes),
            nn.Tanh()
        )

    def features(self, input):
        f = self.c1(input)
        f = self.ac(f)
        f = self.s2(f)
        f = self.c3(f)
        f = self.ac(f)
        f = self.s4(f)
        f = self.c5(f)
        f = self.ac(f)
        return f

    def logits(self, feature):
        output = feature.view(feature.size(0), -1)
        output = self.classifier(output)
        return output
    
    def forward(self, input):
        x = self.features(input)
        x = self.logits(x)
        return x

# ====================
# ||   Exist Model  ||
# ====================
# the base model I choose resnet18
class GodModelPretrained(nn.Module):
    def __init__(self, hidden_dim=64, dropout=0.5, classes=2):
        super().__init__()
        self.base = pretrainedmodels.__dict__['resnet18'](num_classes=1000, pretrained='imagenet') # the pretrain model
        # the list of aviable models
        # print(pretrainedmodels.model_names)
        # get the dim of output feature map from base model
        dim_feats = self.base.last_linear.in_features 
        self.linear1 = nn.Linear(dim_feats, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, classes)
        self.dropout = nn.Dropout(p=dropout)
        self.pool = nn.AdaptiveAvgPool2d((1,1))
        self.act = nn.ReLU()

    def features(self, input):
        return self.base.features(input)

    def logits(self, feature):
        output = self.pool(feature)
        output = output.view(output.size(0), -1)
        output = self.linear1(output)
        output = self.act(output)
        output = self.dropout(output)
        output = self.linear2(output)
        return output
    
    def forward(self, input):
        x = self.features(input)
        x = self.logits(x)
        return x

if __name__ == "__main__":
    import easydict
    from torch.utils.tensorboard import SummaryWriter
    
    args = easydict.EasyDict({'image_size':256})
    model = GodModelSelf(args=args)
    # this is a simple check for the model
    # print(model)

    img = torch.randn((1,3,256,256))
    writer = SummaryWriter('runs/model')
    writer.add_graph(model, img)
    writer.close()
    