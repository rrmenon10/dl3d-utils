import os
import copy
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms

from models import ResNetSimCLR


class SimCLR(nn.Module):

    def __init__(self,
                 img_shape,
                 model_name='resnet18',
                 embedding_size=512,
                 temp=0.8,
                 transform=True,
                 device=torch.device('cpu')):
        
        super(SimCLR, self).__init__()
        
        self.img_shape = img_shape
        self.temp = temp
        self.device = device
        self.C = embedding_size
        self.criterion = torch.nn.CrossEntropyLoss(reduction="sum")

        # Define transforms
        color_jitter = transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)
        self.transforms = transforms.Compose([transforms.ToPILImage(),
                                              transforms.RandomResizedCrop(size=self.img_shape),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.RandomApply([color_jitter], p=0.8),
                                              transforms.RandomGrayscale(p=0.2),
                                              transforms.ToTensor()]) if transform else lambda x: x


        # Defining model for computations
        self.model = ResNetSimCLR(model_name, self.C).to(device)
    
    def get_input(self, samples):

        samples = torch.stack([self.transforms(sample) for sample in samples], 
                              dim=0)
        return samples
    
    def forward(self, samples):

        N, H, W, _ = samples.shape

        x_t1 = self.get_input(samples).to(self.device)
        x_t2 = self.get_input(samples).to(self.device)

        # Get projection head representations
        z1, z2 = self.model(x_t1), self.model(x_t2)
        rep = torch.cat([z1, z2], dim=0)    # [2N, C]
        rep = F.normalize(rep, dim=1)

        # Compute similarities between representations
        sim = torch.mm(rep, rep.T)          # [2N, 2N]
        ij_pos = torch.diag(sim, N)         # [N, ]
        ji_pos = torch.diag(sim, -N)        # [N, ]
        positive_sim = torch.cat([ij_pos, ji_pos])[:, None] # [2N, 1]

        # pos_mask
        mask = torch.diag(torch.ones(N), N) + torch.diag(torch.ones(N), -N) + torch.eye(2*N) # [2N, 2N]
        # neg_mask
        mask = (1. - mask).bool().to(self.device)
        negative_sim = sim[mask].reshape(2*N, -1)           # [2N, 2*(N-1)]
        assert negative_sim.shape[-1]==2*(N-1)

        logits = torch.cat([positive_sim, negative_sim], dim=1)    # [2N, 2N-1]
        labels = torch.zeros(2*N).long().to(self.device)
        loss = self.criterion(logits/self.temp, labels)
        loss /= 2*N

        return loss