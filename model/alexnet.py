import torch
import torch.nn as nn
from torchvision import models


class AlexNet(nn.Module):
    def __init__(self, bits, classes, class_mask_rate):
        super(AlexNet, self).__init__()
        self.model = models.alexnet(pretrained=True)
        self.model.classifier[6] = nn.Linear(4096, classes)
        self.model.b = nn.Linear(classes, bits)

        self.class_mask_rate = class_mask_rate

    def forward(self, x):
        # x = self.model(x)
        x = self.model.features(x)

        fm = x
        A = torch.sum(fm.detach(), dim=1, keepdim=True)
        a = torch.mean(A, dim=[2, 3], keepdim=True)
        M = (A > a).float().detach() + (A < a).float().detach() * 0.5
        # print(M.size())
        x = x * M

        x = self.model.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.model.classifier(x)

        x_mask = torch.ones(x.size()).detach().cuda() * self.class_mask_rate#0.1
        for i in range(x_mask.size()[0]):
            x_mask[i, torch.argmax(x[i])] = 1

        x_b = x * x_mask
        b = self.model.b(x_b)
        return fm, x, b