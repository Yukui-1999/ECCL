import torch
import torch.nn as nn
import torchvision.models as models


class ResNet18(nn.Module):
    def __init__(self, in_channels, latent_dim=2048,pretrained=False):
        super(ResNet18, self).__init__()
        self.base_model = models.resnet18(pretrained=pretrained)  # 使用预训练的权重或者不使用预训练权重
        self.base_model.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.base_model.fc = nn.Linear(512, latent_dim)

    def forward(self, x):
        return self.base_model(x)

class ResNet34(nn.Module):
    def __init__(self, in_channels, latent_dim=2048,pretrained=False):
        super(ResNet34, self).__init__()
        self.base_model = models.resnet34(pretrained=pretrained)  # 使用预训练的权重或者不使用预训练权重
        self.base_model.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.base_model.fc = nn.Linear(512, latent_dim)

    def forward(self, x):
        return self.base_model(x)

class ResNet50(nn.Module):
    def __init__(self, in_channels, latent_dim=2048,pretrained=False):
        super(ResNet50, self).__init__()
        self.base_model = models.resnet50(pretrained=pretrained)  # 使用预训练的权重或者不使用预训练权重
        self.base_model.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.base_model.fc = nn.Linear(2048, latent_dim)

    def forward(self, x):
        return self.base_model(x)
    
class ResNet101(nn.Module):
    def __init__(self, in_channels, latent_dim=2048,pretrained=False):
        super(ResNet101, self).__init__()
        self.base_model = models.resnet101(pretrained=pretrained)  # 使用预训练的权重或者不使用预训练权重
        self.base_model.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.base_model.fc = nn.Linear(2048, latent_dim)

    def forward(self, x):
        return self.base_model(x)


def resnet18(in_channels,latent_dim,pretrained):
    model = ResNet18(in_channels,latent_dim,pretrained)
    return model

def resnet34(in_channels,latent_dim,pretrained):
    model = ResNet34(in_channels,latent_dim,pretrained)
    return model

def resnet50(in_channels,latent_dim,pretrained):
    model = ResNet50(in_channels,latent_dim,pretrained)
    return model

def resnet101(in_channels,latent_dim,pretrained):
    model = ResNet101(in_channels,latent_dim,pretrained)
    return model