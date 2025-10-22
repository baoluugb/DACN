import torch, torch.nn as nn
from .posenc import SinusoidalPositionalEncoding2D
from torchvision import models

class Encoder(nn.Module):
    def __init__(self, d_model: int = 256):
        super().__init__()
        backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)


        # Take mean weights to convert from 3-channel to 1-channel input
        old_weights = backbone.conv1.weight.data  # [64, 3, 7, 7]
        new_weights = old_weights.mean(dim=1, keepdim=True)  # [64, 1, 7, 7]
        backbone.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        backbone.conv1.weight.data = new_weights


        self.feature_extractor = nn.Sequential(*list(backbone.children())[:-2])  # -> (B, 512, H/32, W/32)

        for param in self.feature_extractor.parameters():
            param.requires_grad = False


        for param in self.feature_extractor[-1].parameters():
            param.requires_grad = True

        for param in self.feature_extractor[-2].parameters():
            param.requires_grad = True

        self.proj = nn.Conv2d(512, d_model, kernel_size=1)

        self.pe2d = SinusoidalPositionalEncoding2D(d_model=d_model)

    def forward(self, x: torch.Tensor):
        feat = self.feature_extractor(x) # (B, 512, H', W')
        feat = self.proj(feat) # (B, d_model, H', W')
        seq = self.pe2d(feat) # (B, H'*W', d_model)
        return seq, feat.shape[-2:] # (sequence, spatial size)
