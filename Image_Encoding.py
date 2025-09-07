import torch
import torch.nn as nn
from torch.amp import autocast
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.norm1 = nn.GroupNorm(8, channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.norm2 = nn.GroupNorm(8, channels)
        
    def forward(self, x):
        residual = x
        x = F.gelu(self.norm1(self.conv1(x)))
        x = self.norm2(self.conv2(x))
        return F.gelu(x + residual)
        
class CompactCNN(nn.Module):
    def __init__(self, output_dim=256):
        super().__init__()
        self.conv = nn.Sequential(        
            nn.Conv2d(3, 32, kernel_size=6, stride=2, padding=2, bias=False), # 256->128
            nn.GroupNorm(8, 32), nn.GELU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False), # 128->64
            nn.GroupNorm(8, 64), nn.GELU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False), # stays 64
            nn.GroupNorm(8, 128), nn.GELU(),
    
            # ADD: One residual block here (2 conv layers but with skip connection)
            # This adds capacity without hurting gradient flow
            ResidualBlock(128),  # Most important level for path features
    
            # Rest unchanged
            nn.Conv2d(128, 32, kernel_size=1, stride=1, padding=0, bias=False),
            nn.GroupNorm(8, 32), nn.GELU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(8, 32), nn.GELU(),
            nn.AdaptiveAvgPool2d((8, 16)),
            nn.Flatten(),
            nn.Linear(32 * 8 * 16, output_dim)
            )

        self.load_backbone()

    def forward(self, x):
        with torch.no_grad():
            x = self.conv(x)
        return x

    def load_backbone(self, path="backbone.pth", keyword="conv"):
        ckpt = torch.load(path, map_location = torch.device("cpu"))
        state_dict = ckpt.get("policy_state_dict", ckpt)

        # Extract all keys containing 'conv'
        conv_keys = {k: v for k, v in state_dict.items() if keyword in k}

        # Determine prefix to strip
        sample_key = next(iter(conv_keys))
        prefix_idx = sample_key.find(keyword) + len(keyword) + 1  # include dot
        conv_dict = {k[prefix_idx:]: v for k, v in conv_keys.items()}

        # Load into self.conv
        missing, unexpected = self.conv.load_state_dict(conv_dict, strict=False)
        print("Backbone loaded:")
        print("Missing keys:", missing)
        print("Unexpected keys:", unexpected)
