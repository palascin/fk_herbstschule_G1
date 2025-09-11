import torch
import torch.nn as nn
from torch.amp import autocast
import math
from Image_Encoding import CompactCNN
import torch.nn as nn

class CustomModel(nn.Module):
    def __init__(self, state_dim, action_dim, num_quantiles=1):
        super(CustomModel, self).__init__()

        self.action_dim = action_dim
        self.state_dim = state_dim
        self.num_quantiles = num_quantiles
        self.feature_norm = nn.LayerNorm(501)
        self.obs_norm = nn.Identity()

        # CNN-Encoder für Bilder
        self.image_encoder = CompactCNN(501)  # liefert 501 Features

        dim_body = 128
        # Voll verbundene Layers in einem Sequential
        self.body = nn.Sequential(
            nn.Linear(state_dim, 512),
            nn.GELU(),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128,dim_body),
            nn.GELU()
        )

        self.action = nn.Sequential(
            nn.LayerNorm(dim_body),
            nn.Linear(dim_body, 128),
            nn.GELU(),
            nn.Linear(128, 128),
            nn.GELU(),
            nn.Linear(128, 2*action_dim),
        )

        self.value_function = nn.Sequential(
            nn.LayerNorm(dim_body),
            nn.Linear(dim_body, 128),
            nn.GELU(),
            nn.Linear(128, 128),
            nn.GELU(),
            nn.Linear(128, num_quantiles),
        )


    def forward(self,x):
        # Erst durch den gemeinsamen Body
        x = torch.cat((self.obs_norm(x[:,:self.state_dim-501]), self.feature_norm(x[:,self.state_dim-501:])), dim = -1)
        features = self.body(x)

        # Danach Action-Kopf
        action_out = self.action(features)

        # Optional: Value-Kopf (falls du ihn brauchst)
        value_out = self.value_function(features)

        return action_out, value_out

    def forward_critic(self,x):
        # Erst durch den gemeinsamen Body
        x = torch.cat((self.obs_norm(x[:,:self.state_dim-501]), self.feature_norm(x[:,self.state_dim-501:])), dim = -1)
        features = self.body(x)
        # Implement forward pass for value function 
        # Optional: Value-Kopf (falls du ihn brauchst)
        value_out = self.value_function(features)
        return value_out
