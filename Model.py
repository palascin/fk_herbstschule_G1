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

        # CNN-Encoder für Bilder
        self.image_encoder = CompactCNN(501)  # liefert 501 Features

        dim_body = 128
        # Voll verbundene Layers in einem Sequential
        self.body = nn.Sequential(
            nn.Linear(state_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128,dim_body ),
            nn.ReLU()
        )

        self.action = nn.Sequential(
            nn.Linear(dim_body, 2*action_dim),
        )

        self.value_function = nn.Sequential(
            nn.Linear(dim_body, num_quantiles),
        )




    def forward(self,x):
        # Erst durch den gemeinsamen Body
        features = self.body(x)

        # Danach Action-Kopf
        action_out = self.action(features)

        # Optional: Value-Kopf (falls du ihn brauchst)
        value_out = self.value_function(features)

        return action_out, value_out

    def forward_critic(self,x):
        # Erst durch den gemeinsamen Body
        features = self.body(x)
        # Implement forward pass for value function 
        # Optional: Value-Kopf (falls du ihn brauchst)
        value_out = self.value_function(features)
        return value_out
