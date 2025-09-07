import torch
import torch.nn as nn
from torch.amp import autocast
import math
from Image_Encoding import CompactCNN

class CustomModel(nn.Module):
    def __init__(self, state_dim, action_dim, num_quantiles=1):
        super(CustomModel, self).__init__()

        # Implement policy architecture

        self.action_dim = action_dim
        self.state_dim = state_dim
        self.image_encoder = CompactCNN(501)

    def forward(self,x): 

        # Implement forward pass including value function 
        
        action_output = 0
        state_values = 0
        return action_output, state_values

    def forward_critic(self,x):

        # Implement forward pass for value function 

        state_values = 0
        return state_values
