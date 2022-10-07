import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

## seed
seed = 816
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

class GraphSR(nn.Module):
    
    """
    Super-Resolution of graphs
    
    """
    
    def __init__(self, linkage):
        super().__init__()
        
        self.linkage = linkage
        self.low_size = linkage.size(0)
        self.super_size = linkage.size(1)
        self.hidden_size = int(np.floor(self.super_size/2))
        
        ## linear layers
        self.lr1 = nn.Linear(self.low_size, self.hidden_size)
        self.lr2 = nn.Linear(self.hidden_size, self.super_size)
        
        ## initialization
        nn.init.xavier_uniform_(self.lr1.weight, gain=0.01)
        nn.init.xavier_uniform_(self.lr2.weight, gain=0.01)
        nn.init.uniform_(self.lr1.bias, b=0.01)
        nn.init.uniform_(self.lr2.bias, b=0.01)
        
    def forward(self, att):
        
        ## projection
        x = F.relu(self.lr1(att))
        x = F.relu(self.lr2(x))
        
        return x