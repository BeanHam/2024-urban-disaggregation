import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

## seed
seed = 100
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic=True

# ---------------------
# weights initialization
# ---------------------
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        torch.nn.init.normal_(m.weight, 0.0, 0.01)
        torch.nn.init.constant_(m.bias, 0.01)
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight, 0.0, 0.01)
    elif classname.find('BatchNorm') != -1:
        torch.nn.init.normal_(m.weight, 0.0, 0.01)
        torch.nn.init.zeros_(m.bias)

# ---------------------
# Disaggregation Model
# ---------------------
class DisAgg(nn.Module):
    
    """
    Super-Resolution of graphs
    
    """
    
    def __init__(self, low_size, high_size, hidden_sizes):
        super().__init__()
        
        self.low_size = low_size
        self.high_size = high_size
        self.hidden_size_0 = hidden_sizes[0]
        self.hidden_size_1 = hidden_sizes[1]
        
        ## linear layers
        self.lr0 = nn.Linear(self.low_size, self.low_size)
        self.lr1 = nn.Linear(self.low_size, self.hidden_size_0)
        self.lr2 = nn.Linear(self.hidden_size_0, self.hidden_size_1)
        self.lr3 = nn.Linear(self.hidden_size_1, self.high_size)
        self.lr4 = nn.Linear(self.high_size, self.high_size)
        
    def forward(self, att):
        
        ## projection
        puma = self.lr0(att)
        nta = self.lr1(F.relu(puma))
        tract = self.lr2(F.relu(nta))
        block = self.lr3(F.relu(tract))
        block = self.lr4(F.relu(block))
        
        return puma, nta, tract, block