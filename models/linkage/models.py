import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from sklearn.model_selection import train_test_split
from tqdm import tqdm

class GraphSR(nn.Module):
    
    """
    Super-Resolution of graphs
    
    """
    
    def __init__(self, linkage):
        super().__init__()
        
        self.linkage = linkage
                
        ## projection
        self.proj = nn.Parameter(torch.rand_like(self.linkage))
        
    def forward(self, att):
        
        ## down sampling layers
        x = F.relu((self.proj).T@att)
        
        return x