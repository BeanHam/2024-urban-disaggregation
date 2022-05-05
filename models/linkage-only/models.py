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
    
    def __init__(self, 
                 input_dim,
                 hidden_dim,
                 output_dim,
                 linkage):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.linkage = linkage
                
        ## projection
        self.proj = nn.Parameter(torch.rand_like(self.linkage))
        
    def forward(self, att, adj_super):
        
        ## down sampling layers
        x = (self.proj*self.linkage).T@att
        
        return x