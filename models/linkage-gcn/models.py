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
    
class GraphConvolutionLayer(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, adj, bias=True):
        super(GraphConvolutionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.adj = adj
        
        ## linear layer
        self.linear_layer = nn.Linear(in_features, out_features, bias=bias)
        
        ## initiate weights
        nn.init.xavier_uniform_(self.linear_layer.weight, gain=0.01)
        nn.init.constant_(self.linear_layer.bias, 0.01)
        
    def forward(self, att):
        
        ## forward
        output = self.linear_layer(torch.matmul(self.adj, att))
        return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')' 

    
class GraphSR(nn.Module):
    
    """
    Super-Resolution of graphs
    
    """
    
    def __init__(self, linkage):
        super().__init__()
        
        self.linkage = linkage
        self.adj_size = linkage.size(1)
        
        ## projection
        self.proj = nn.Parameter(torch.randn_like(self.linkage))
        nn.init.xavier_uniform_(self.proj, gain=0.01)
        self.adj_super = nn.Parameter(torch.randn(self.adj_size, self.adj_size))
        nn.init.xavier_uniform_(self.adj_super, gain=0.01)
        
        ## gcc layers
        self.gc1 = GraphConvolutionLayer(1,1,self.adj_super)
        self.gc2 = GraphConvolutionLayer(1,1,self.adj_super)
        
    def forward(self, att):
        
        ## projection layer
        x = F.relu((self.proj).T@att)
        
        ## gcn layers
        x = F.relu(self.gc1(x))
        x = F.relu(self.gc2(x))
        
        return x