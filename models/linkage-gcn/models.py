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

    def __init__(self,
                 in_features, 
                 out_features,
                 bias=True):
        super(GraphConvolutionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
            
        ## linear layer
        self.linear_layer = nn.Linear(in_features, out_features, bias=bias)
        nn.init.uniform_(self.linear_layer.weight, a=0,b=0.1)
        self.relu = nn.ReLU()

    def forward(self, att, adj):
        
        ## if this is a projection layer    
        output = self.relu(self.linear_layer(adj@att))

        return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

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
        
        ## upsampling
        self.gc1 = GraphConvolutionLayer(self.input_dim, self.hidden_dim)
        self.gc2 = GraphConvolutionLayer(self.hidden_dim, 2*self.hidden_dim)
        self.gc3 = GraphConvolutionLayer(2*self.hidden_dim, self.hidden_dim)
        self.gc4 = GraphConvolutionLayer(self.hidden_dim, self.output_dim)

    def forward(self, att, adj_super):
        
        ## projection layer
        x = (self.proj*self.linkage).T@att
        
        ## linear layers
        x = self.gc1(x, adj_super)
        x = self.gc2(x, adj_super)
        x = self.gc3(x, adj_super)
        x = self.gc4(x, adj_super)
        
        return x