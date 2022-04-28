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

def weights_init(init_type='gaussian'):
    
    """
    Function to initialize weights
    
    """
    
    def init_fun(m):
        classname = m.__class__.__name__
        if (classname.find('Conv') == 0 or classname.find(
                'Linear') == 0) and hasattr(m, 'weight'):
            if init_type == 'gaussian':
                nn.init.normal_(m.weight, 0.0, 0.02)
            elif init_type == 'xavier':
                nn.init.xavier_normal_(m.weight, gain=math.sqrt(2))
            elif init_type == 'uniform':
                nn.init.uniform_(m.weight,a=0,b=0.01)
            elif init_type == 'kaiming':
                nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                nn.init.orthogonal_(m.weight, gain=math.sqrt(2))
            elif init_type == 'ones':
                nn.init.ones_(m.weight)                
            elif init_type == 'default':
                pass
            else:
                assert 0, "Unsupported initialization: {}".format(init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias, 0.0)
    return init_fun


class GraphAttentionLayer(nn.Module):
    
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    
    Implementation modified based on repo: https://github.com/Diego999/pyGAT
    
    """
    
    def __init__(self, 
                 in_features, 
                 out_features, 
                 concat=True):
        super(GraphAttentionLayer, self).__init__()
        #self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.concat = concat
        
        self.W = nn.Linear(in_features, out_features)
        self.a1 = nn.Linear(out_features, 1)
        self.a2 = nn.Linear(out_features, 1)
        self.relu = nn.ReLU()
        
        nn.init.uniform_(self.W.weight, a=0,b=0.1)
        nn.init.uniform_(self.a1.weight, a=0,b=0.1)
        nn.init.uniform_(self.a2.weight, a=0,b=0.1)

    def forward(self, h, adj):
        Wh = self.W(h) # h.shape: (N, in_features), Wh.shape: (N, out_features)
        e = self._prepare_attentional_mechanism_input(Wh)
        zero_vec = -9e15*torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        #attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, Wh)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def _prepare_attentional_mechanism_input(self, Wh):
        # Wh.shape (N, out_feature)
        # self.a.shape (2 * out_feature, 1)
        # Wh1&2.shape (N, 1)
        # e.shape (N, N)
        Wh1 = self.a1(Wh)
        Wh2 = self.a2(Wh)
        # broadcast add
        e = Wh1+Wh2.transpose(1,2)
        return self.relu(e)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'

    
class GraphConvolutionLayer(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, 
                 in_features, 
                 out_features,
                 attention=False,                 
                 projection=False,
                 bias=True):
        super(GraphConvolutionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        ## indicator of whether this is a projection layer
        self.projection = projection
        
        ## indicator of using attention mechanism
        self.attention = attention
        if self.attention:
            self.att_layer = GraphAttentionLayer(self.out_features, self.out_features)
            
        ## linear layer
        self.linear_layer = nn.Linear(in_features, out_features, bias=bias)
        self.relu = nn.ReLU()
        
        ## weight initialization
        self.linear_layer.apply(weights_init('uniform'))
        
    def forward(self, att, adj):
        
        ## if this is a projection layer
        if self.projection:
            att = torch.mean(att, dim=1, keepdim=True)
            output = self.relu(self.linear_layer(att)).transpose(1,2)
        else:
            output = self.relu(self.linear_layer(adj@att))
        
        ## if using attention
        if self.attention:
            output = self.att_layer(output, adj)
        
        return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

class GraphSR(nn.Module):
    
    """
    Super-Resolution of graphs
    
    """
    
    def __init__(self, output_node, attention=False):
        super().__init__()
        
        self.attention = attention
        self.output_node = output_node
        
        ## down sampling
        self.gc1 = GraphConvolutionLayer(1,64,attention=self.attention)
        self.gc2 = GraphConvolutionLayer(64,32,attention=self.attention)
        self.gc3 = GraphConvolutionLayer(32,16,attention=self.attention)
        
        ## projection
        self.gc4 = GraphConvolutionLayer(16,self.output_node,projection=True)
        
        ## upsampling
        self.gc5 = GraphConvolutionLayer(1,64,attention=self.attention)
        self.gc6 = GraphConvolutionLayer(64,32,attention=self.attention)
        self.gc7 = GraphConvolutionLayer(32,1,attention=self.attention)

    def forward(self, att, adj_low, adj_super):
        
        ## down sampling layers
        x = self.gc1(att, adj_low)
        x = self.gc2(x, adj_low)
        x = self.gc3(x, adj_low)
        
        ## projection layer
        x = self.gc4(x, adj_low)
        
        ## upsampling layers
        x = self.gc5(x, adj_super)
        x = self.gc6(x, adj_super)
        x = self.gc7(x, adj_super)
        
        return x