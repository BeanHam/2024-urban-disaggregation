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
    
    def __init__(self, linkage, super_adj):
        super().__init__()
        
        self.linkage = linkage
        self.super_adj = super_adj
        self.low_size = linkage.size(0)
        self.super_size = linkage.size(1)
        self.hidden_size = int(np.floor(self.super_size/2))
        
        ## linear layers
        self.lr1 = nn.Linear(self.low_size, self.hidden_size)
        self.lr2 = nn.Linear(self.hidden_size, self.super_size)
        self.lr3 = nn.Linear(self.super_size, self.super_size)
        nn.init.xavier_uniform_(self.lr1.weight, gain=0.01)
        nn.init.xavier_uniform_(self.lr2.weight, gain=0.01)
        nn.init.xavier_uniform_(self.lr3.weight, gain=0.5)
        nn.init.uniform_(self.lr1.bias, b=0.01)
        nn.init.uniform_(self.lr2.bias, b=0.01)
        nn.init.uniform_(self.lr3.bias, b=0.01)
        
        ## attention layer
        self.q = nn.Parameter(torch.rand(self.super_size, 1))
        self.k = nn.Parameter(torch.rand(self.super_size, 1))
        self.w = nn.Parameter(torch.rand(self.super_size, 1))
        nn.init.xavier_uniform_(self.q.data, gain=0.01)
        nn.init.xavier_uniform_(self.k.data, gain=0.01)
        nn.init.xavier_uniform_(self.w.data, gain=0.01)
        
    def attention(self, x):
        Wh = self.w*x
        Q = self.q*Wh
        K = self.k*Wh
        e = Q@K.permute(0,2,1)
        zero_vec = -9e15*torch.ones_like(e)
        attention = torch.where(self.super_adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        h = torch.matmul(attention, Wh)
        
        return F.elu(h)
    
    def forward(self, att):
        
        ## projection
        x = F.relu(self.lr1(att))
        x = F.relu(self.lr2(x)).unsqueeze_(-1)
        x = self.attention(x).squeeze_(-1)
        x = F.relu(self.lr3(x))
        
        return x
    