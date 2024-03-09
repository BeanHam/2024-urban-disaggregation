import os
import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
os.environ["CUBLAS_WORKSPACE_CONFIG"]=":16:8"

# ---------------------------
# seeding for reproducibility
# ---------------------------
seed = 100
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic=True
torch.use_deterministic_algorithms(True)
        
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
# min_max 3D normalization
# ---------------------        
def min_max_norm(e, exp_linkage):
    pos_vec = 9e15*torch.ones_like(e)
    neg_vec = -9e15*torch.ones_like(e)
    pos_e = torch.where(exp_linkage>0, e, pos_vec)
    neg_e = torch.where(exp_linkage>0, e, neg_vec)    
    min_values = torch.min(pos_e, dim=-1).values.unsqueeze_(-1)
    max_values = torch.max(neg_e, dim=-1).values.unsqueeze_(-1)
    denomenator = max_values-min_values
    ## when min and max values are the same
    denomenator = torch.where(denomenator==0, 1e-6, denomenator)
    e = (e-min_values)/denomenator
    e = torch.where(exp_linkage>0, e, neg_vec)
    return e

# ---------------------
# attention layer
# ---------------------        
class attention_layer(nn.Module):
    
    """
    Super-Resolution of graphs
    
    """
    
    def __init__(self, low, high, linkage):
        super().__init__()
        
        ## linear layers
        self.dk = high
        self.wk = nn.Parameter(torch.randn(low,high))
        self.wq = nn.Parameter(torch.randn(low,low))
        self.wv = nn.Parameter(torch.randn(low,low))
        self.linkage = linkage
        self.reset_parameters()

    def reset_parameters(self):
        std = 1.0 / np.sqrt(self.dk)
        for w in self.parameters():
            w.data.uniform_(-std, std)
            
    def forward(self, att):
        
        ## projection
        Q = torch.matmul(att, self.wq)
        K = torch.matmul(att, self.wk)
        V = torch.matmul(att, self.wv)        
        e = torch.matmul(Q.permute(0,2,1), K)/np.sqrt(self.dk)
        exp_linkage = self.linkage.clone().unsqueeze_(0).repeat(att.size(0),1,1)
        e = min_max_norm(e, exp_linkage)        
        attention = F.softmax(e, dim=-1)
        x = torch.matmul(V, attention)
        
        return x

# -----------------------
# Disaggreation Attention
# -----------------------
class DisAttention(nn.Module):
    
    """
    multi-head attention layer
    """
    
    def __init__(self, low, high, nheads, linkage):
        
        super(DisAttention, self).__init__()
        self.attentions = [attention_layer(low, high, linkage) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)
        self.out_layer = nn.Linear(high*nheads, high, bias=True)

    def forward(self, x):
        x = torch.cat([att(x) for att in self.attentions], dim=-1)
        x = self.out_layer(x)
        return x
    
# ---------------------
# GRU cell
# ---------------------       
class GRUCell(nn.Module):

    """
    An implementation of GRUCell.

    """

    def __init__(self, input_size, hidden_size, nheads, linkage):
        super(GRUCell, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.x2r = DisAttention(input_size, hidden_size, nheads, linkage)
        self.x2z = DisAttention(input_size, hidden_size, nheads, linkage)
        self.x2n = DisAttention(input_size, hidden_size, nheads, linkage)        
        self.h2r = nn.Linear(hidden_size, hidden_size, bias=True)
        self.h2z = nn.Linear(hidden_size, hidden_size, bias=True)
        self.h2n = nn.Linear(hidden_size, hidden_size, bias=True)
        self.reset_parameters()

    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.hidden_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)
    
    def forward(self, x, h):
        
        ## reset gate
        x2r = self.x2r(x)
        h2r = self.h2r(h)
        resetgate = torch.sigmoid(x2r + h2r)
        
        ## update gate
        x2z = self.x2z(x)
        h2z = self.h2z(h)
        updatetgate = torch.sigmoid(x2z + h2z)
        
        ## new gate
        x2n = self.x2n(x)
        h2n = self.h2n(h*resetgate)
        newgate = torch.tanh(x2n + h2n)
        
        ## new hidden state        
        hy = updatetgate*h + (1-updatetgate)*newgate
        
        return hy

# ---------------------
# GRU model
# ---------------------      
class GRU(nn.Module):
    def __init__(self, input_dim, hidden_dim, nheads, linkage):
        super(GRU, self).__init__()
        
        # Hidden dimensions
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.gru_cell = GRUCell(input_dim, hidden_dim, nheads, linkage)            
             
    def forward(self, x):
        
        hn = Variable(torch.zeros(x.size(0), 1, self.hidden_dim).cuda())
        out = []
        for seq in range(x.size(1)):
            hn = self.gru_cell(x[:,seq,:].clone().unsqueeze_(1), hn) 
            out.append(hn)
        out = torch.concat(out,dim=1)
        return out 
    
# ---------------------
# PUMA TO NTA
# ---------------------
class puma_nta(nn.Module):
    
    """
    Super-Resolution of graphs
    
    """
    
    def __init__(self, hidden_dims, nheads, linkages, rec):
        super().__init__()
        
        self.hidden_dims = hidden_dims
        self.rec = rec
        self.puma_nta = linkages[0]
        
        ## gru layers
        self.gru0 = GRU(self.hidden_dims[0], self.hidden_dims[1], nheads, self.puma_nta)
        self.gru1 = nn.GRU(self.hidden_dims[1], self.hidden_dims[1], batch_first=True)

    def forward(self, att):
        
        ## projection
        nta = self.gru0(att)
        #nta,_ = self.gru1(F.relu(nta))

        ## reconstruction
        if self.rec == 'no':
            rec_pumas = []
        else: 
            rec_pumas = [
                torch.matmul(nta, self.puma_nta.T)
            ]
        
        return {
            'nta': nta,
            'rec_pumas':rec_pumas
        }

# ---------------------
# PUMA TO TRACT
# ---------------------    
class puma_tract(nn.Module):
    
    """
    Super-Resolution of graphs
    
    """
    
    def __init__(self, hidden_dims, nheads, linkages, rec):
        super().__init__()
        
        self.hidden_dims = hidden_dims
        self.rec = rec        
        self.puma_nta = linkages[0]
        self.puma_tract = linkages[1]
        self.nta_tract = linkages[2]  
        
        ## gru layers
        self.gru0 = GRU(self.hidden_dims[0], self.hidden_dims[2], nheads, self.puma_tract)
        self.gru1 = nn.GRU(self.hidden_dims[2], self.hidden_dims[2], batch_first=True)

    def forward(self, att):
        
        ## projection
        tract = self.gru0(att)
        #tract,_ = self.gru1(F.relu(tract))

        ## reconstruction
        if self.rec == 'no': 
            rec_pumas = []
            rec_ntas = []
        elif self.rec == 'bridge':
            rec_pumas = [
                torch.matmul(nta, self.puma_nta.T)
            ]            
            rec_ntas = [
                torch.matmul(tract, self.nta_tract.T)
            ]
        elif self.rec == 'bottomup':
            rec_pumas = [
                torch.matmul(tract, self.puma_tract.T)
            ]            
            rec_ntas = [
                torch.matmul(tract, self.nta_tract.T)
            ]
        else:
            rec_pumas = [
                torch.matmul(nta, self.puma_nta.T),
                torch.matmul(tract, self.puma_tract.T)
            ]         
            rec_ntas = [
                torch.matmul(tract, self.nta_tract.T)
            ]
            
        return {
            'tract': tract,
            'rec_pumas':rec_pumas,
            'rec_ntas': rec_ntas
        }

# ---------------------
# PUMA TO BLOCK
# ---------------------    
class puma_block(nn.Module):
    
    """
    Super-Resolution of graphs
    
    """
    
    def __init__(self, hidden_dims, nheads, linkages, rec):
        super().__init__()
        
        self.hidden_dims = hidden_dims
        self.rec = rec        
        self.puma_nta = linkages[0]
        self.puma_tract = linkages[1]
        self.puma_block = linkages[2] 
        self.nta_tract = linkages[3]
        self.nta_block = linkages[4]
        self.tract_block = linkages[5]  
        
        ## gru layers
        self.gru0 = GRU(self.hidden_dims[0], self.hidden_dims[3], nheads, self.puma_block)
        self.gru1 = nn.GRU(self.hidden_dims[3], self.hidden_dims[3], batch_first=True)

    def forward(self, att):
        
        ## projection
        block = self.gru0(att)
        #block,_ = self.gru1(F.relu(block))

        ## reconstruction
        if self.rec == 'no': 
            rec_pumas = []
            rec_ntas = []
            rec_tracts = []
        elif self.rec == 'bridge':
            rec_pumas = [
                torch.matmul(nta, self.puma_nta.T)
            ]        
            rec_ntas = [
                torch.matmul(tract, self.nta_tract.T)
            ]        
            rec_tracts = [
                torch.matmul(block, self.tract_block.T)
            ]
        elif self.rec == 'bottomup':
            rec_pumas = [
                torch.matmul(block, self.puma_block.T)
            ]        
            rec_ntas = [
                torch.matmul(block, self.nta_block.T)
            ]        
            rec_tracts = [
                torch.matmul(block, self.tract_block.T)
            ]
        else:
            rec_pumas = [
                torch.matmul(nta, self.puma_nta.T),
                torch.matmul(tract, self.puma_tract.T),
                torch.matmul(block, self.puma_block.T)
            ]            
            rec_ntas = [
                torch.matmul(tract, self.nta_tract.T),
                torch.matmul(block, self.nta_block.T)
            ]            
            rec_tracts = [
                torch.matmul(block, self.tract_block.T)
            ]
        
        return {
            'block': block,
            'rec_pumas':rec_pumas,
            'rec_ntas': rec_ntas,
            'rec_tracts': rec_tracts
        }    