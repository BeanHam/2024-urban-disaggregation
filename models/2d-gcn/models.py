import torch
import torch.nn as nn
import torch.nn.functional as F

## seed
seed = 816
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

class GraphConvolutionLayer(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, adj, bias=True):
        super(GraphConvolutionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.adj = adj
        
        ## linear layer
        self.w = nn.Parameter(torch.rand(self.in_features, self.out_features))
        self.w_bias = nn.Parameter(torch.rand(self.in_features, self.out_features))
        nn.init.xavier_uniform_(self.w.data, gain=0.01)
        nn.init.constant_(self.w_bias.data, 0.01)
        
    def forward(self, att):
        
        ## forward
        x = torch.matmul(self.adj, att)
        x = x*self.w + self.w_bias

        return x

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
        
        ## projection layer
        self.proj = nn.Parameter(torch.rand_like(self.linkage))
        nn.init.xavier_uniform_(self.proj.data, gain=0.01)

        ## learn adjacency
        self.adj_super = nn.Parameter(torch.randn(self.adj_size, self.adj_size))
        nn.init.xavier_uniform_(self.adj_super, gain=0.01)
        
        ## gcc layers
        self.gc1 = GraphConvolutionLayer(self.adj_size, 1, self.adj_super)

    def forward(self, att):
        
        ## projection layer
        x = F.relu((self.proj).T@att)

        ## gcn layers
        x = F.relu(self.gc1(x))
        
        return x