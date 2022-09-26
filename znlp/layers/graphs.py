import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class GraphGNN(nn.Module):
    def __init__(self,in_dim,out_dim,dropout=0.2):
        super(GraphGNN,self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.dropout = dropout
        self.act = nn.ReLU()
        
        self.encode = nn.Linear(in_dim,out_dim)
        self.z0 = nn.Linear(out_dim,out_dim)
        self.z1 = nn.Linear(out_dim,out_dim)
        self.r0 = nn.Linear(out_dim,out_dim)
        self.r1 = nn.Linear(out_dim,out_dim)
        self.h0 = nn.Linear(out_dim,out_dim)
        self.h1 = nn.Linear(out_dim,out_dim)
    def forward(self,x,adj_mat,mask=None):
        # dropout
        x = F.dropout(x,p=self.dropout)
        if mask is not None:
            x = mask * self.act(x)
        x = self.encode(x)
        # message passing
        a = torch.matmul(adj_mat, x)
        
        # update gate        
        z0 = self.z0(a)
        z1 = self.z1(x)
        z = torch.sigmoid(z0 + z1)
        
        # reset gate
        r0 = self.r0(a)
        r1 = self.r1(x)
        r = torch.sigmoid(r0 + r1)

        # update embeddings     
        h0 = self.h0(a)
        h1 = self.h1(r*x)
        h = self.act(h0 + h1)
        
        return h*z + x*(1-z)
    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_dim) + ' -> ' \
               + str(self.out_dim) + ')'
    
class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, inputs, adj_mat):
        support = torch.matmul(inputs, self.weight)
        output = torch.matmul(adj_mat, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'   

class VGAE(nn.Module):
    def __init__(self,in_dim,hid_dim,out_dim,dropout):
        super(VGAE,self).__init__()
        self.in_dim = in_dim
        self.hid_dim = hid_dim
        self.out_dim = out_dim
        self.dropout = dropout
        
        self.base_gcn = GraphGNN(in_dim,hid_dim,dropout)
        self.gcn_mean = GraphGNN(hid_dim,out_dim,dropout)
        self.gcn_logstddev = GraphGNN(hid_dim,out_dim,dropout)
    def encode(self, x,adj_mat):
        batch_size,seq_len = x.size(0),x.size(1)
        device = x.device
        hidden = self.base_gcn(x,adj_mat)
        mean = self.gcn_mean(hidden,adj_mat)
        logstd = self.gcn_logstddev(hidden,adj_mat)
        gaussian_noise = torch.randn(batch_size,seq_len,self.out_dim).to(device)
        sampled_z = gaussian_noise*torch.exp(logstd) + mean
        return sampled_z
    def forward(self,x,adj_mat):
        z = self.encode(x,adj_mat)
        A_pred = torch.sigmoid(torch.matmul(z,z.transpose(2,1)))
        return A_pred
    
class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, inputs, adj_mat):
        support = torch.matmul(inputs, self.weight)
        output = torch.matmul(adj_mat, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')' 

