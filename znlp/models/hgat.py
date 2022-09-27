import math
from functools import reduce
import torch
import torch.nn as nn
import torch.nn.functional as F

class GraphConvolution(nn.Module):
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

    def forward(self, inputs, adj, global_W = None):
        if len(adj._values()) == 0:
            return torch.zeros(adj.shape[0], self.out_features, device=inputs.device)

        support = torch.spmm(inputs, self.weight)
        if global_W is not None:
            support = torch.spmm(support, global_W)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class SelfAttention_ori(nn.Module):
    def __init__(self, in_features):
        super(SelfAttention, self).__init__()
        self.a = nn.Parameter(torch.FloatTensor(2 * in_features, 1))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.a.size(1))
        self.a.data.uniform_(-stdv, stdv)

    def forward(self, inputs):
        x = inputs.transpose(0, 1)
        self.n = x.size()[0]
        x = torch.cat([x, torch.stack([x] * self.n, dim=0)], dim=2)
        U = torch.matmul(x, self.a).transpose(0, 1)
        U = F.leaky_relu(U)
        weights = F.softmax(U, dim=1)
        outputs = torch.matmul(weights.transpose(1, 2), inputs).squeeze(1)
        return outputs, weights


class SelfAttention(nn.Module):
    def __init__(self, in_features, idx, hidden_dim):
        super(SelfAttention, self).__init__()
        self.idx = idx
        self.linear = torch.nn.Linear(in_features, hidden_dim)
        self.a = nn.Parameter(torch.FloatTensor(2 * hidden_dim, 1))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.a.size(1))
        self.a.data.uniform_(-stdv, stdv)

    def forward(self, inputs):
        # inputs size:  node_num * 3 * in_features
        x = self.linear(inputs).transpose(0, 1)
        self.n = x.size()[0]
        x = torch.cat([x, torch.stack([x[self.idx]] * self.n, dim=0)], dim=2)
        U = torch.matmul(x, self.a).transpose(0, 1)
        U = F.leaky_relu_(U)
        weights = F.softmax(U, dim=1)
        outputs = torch.matmul(weights.transpose(1, 2), inputs).squeeze(1) * 3
        return outputs, weights


class GraphAttentionConvolution(nn.Module):
    def __init__(self, in_features_list, out_features, bias=True, gamma = 0.1):
        super(GraphAttentionConvolution, self).__init__()
        self.ntype = len(in_features_list)
        self.in_features_list = in_features_list
        self.out_features = out_features
        self.weights = nn.ParameterList()
        for i in range(self.ntype):
            cache = nn.Parameter(torch.FloatTensor(in_features_list[i], out_features))
            nn.init.xavier_normal_(cache.data, gain=1.414)
            self.weights.append( cache )
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
            stdv = 1. / math.sqrt(out_features)
            self.bias.data.uniform_(-stdv, stdv)
        else:
            self.register_parameter('bias', None)
        
        self.att_list = nn.ModuleList()
        for i in range(self.ntype):
            self.att_list.append( Attention_NodeLevel(out_features, gamma) )


    def forward(self, inputs_list, adj_list, global_W = None):
        h = []
        for i in range(self.ntype):
            h.append( torch.spmm(inputs_list[i], self.weights[i]) )
        if global_W is not None:
            for i in range(self.ntype):
                h[i] = ( torch.spmm(h[i], global_W) )
        outputs = []
        for t1 in range(self.ntype):
            x_t1 = []
            for t2 in range(self.ntype):
                # adj has no non-zeros
                if len(adj_list[t1][t2]._values()) == 0:
                    x_t1.append( torch.zeros(adj_list[t1][t2].shape[0], self.out_features, device=self.bias.device) )
                    continue

                if self.bias is not None:
                    x_t1.append( self.att_list[t1](h[t1], h[t2], adj_list[t1][t2]) + self.bias )
                else:
                    x_t1.append( self.att_list[t1](h[t1], h[t2], adj_list[t1][t2]) )
            outputs.append(x_t1)
            
        return outputs


        
     

class Attention_NodeLevel(nn.Module):
    def __init__(self, dim_features, gamma = 0.1):
        super(Attention_NodeLevel, self).__init__()

        self.dim_features = dim_features

        self.a1 = nn.Parameter(torch.zeros(size=(dim_features, 1)))
        self.a2 = nn.Parameter(torch.zeros(size=(dim_features, 1)))
        nn.init.xavier_normal_(self.a1.data, gain=1.414)
        nn.init.xavier_normal_(self.a2.data, gain=1.414)        

        self.leakyrelu = nn.LeakyReLU(0.2)
        self.gamma = gamma

    def forward(self, input1, input2, adj):
        h = input1
        g = input2
        N = h.size()[0]
        M = g.size()[0]

        e1 = torch.matmul(h, self.a1).repeat(1, M)
        e2 = torch.matmul(g, self.a2).repeat(1, N).t()
        e = e1 + e2  
        e = self.leakyrelu(e)
        
        zero_vec = -9e15*torch.ones_like(e)
        if 'sparse' in adj.type():
            adj_dense = adj.to_dense()
            attention = torch.where(adj_dense > 0, e, zero_vec)
            attention = F.softmax(attention, dim=1)
            attention = torch.mul(attention, adj_dense.sum(1).repeat(M, 1).t())
            attention = torch.add(attention * self.gamma, adj_dense * (1 - self.gamma))
            del(adj_dense)
        else:
            attention = torch.where(adj > 0, e, zero_vec)
            attention = F.softmax(attention, dim=1)
            attention = torch.mul(attention, adj.sum(1).repeat(M, 1).t())
            attention = torch.add(attention * self.gamma, adj.to_dense() * (1 - self.gamma))
        del(zero_vec)

        h_prime = torch.matmul(attention, g)

        return h_prime
    
    
    
class HGAT(nn.Module):
    def __init__(self, nfeat_list, nhid, nclass, dropout,
                 type_attention=True, node_attention=True,
                 gamma=0.1, sigmoid=False, orphan=True,
                 write_emb=True
                 ):
        super(HGAT, self).__init__()
        self.sigmoid = sigmoid
        self.type_attention = type_attention
        self.node_attention = node_attention

        self.write_emb = write_emb
        if self.write_emb:
            self.emb = None
            self.emb2 = None

        self.nonlinear = F.relu_

        self.nclass = nclass
        self.ntype = len(nfeat_list)

        dim_1st = nhid
        dim_2nd = nclass
        if orphan:
            dim_2nd += self.ntype - 1
        
        self.gc2 = nn.ModuleList()
        if not self.node_attention:
            self.gc1 = nn.ModuleList()
            for t in range(self.ntype):
                self.gc1.append(GraphConvolution(nfeat_list[t], dim_1st, bias=False) )
                self.bias1 = nn.Parameter( torch.FloatTensor(dim_1st) )
                stdv = 1. / math.sqrt(dim_1st)
                self.bias1.data.uniform_(-stdv, stdv)
        else:
            self.gc1 = GraphAttentionConvolution(nfeat_list, dim_1st, gamma=gamma)
        self.gc2.append( GraphConvolution(dim_1st, dim_2nd, bias=True) )

        if self.type_attention:
            self.at1 = nn.ModuleList()
            self.at2 = nn.ModuleList()
            for t in range(self.ntype):
                self.at1.append( SelfAttention(dim_1st, t, 50) )
                self.at2.append( SelfAttention(dim_2nd, t, 50) )
           
        self.dropout = dropout

    def forward(self, x_list, adj_list, adj_all = None):
        x0 = x_list
        
        if not self.node_attention:
            x1 = [None for _ in range(self.ntype)]
            # First Layer
            for t1 in range(self.ntype):
                x_t1 = []
                for t2 in range(self.ntype):
                    idx = t2
                    x_t1.append( self.gc1[idx](x0[t2], adj_list[t1][t2]) + self.bias1 )
                if self.type_attention:
                    x_t1, weights = self.at1[t1]( torch.stack(x_t1, dim=1) )
                else:
                    x_t1 = reduce(torch.add, x_t1)
                    
                x_t1 = self.nonlinear(x_t1)
                x_t1 = F.dropout(x_t1, self.dropout, training=self.training)
                x1[t1] = x_t1
        else:
            x1 = [None for _ in range(self.ntype)]
            x1_in = self.gc1(x0, adj_list)
            for t1 in range(len(x1_in)):
                x_t1 = x1_in[t1]
                if self.type_attention:
                    x_t1, weights = self.at1[t1]( torch.stack(x_t1, dim=1) )
                else:
                    x_t1 = reduce(torch.add, x_t1)
                x_t1 = self.nonlinear(x_t1)
                x_t1 = F.dropout(x_t1, self.dropout, training=self.training)
                x1[t1] = x_t1
        if self.write_emb:
            self.emb = x1[0]        
        
        x2 = [None for _ in range(self.ntype)]
        # Second Layer
        for t1 in range(self.ntype):
            x_t1 = []
            for t2 in range(self.ntype):
                if adj_list[t1][t2] is None:
                    continue
                idx = 0
                x_t1.append( self.gc2[idx](x1[t2], adj_list[t1][t2]) )
            if self.type_attention:
                x_t1, weights = self.at2[t1]( torch.stack(x_t1, dim=1) )
            else:
                x_t1 = reduce(torch.add, x_t1)

            x2[t1] = x_t1
            if self.write_emb and t1 == 0:
                self.emb2 = x2[t1]

            # output layer
            if self.sigmoid:
                x2[t1] = torch.sigmoid(x_t1)
            else:
                x2[t1] = F.log_softmax(x_t1, dim=1)
        return x2
    
