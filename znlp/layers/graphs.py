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
        
        self.base_gcn = GraphConvolution(in_dim,hid_dim,self.dropout)
        self.gcn_mean = GraphGNN(hid_dim,out_dim,self.dropout)
        self.gcn_logstddev = GraphGNN(hid_dim,out_dim,self.dropout)
        self.gelu = nn.GELU()
    def encode(self, x,adj_mat):
        batch_size,seq_len = x.size(0),x.size(1)
        device = x.device
        hidden = self.gelu(self.base_gcn(x,adj_mat))
        mean = torch.tanh(self.gcn_mean(hidden,adj_mat))
        logstd = torch.tanh(self.gcn_logstddev(hidden,adj_mat))
        gaussian_noise = torch.randn(batch_size,seq_len,self.out_dim).to(device)
        # sampled_z = gaussian_noise*(1+logstd) + mean
        sampled_z = gaussian_noise*torch.exp(logstd) + mean
        # print(sampled_z)
        return sampled_z
    def forward(self,x,adj_mat):
        z = self.encode(x,adj_mat)
        A_pred = torch.sigmoid(torch.matmul(z,z.transpose(2,1)))
        return A_pred

class HyperGraphAttentionLayerSparse(nn.Module):
    
    def __init__(self, in_features, out_features, dropout, alpha, transfer, concat=True, bias=False):
        super(HyperGraphAttentionLayerSparse, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.transfer = transfer

        if self.transfer:
            self.weight = nn.Parameter(torch.Tensor(self.in_features, self.out_features))
        else:
            self.register_parameter('weight', None)

        self.weight2 = nn.Parameter(torch.Tensor(self.in_features, self.out_features))
        self.weight3 = nn.Parameter(torch.Tensor(self.out_features, self.out_features))

        if bias:
            self.bias = nn.Parameter(torch.Tensor(self.out_features))
        else:
            self.register_parameter('bias', None)

        self.word_context = nn.Embedding(1, self.out_features)
      
        self.a = nn.Parameter(torch.zeros(size=(2*out_features, 1)))   
        self.a2 = nn.Parameter(torch.zeros(size=(2*out_features, 1)))        
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.out_features)
        if self.weight is not None:
            self.weight.data.uniform_(-stdv, stdv)
        self.weight2.data.uniform_(-stdv, stdv)
        self.weight3.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)
        
        nn.init.uniform_(self.a.data, -stdv, stdv)
        nn.init.uniform_(self.a2.data, -stdv, stdv)
        nn.init.uniform_(self.word_context.weight.data, -stdv, stdv)


    def forward(self, x, adj):
        # x:(b,s,i)
        # w:(i,o)
        x_4att = x.matmul(self.weight2) # (b,s,o)

        if self.transfer:
            x = x.matmul(self.weight) # (b,s,o)
            if self.bias is not None:
                x = x + self.bias        

        N1 = adj.shape[1] #number of edge ,s
        N2 = adj.shape[2] #number of node ,s

        pair = adj.nonzero().t()        

        get = lambda i: x_4att[i][adj[i].nonzero().t()[1]]
        x1 = torch.cat([get(i) for i in torch.arange(x.shape[0]).long()])


        q1 = self.word_context.weight[0:].view(1, -1).repeat(x1.shape[0],1).view(x1.shape[0], self.out_features)
        
        pair_h = torch.cat((q1, x1), dim=-1)
        pair_e = self.leakyrelu(torch.matmul(pair_h, self.a).squeeze()).t()
        assert not torch.isnan(pair_e).any()
        pair_e = F.dropout(pair_e, self.dropout, training=self.training)

        e = torch.sparse_coo_tensor(pair, pair_e, torch.Size([x.shape[0], N1, N2])).to_dense()

        zero_vec = -9e15*torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)


        attention_edge = F.softmax(attention, dim=2)

        edge = torch.matmul(attention_edge, x)
        
        edge = F.dropout(edge, self.dropout, training=self.training)
    
        edge_4att = edge.matmul(self.weight3)

        get = lambda i: edge_4att[i][adj[i].nonzero().t()[0]]
        y1 = torch.cat([get(i) for i in torch.arange(x.shape[0]).long()])

        get = lambda i: x_4att[i][adj[i].nonzero().t()[1]]
        q1 = torch.cat([get(i) for i in torch.arange(x.shape[0]).long()])

        pair_h = torch.cat((q1, y1), dim=-1)
        pair_e = self.leakyrelu(torch.matmul(pair_h, self.a2).squeeze()).t()
        assert not torch.isnan(pair_e).any()
        pair_e = F.dropout(pair_e, self.dropout, training=self.training)

        e = torch.sparse_coo_tensor(pair, pair_e, torch.Size([x.shape[0], N1, N2])).to_dense()

        zero_vec = -9e15*torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)

        attention_node = F.softmax(attention.transpose(1,2), dim=2)

        node = torch.matmul(attention_node, edge)


        if self.concat:
            node = F.elu(node)

        return node

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'

