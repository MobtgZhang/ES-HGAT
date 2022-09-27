import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiMatch(nn.Module):
    """
	Multi match layers for network.
	"""
    def __init__(self,in_size,hidden_size,graind_size,dropout_rate=0.2):
        super(MultiMatch,self).__init__()
        self.graind_size = graind_size
        self.in_size = in_size
        self.dp = dropout_rate
        self.linear = nn.Linear(in_size,hidden_size)
        self.weights = nn.Linear(hidden_size,graind_size)
        self.gelu = nn.GELU()
    def forward(self, inpx):
        """
        inpx: (batch_size,seq_len,in_size)
        inpx: (batch_size,graind_size,hidden_size)
        """
        s0 = self.gelu(self.linear(inpx)) # (batch_size,seq_len,hidden_size)
        sj = self.weights(s0).transpose(2, 1) # (batch_size,graind_size,seq_len)
        rp = F.softmax(sj, 2).bmm(s0) # (batch_size,graind_size,hidden_size)
        rp = F.dropout(rp, p=self.dp, training=self.training)
        return rp
class MatchNetwork(nn.Module):
    """
	This model is for multilayers network.
	"""
    def __init__(self,doc_size, hidden_size,label_size,class_size,dropout_rate=0.2):
        super(MatchNetwork, self).__init__()
        self.doc_size = doc_size
        self.hidden_size = hidden_size
        self.label_size = label_size
        self.class_size = class_size
        self.dp = dropout_rate

        self.label_lin = MultiMatch(doc_size,hidden_size,label_size)
        self.class_lin = MultiMatch(doc_size,hidden_size,class_size)
    def forward(self,x):
        """
        x: (batch_size,num_capsules,doc_size)
        outputs: (batch_size,class_size,label_size)
        """
        f_c = self.label_lin(x) # (batch_size,label_size,hidden_size)
        s_c = self.class_lin(x) # (batch_size,class_size,hidden_size)

        predict = s_c.bmm(f_c.transpose(2, 1)) # (batch_size,class_size,label_size)
        score = F.log_softmax(predict,dim=1)
        return score


class ReadoutLayer(nn.Module):
    def __init__(self,in_dim,hid_dim,class_size,dropout=0.2):
        super(ReadoutLayer,self).__init__()
        self.in_dim = in_dim
        self.hid_dim = hid_dim
        self.dropout = dropout
        self.act = nn.ReLU()
        self.lin_att = nn.Linear(in_dim,hid_dim)
        self.lin_emb = nn.Linear(in_dim,hid_dim)
        self.lin_mlp = nn.Linear(hid_dim,class_size)
    def forward(self,x,mask):
        # soft attention
        att = torch.sigmoid(self.lin_att(x))
        emb = self.act(self.lin_emb(x))    
        # graph summation
        mask = mask.unsqueeze(2).repeat(1,1,self.hid_dim)
        g = mask * att * emb
        N = torch.sum(mask,axis=1)
        M = (mask-1) * 1e9
        M_v,_ = torch.max(g + M, axis=1)
        g = torch.sum(g, axis=1) / N + M_v
        g = F.dropout(g,p=self.dropout)

        # classification
        output =  self.lin_mlp(g)
        logits = F.log_softmax(output,dim=1)
        return logits
class MatchSimpleNet(nn.Module):
    def __init__(self,in_dim,label_size,class_size,dropout=0.2):
        super(MatchSimpleNet,self).__init__()
        self.dropout = dropout
        self.linl = nn.Linear(in_dim,label_size)
        self.linc = nn.Linear(in_dim,class_size)
        self.gelu = nn.GELU()
    def forward(self,inputs):
        sl = self.gelu(self.linl(inputs)) # (b,s,l)
        sc = self.linc(inputs).transpose(2,1) # (b,s,c)
        rp = F.softmax(sc, 2).bmm(sl) # (b,c,s)*(b,s,l)=(b,c,l)
        rp = F.dropout(rp, p=self.dropout, training=self.training)
        return rp
    