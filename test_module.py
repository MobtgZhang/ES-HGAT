import torch
import torch.nn as nn
from znlp.layers import *
def test_MutiAttentionLayer():
    batch_size = 16
    seq_lenq,seq_lenk = 47,37
    dim_q,dim_k,dim_v = 100,80,60
    query = torch.rand(batch_size,seq_lenq,dim_q)
    key = torch.rand(batch_size,seq_lenk,dim_k)
    value = torch.rand(batch_size,seq_lenq,dim_v)
    model  = MutiAttentionLayer(dim_q,dim_k,dim_v)
    outs,atts = model(query,key,value)
    print("MutiAttentionLayer: ",outs.shape,atts.shape)
def test_MatchNetwork():
    batch_size = 16
    seq_len = 47
    doc_dim = 100
    hid_dim = 80
    label_size = 20
    class_size = 4
    dropout = 0.2
    model = MatchNetwork(doc_dim,hid_dim,label_size,class_size,dropout)
    inputs = torch.rand(batch_size,seq_len,doc_dim)
    outputs = model(inputs)
    print("MatchNetwork: ",outputs.shape)
def test_ReadoutLayer():
    batch_size = 16
    seq_len = 47
    in_dim = 100
    hid_dim = 80
    class_size = 20
    dropout = 0.2
    model = ReadoutLayer(in_dim,hid_dim,class_size,dropout)
    mask = torch.randint(0,2,size=(batch_size,seq_len))
    inputs = torch.rand(batch_size,seq_len,in_dim)
    outputs = model(inputs,mask)
    print("ReadoutLayer: ",outputs.shape)
def test_CapsuleLayer():
    batch_size = 16
    num_capsules = 4
    in_dim = 100
    capsules_dim = 60
    num_routing = 5
    seq_len = 47
    model = CapsuleLayer(num_capsules,in_dim,capsules_dim,num_routing)
    inputs = torch.rand(batch_size,seq_len,in_dim)
    outputs = model(inputs)
    print("CapsuleLayer: ",outputs.shape)
def test_MultiCNNLayer():
    batch_size = 16
    seq_len = 47
    hid_dim = 100
    n_filters = 4
    filter_sizes = (5,6,7,8)
    out_dim = 50
    dropout = 0.2
    model = MultiCNNLayer(hid_dim,n_filters,filter_sizes,out_dim,dropout)
    inputs = torch.rand(batch_size,seq_len,hid_dim)
    outputs = model(inputs)
    print("MultiCNNLayer: ",outputs.shape)
def test_SFU():
    batch_size = 16
    seq_len = 47
    in_dim = 100
    fusion_dim = 80
    model = SFU(in_dim,fusion_dim)
    inputs_a = torch.rand(batch_size,seq_len,in_dim)
    inputs_b = torch.rand(batch_size,seq_len,fusion_dim)
    inputs_c = model(inputs_a,inputs_b)
    print("SFU: ",inputs_c.shape)
def test_GraphConvolution():
    in_dim = 100
    out_dim = 40
    batch_size = 16
    seq_len = 47
    model = GraphConvolution(in_dim,out_dim)
    inputs = torch.rand(batch_size,seq_len,in_dim)
    adj_mat = torch.rand(batch_size,seq_len,seq_len)
    outputs = model(inputs,adj_mat)
    print("GraphConvolution: ",outputs.shape)
def test_GraphGNN():
    batch_size = 16
    seq_len = 47
    in_dim = 100
    out_dim = 60
    dropout = 0.2
    model = GraphGNN(in_dim,out_dim,dropout)
    inputs = torch.rand(batch_size,seq_len,in_dim)
    ins_mask = torch.rand(batch_size,seq_len,in_dim)
    adj_mat = torch.rand(batch_size,seq_len,seq_len)
    outputs = model(inputs,adj_mat,ins_mask)
    print("GraphGNN: ",outputs.shape)
def test_VGAE():
    batch_size = 16
    seq_len = 47
    in_dim = 100
    hid_dim = 80
    out_dim = 60
    dropout = 0.2
    model = VGAE(in_dim,hid_dim,out_dim,dropout)
    inputs = torch.rand(batch_size,seq_len,in_dim)
    adj_mat = torch.rand(batch_size,seq_len,seq_len)
    outputs = model(inputs,adj_mat)
    print("VGAE: ",outputs.shape)
def test_HyperGraphAttentionLayerSparse():
    batch_size = 16
    seq_len = 47
    in_dim = 100
    out_dim = 80
    dropout = 0.2
    transfer = True
    alpha = 0.1
    model = HyperGraphAttentionLayerSparse(in_dim,out_dim,dropout,alpha,transfer)
    inputs = torch.rand(batch_size,seq_len,in_dim)
    adj_mat = torch.rand(batch_size,seq_len,seq_len)
    outputs = model(inputs,adj_mat)
    print("HyperGraphAttentionLayerSparse: ",outputs.shape)
def test_MatchSimpleNet():
    batch_size = 16
    seq_len = 47
    in_dim = 100
    label_size = 20
    class_size = 4
    model = MatchSimpleNet(in_dim,label_size,class_size)
    inputs = torch.rand(batch_size,seq_len,in_dim)
    outputs = model(inputs)
    print("MatchSimpleNet: ",outputs.shape)
def test_HyperGraphAttentionLayerSparse():
    in_dim = 300
    out_dim = 200
    dropout = 0.2
    alpha = 0.1
    transfer = True
    batch_size = 16
    seq_len = 47
    model = HyperGraphAttentionLayerSparse(in_dim,out_dim,dropout,alpha,transfer)
    inputs = torch.rand(batch_size,seq_len,in_dim)
    adj_mat = torch.rand(batch_size,seq_len,seq_len)
    outputs = model(inputs,adj_mat)
    print("HyperGraphAttentionLayerSparse: ",outputs.shape)
def main():
    test_MutiAttentionLayer()
    test_MatchNetwork()
    test_ReadoutLayer()
    test_CapsuleLayer()
    test_MultiCNNLayer()
    test_SFU()
    test_GraphConvolution()
    test_GraphGNN()
    test_VGAE()
    test_HyperGraphAttentionLayerSparse()
    test_MatchSimpleNet()
if __name__ == "__main__":
    main()


