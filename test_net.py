import torch
from znlp.models import *
class Config:
    def __init__(self):
        pass
def test_BiLSTMAtt():
    config = Config()
    config.num_words = 5623
    config.embedding_dim = 300
    config.hidden_dim = 100
    config.out_dim = 50
    config.class_size = 4
    config.label_size = 20
    batch_size = 16
    seq_len = 47
    for single in [True,False]:
        model = BiLSTMAtt(config,single)
        content_words = torch.randint(0,config.num_words,size=(batch_size,seq_len))
        w_mask = torch.randint(0,2,size=(batch_size,seq_len))
        inputs_dict = {
            "content_words":content_words,
            "w_mask":w_mask
        }
        outputs,atts = model(**inputs_dict)
        print("BiLSTMAtt: ",outputs.shape,atts.shape)
def test_CapsuleNet():
    config = Config()
    config.num_words = 5794
    config.embedding_dim = 300
    config.num_capsules = 5
    config.capsules_dim = 100
    config.num_routing = 3
    config.class_size = 4
    config.label_size = 20
    
    batch_size = 16
    seq_len = 47
    for single in [True,False]:
        model = CapsuleNet(config,single)
        content_words = torch.randint(0,config.num_words,size=(batch_size,seq_len))
        w_mask = torch.randint(0,2,size=(batch_size,seq_len))
        inputs_dict = {
            "content_words":content_words,
            "w_mask":w_mask
        }
        outputs = model(**inputs_dict)
        print("CapsuleNet: ",outputs.shape)
def test_CharBiLSTM():
    config = Config()
    config.num_words = 5794
    config.embedding_dim = 300
    config.hidden_dim = 200
    config.class_size = 4
    config.label_size = 20
    
    batch_size = 16
    seq_len = 47
    for single in [True,False]:
        model = CharBiLSTM(config,single)
        content_chars = torch.randint(0,config.num_words,size=(batch_size,seq_len))
        c_mask = torch.randint(0,2,size=(batch_size,seq_len))
        inputs_dict = {
            "content_chars":content_chars,
            "c_mask":c_mask
        }
        outputs = model(**inputs_dict)
        print("CharBiLSTM: ",outputs.shape)
def test_FGGNN():
    config = Config()
    config.num_words = 5794
    config.w_embedding_dim = 300
    config.num_chars = 579
    config.c_embedding_dim = 300
    
    config.num_layers = 3
    config.hid_dim = 200
    config.out_dim = 100
    config.dropout = 0.2
    config.num_capsules = 5
    config.capsules_dim = 60
    config.dim_v = 70
    config.dim_k = 30
    config.class_size = 4
    config.label_size = 20
    
    batch_size = 16
    seq_len = 47
    for single in [True,False]:
        model = FGGNN(config,single)
        words2ids = torch.randint(0,config.num_words,size=(batch_size,seq_len))
        c_mask = torch.randint(0,2,size=(batch_size,seq_len))
        chars2ids = torch.randint(0,config.num_chars,size=(batch_size,seq_len))
        w_mask = torch.randint(0,2,size=(batch_size,seq_len))
        entropy_mat = torch.rand(batch_size,seq_len,seq_len)
        inputs_dict = {
            "chars2ids":chars2ids,
            "c_mask":c_mask,
            "words2ids":words2ids,
            "w_mask":w_mask,
            "entropy_mat":entropy_mat
        }
        outputs,atts = model(**inputs_dict)
        print("FGGNN: ",outputs.shape,atts.shape)
def test_HyperGAT():
    config = Config()
    config.num_words = 5789
    config.embedding_dim = 300
    config.n_hid = 200
    config.out_dim = 50
    config.dropout = 0.2

    batch_size = 16
    seq_len = 47
    config.class_size = 4
    config.label_size = 20
    for single in [True,False]:
        model = HyperGAT(config,single)
        words2ids = torch.randint(0,config.num_words,size=(batch_size,seq_len))
        paris_mat = torch.rand(batch_size,seq_len,seq_len)
        inputs_dict = {
            "words2ids":words2ids,
            "paris_mat":paris_mat
        }
        outputs = model(**inputs_dict)
        print("HyperGAT: ",outputs.shape)
def test_TextCNN():
    config = Config()
    config.num_chars = 5789
    config.embedding_dim = 300
    config.n_filters = 5
    config.filter_sizes = (3,4,5,6)
    config.out_dim = 50
    config.dropout = 0.2

    batch_size = 16
    seq_len = 47
    config.class_size = 4
    config.label_size = 20
    for single in [True,False]:
        model = TextCNN(config,single)
        content_chars = torch.randint(0,config.num_chars,size=(batch_size,seq_len))
        c_mask = torch.randint(0,2,size=(batch_size,seq_len))
        inputs_dict = {
            "content_chars":content_chars,
            "c_mask":c_mask
        }
        outputs = model(**inputs_dict)
        print("TextCNN: ",outputs.shape)
def test_TextGCN():
    config = Config()
    config.num_words = 5789
    config.embedding_dim = 300
    config.nhid_dim = 200
    config.ohid_dim = 100

    batch_size = 16
    seq_len = 47
    config.class_size = 4
    config.label_size = 20
    for single in [True,False]:
        model = TextGCN(config,single)
        words2ids = torch.randint(0,config.num_words,size=(batch_size,seq_len))
        i_mask = torch.randint(0,2,size=(batch_size,seq_len))
        paris_mat = torch.rand(batch_size,seq_len,seq_len)
        inputs_dict = {
            "words2ids":words2ids,
            "i_mask":i_mask,
            "paris_mat":paris_mat
        }
        outputs = model(**inputs_dict)
        print("TextGCN: ",outputs.shape)
def test_Texting():
    config = Config()
    config.num_words = 5789
    config.embedding_dim = 300
    config.num_layers = 5
    config.dropout = 0.2
    config.hidden_dim = 200
    config.out_dim = 100
    
    batch_size = 16
    seq_len = 47
    config.class_size = 4
    config.label_size = 20
    for single in [True,False]:
        model = TextING(config,single)
        words2ids = torch.randint(0,config.num_words,size=(batch_size,seq_len))
        i_mask = torch.randint(0,2,size=(batch_size,seq_len))
        paris_mat = torch.rand(batch_size,seq_len,seq_len)
        inputs_dict = {
            "words2ids":words2ids,
            "i_mask":i_mask,
            "paris_mat":paris_mat
        }
        outputs = model(**inputs_dict)
        print("TextING: ",outputs.shape)
def test_TextRCNN():
    config = Config()
    config.num_words = 5789
    config.embedding_dim = 300
    config.num_layers = 5
    config.r_dropout = 0.2
    config.hidden_dim = 200
    
    batch_size = 16
    seq_len = 47
    config.class_size = 4
    config.label_size = 20
    for single in [True,False]:
        model = TextRCNN(config,single)
        content_words = torch.randint(0,config.num_words,size=(batch_size,seq_len))
        w_mask = torch.randint(0,2,size=(batch_size,seq_len))
        inputs_dict = {
            "content_words":content_words,
            "w_mask":w_mask
        }
        outputs = model(**inputs_dict)
        print("TextRCNN: ",outputs.shape)
def test_HGAT():
    nfeat_list = [300,300,300]
    nhid = 300
    nclass = 4
    dropout = 0.2
    batch_size = 16
    seq_len = 47
    
    for single in [True,False]:
        model = HGAT(nfeat_list,nhid,nclass,dropout)
        inputs = torch.rand(batch_size,seq_len,nfeat_list[0])
        adj_mat = torch.rand(batch_size,seq_len,seq_len)
        outputs = model(inputs,adj_mat)
        print("HGAT: ",outputs.shape)
def main():
    test_BiLSTMAtt()
    test_CapsuleNet()
    test_CharBiLSTM()
    test_FGGNN()
    test_HyperGAT()
    test_TextCNN()
    test_TextGCN()
    test_Texting()
    test_TextRCNN()
    test_HGAT()
if __name__ == "__main__":
    main()


