import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import torch
from znlp.models import *
from znlp.data import Dictionary,ContentReviewDataset,batchfy
def draw_mat(graph):
    ax = plt.matshow(graph)
    plt.colorbar(ax.colorbar, fraction=0.025)
    plt.title("matrix X");
    plt.savefig("test.jpg")
    plt.close()
def main():
    result_dir = "./result"
    log_dir = "./log"
    list_length = 5000
    batch_size = 6
    
    for dataset in os.listdir(result_dir):
        re_model_dir = os.path.join(result_dir,dataset)
        log_model_dir = os.path.join(log_dir,dataset,"FGGNN-XLNet")
        re_filename = os.path.join(re_model_dir,"valid-graph.pkl")
        log_filename = None
        for filename in os.listdir(log_model_dir):
             if filename.split(".")[-1]=="npz":
                 log_filename = os.path.join(log_model_dir,filename)
        if log_filename is None:
            continue
        with open(re_filename,mode="rb")as rfp:
            all_datas = pickle.load(rfp)
        dataset_list = np.load(log_filename,allow_pickle=True)
        for graph,data_dict in zip(dataset_list["save_attentions_list"],all_datas):
            content = data_dict["content"]
            id2words = data_dict["id2words"]
            print(graph)
            #draw_mat(graph)
            print(graph.shape,len(content),len(id2words))
            #exit()
if __name__ == "__main__":
    main()

