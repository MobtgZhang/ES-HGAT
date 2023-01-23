import json
import numpy as np
from tqdm import tqdm

def write_to_json(json_data_list,save_json_file_name):
    with open(save_json_file_name,mode="w",encoding='utf-8') as wfp:
        for line in json_data_list:
            wfp.write(json.dumps(line)+"\n")
def split_content_labels(save_file_name,tokenizer,data_ids,data_contents,data_labels):
    with open(save_file_name,mode="w",encoding="utf-8") as wfp:
        for idx,content,labels in tqdm(zip(data_ids,data_contents,data_labels),desc="Processing data"):
            data_dict = {
                "index":str(idx),
                "content":tokenizer.seg(content),
                "labels":labels
            }
            jline = json.dumps(data_dict,ensure_ascii=False)
            wfp.write(jline+"\n")
def build_embeddings(load_embedding_file,embedding_file,data_dict):
    with open(load_embedding_file,mode="r",encoding='utf-8') as rfp:
        num_words,embedding_dim = rfp.readline().split()
        num_words = int(num_words)
        embedding_dim = int(embedding_dim)
        embeddings = np.random.rand(len(data_dict),embedding_dim)
        num = 0
        for line in tqdm(rfp,desc="processing embedding"):
            num += 1
            line = line.split()
            word = line[0]
            if word in data_dict:
                index = data_dict[word]
                vecs = [float(v) for v in line[-300:]]
                embeddings[index] = vecs
    np.save(embedding_file,embeddings)


