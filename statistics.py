import json
import os
from pydoc import ispath
import pandas as pd
import matplotlib.pyplot as plt
def main():
    root_dir = "./result"
    # # texts avg. length # classes # train (ratio) # words
    headers = ["Dataset","texts","avg. length","classes","train/dev/test(ratio)","words"]
    all_dataset = []
    for dataset in os.listdir(root_dir):
        files_dir = os.path.join(root_dir,dataset)
        if not os.path.isdir(files_dir):
            continue
        tag_names = ["train","valid","test"]
        tag_numbers = [0,0,0]
        text_length = 0
        text_number = 0
        text_classnum = 0
        words_number = 0
        load_labels_file = os.path.join(files_dir,"names-dict.json")
        with open(load_labels_file,mode="r",encoding="utf-8") as rfp:
            data_dict = json.load(rfp)
            text_classnum = len(data_dict["label2id"])
        for idx,name in enumerate(tag_names):
            load_file_name = os.path.join(files_dir,"%sset.json"%name)
            tp_num = 0
            if not os.path.exists(load_file_name):
                pass
            else:
                with open(load_file_name,mode="r",encoding="utf-8") as rfp:
                    for jline in rfp:
                        data_dict = json.loads(jline)
                        words_number += len(data_dict["content"].split())
                        sentence = data_dict["content"].replace(" ","")
                        text_length += len(sentence)
                        text_number += 1
                        tp_num += 1
            tag_numbers[idx] = tp_num
        text_length = round(text_length/text_number,2)
        tag_numbers = [round(item/text_number,2) for item in tag_numbers]
        tag_num_str = "%0.1f/%0.1f/%0.1f"%(tag_numbers[0]*100,tag_numbers[1]*100,tag_numbers[2]*100)
        val_list = [dataset,text_number,text_length,text_classnum,tag_num_str,words_number]
        all_dataset.append(val_list)
    save_stat_file = os.path.join(root_dir,"statistics.csv")
    all_dataset = pd.DataFrame(all_dataset,columns=headers)
    all_dataset.to_csv(save_stat_file,index=None)
if __name__ == "__main__":
    main()


