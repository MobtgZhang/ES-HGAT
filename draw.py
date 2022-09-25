import os
import json
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
def draw_picture(dataset_name,value_dict,save_pic_dir):
    for tag_name in value_dict:
        for score_name in value_dict[tag_name]:
            for model_name in value_dict[tag_name][score_name]:
                score_name_list = value_dict[tag_name][score_name][model_name]
                print(score_name_list.shape)
                x_list = np.linspace(0,len(score_name_list)-1,len(score_name_list))
                plt.plot(x_list,score_name_list,label=model_name)
            plt.legend(loc="best")
            plt.title("The %s of %s with validation of score %s"%(dataset_name,tag_name,score_name))
            plt.xlabel("The epoches")
            plt.ylabel("The score")
            save_fig_name = os.path.join(save_pic_dir,"%s-%s-%s"%(score_name,tag_name,dataset_name,)+".png")
            plt.savefig(save_fig_name)
            plt.close()
def main():
    log_dir = "./relog"
    save_pic_dir = "./picture"
    if not os.path.exists(save_pic_dir):
        os.mkdir(save_pic_dir)
    for dataset_name in os.listdir(log_dir):
        root_dir = os.path.join(log_dir,dataset_name)
        value_dict = {}
        for model_name in os.listdir(root_dir):
            file_dir = os.path.join(root_dir,model_name)
            data_list = ["train","valid","test"]
            for tag_name in data_list:
                tag_list = [os.path.join(file_dir,filename) for filename in os.listdir(file_dir) 
                          if re.match("^[\s\S]*\-%s\.(csv)$"%tag_name,filename) is not None]
                if len(tag_list)!=0:
                    load_file_name = tag_list[0]
                    if tag_name not in value_dict:
                        value_dict[tag_name] = {}
                        dataset = pd.read_csv(load_file_name)
                        value_dict[tag_name]["f1_score"] = {
                            model_name:dataset["f1_score"].values
                        }
                        value_dict[tag_name]["em_score"] = {
                            model_name:dataset["em_score"].values
                        }
                        value_dict[tag_name]["loss"] = {
                            model_name:dataset["loss"].values
                        }
                    else:
                        value_dict[tag_name]["f1_score"][model_name] = dataset["f1_score"].values
                        value_dict[tag_name]["em_score"][model_name] = dataset["em_score"].values
                        value_dict[tag_name]["loss"][model_name] = dataset["loss"].values
        draw_picture(dataset_name,value_dict,save_pic_dir)
if __name__ == "__main__":
    main()

