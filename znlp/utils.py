import time
import pandas as pd

class Timer(object):
    """Computes elapsed time."""
    def __init__(self):
        self.running = True
        self.total = 0
        self.start = time.time()
    def reset(self):
        self.running = True
        self.total = 0
        self.start = time.time()
        return self
    def resume(self):
        if not self.running:
            self.running = True
            self.start = time.time()
        return self
    def stop(self):
        if self.running:
            self.running = False
            self.total += time.time() - self.start
        return self
    def time(self):
        if self.running:
            return self.total + time.time() - self.start
        return self.total
class DataSaver(object):
    """save every epoch datas."""
    def __init__(self,save_file_name):
        names = ["f1_score","em_score","loss","time"]
        self.value_list = pd.DataFrame(columns=names)
        self.save_file_name = save_file_name
    def add_values(self,f1_score,em_score,loss,time):
        data = {"f1_score":f1_score,
                "em_score":em_score,
                "loss":loss,
                "time":time}
        idx = len(self.value_list)
        self.value_list.loc[idx] = data
        self.value_list.to_csv(self.save_file_name,index=None)



