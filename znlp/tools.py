import jieba
class JiebaTokenizer:
    def __init__(self):
        pass
    def seg(self,sentence):
        tokens = jieba.cut(sentence)
        return " ".join(list(tokens))
class BaseTokenizer:
    def __init__(self):
        pass
    def seg(self,sentence):
        return sentence

