import jieba
from janome.tokenizer import Tokenizer
class JiebaTokenizer:
    def __init__(self):
        pass
    def seg(self,sentence):
        tokens = jieba.cut(sentence)
        return " ".join(list(tokens))
class JanomeTokenizer:
    def __init__(self):
        self.tokenizer = Tokenizer()
    def seg(self,sentence):
        content = [item.surface for item in self.tokenizer.tokenize(sentence)]
        return " ".join(content)
class BaseTokenizer:
    def __init__(self):
        pass
    def seg(self,sentence):
        return sentence

