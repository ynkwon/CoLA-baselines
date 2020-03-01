from transformers import BertModel, BertTokenizer
import torch
import torch.nn as nn
import numpy as np


class BertEmbeddings():
    def __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertModel.from_pretrained('bert-base-uncased')
        # self.input_ids = torch.tensor(self.tokenizer.encode(self.sentence, add_special_tokens=True)).unsqueeze(0)  # Batch size 1
        # self.outputs = self.model(self.input_ids)[0]
        # self.shape = np.shape(self.outputs)
        # self.avg = nn.AdaptiveAvgPool2d((self.shape[0], self.shape[2]))
        # self.mp = nn.AdaptiveMaxPool2d((self.shape[0], self.shape[2]))

    def average_embedding(self, input_sentence):
        input_ids = torch.tensor(self.tokenizer.encode(input_sentence, add_special_tokens=True)).unsqueeze(0)  # Batch size 1
        outputs = self.model(input_ids)[0]
        shape = np.shape(outputs)
        avg = nn.AdaptiveAvgPool2d((shape[0], shape[2]))
        return avg(outputs).squeeze()

    def maxpool_embedding(self, input_sentence):
        input_ids = torch.tensor(self.tokenizer.encode(input_sentence, add_special_tokens=True)).unsqueeze(0)  # Batch size 1
        outputs = self.model(input_ids)[0]
        shape = np.shape(outputs)
        mp = nn.AdaptiveMaxPool2d((shape[0], shape[2]))
        return mp(outputs).squeeze()


em = BertEmbeddings()
print(em.average_embedding("hello i'm happy").shape)