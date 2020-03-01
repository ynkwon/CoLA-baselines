from transformers import BertModel, BertTokenizer
import torch
import torch.nn as nn
import numpy as np
from acceptability.models import LinearClassifier

class BertEncoder():
    def __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertModel.from_pretrained('bert-base-uncased')
        # self.input_ids = torch.tensor(self.tokenizer.encode(self.sentence, add_special_tokens=True)).unsqueeze(0)  # Batch size 1
        # self.outputs = self.model(self.input_ids)[0]
        # self.shape = np.shape(self.outputs)
        # self.avg = nn.AdaptiveAvgPool2d((self.shape[0], self.shape[2]))
        # self.mp = nn.AdaptiveMaxPool2d((self.shape[0], self.shape[2]))

    def average_embedding(self, input_sentence):
        #input_ids = torch.tensor(self.tokenizer.encode(input_sentence, add_special_tokens=True)).unsqueeze(0)  # Batch size 1
        input_ids = input_sentence#.unsqueeze(0)
        #breakpoint()
        outputs = self.model(input_ids)[0]
        #breakpoint()
        #shape = np.shape(outputs)
        #avg = nn.AdaptiveAvgPool1d(shape[2])
        #breakpoint()
        #return avg(outputs)#.squeeze()
        return torch.mean(outputs, dim=1)

    def maxpool_embedding(self, input_sentence):
        input_ids = torch.tensor(self.tokenizer.encode(input_sentence, add_special_tokens=True)).unsqueeze(0)  # Batch size 1
        outputs = self.model(input_ids)[0]
        shape = np.shape(outputs)
        mp = nn.AdaptiveMaxPool2d((shape[2]))
        return mp(outputs).squeeze()

class BertClassifier(nn.Module):
    def __init__(self, hidden_size, encoding_size, dropout=0.5):
        super(BertClassifier, self).__init__()
        #self.classifier = LinearClassifier(hidden_size, encoding_size, dropout)
        self.classifier = LinearClassifier(hidden_size, 384, dropout)
        self.encoder = BertEncoder()

    def forward(self, x):
        encoding = self.encoder.average_embedding(x)
        #breakpoint()
        output = self.classifier(encoding)
        return output, None


#em = BertEmbeddings()
#print(em.average_embedding("hello i'm happy").shape)
