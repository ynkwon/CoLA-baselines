from transformers import BertModel, BertTokenizer
import torch
import torch.nn as nn
import numpy as np
from acceptability.models import LinearClassifier

class BertEncoder():
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertModel.from_pretrained('bert-base-uncased').to(self.device)

    def average_embedding(self, input_sentence):
        input_ids = input_sentence.to(self.device)
        outputs = self.model(input_ids)[0]
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
        self.classifier = LinearClassifier(hidden_size, 384, dropout)
        self.encoder = BertEncoder()

    def forward(self, x):
        encoding = self.encoder.average_embedding(x)
        output = self.classifier(encoding)
        return output, None


#em = BertEmbeddings()
#print(em.average_embedding("hello i'm happy").shape)
