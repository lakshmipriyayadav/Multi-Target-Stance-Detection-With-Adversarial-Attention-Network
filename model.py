import torch
import torch.nn as nn
from transformers import BertModel

class BERT_MTAAN(nn.Module):
    def __init__(self):
        super(BERT_MTAAN, self).__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.attention = nn.Linear(768, 768)
        self.classifier = nn.Linear(768, 3)  # 3 classes: Favor, Against, Neutral
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input_ids, attention_mask):
        bert_output = self.bert(input_ids, attention_mask=attention_mask).last_hidden_state
        attn_weights = torch.tanh(self.attention(bert_output))
        weighted_rep = attn_weights * bert_output
        pooled_output = torch.mean(weighted_rep, dim=1)
        logits = self.classifier(pooled_output)
        return logits
