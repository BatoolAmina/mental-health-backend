import torch
import torch.nn as nn
from transformers import RobertaModel, BertModel


class MultiHeadContextAttention(nn.Module):
    def __init__(self, hidden_dim):
        super(MultiHeadContextAttention, self).__init__()
        self.ad_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1)
        )
    def forward(self, lstm_output):
        weights = torch.softmax(self.ad_layer(lstm_output), dim=1)
        context = torch.sum(weights * lstm_output, dim=1)
        return context

class UltraHybridClassifier(nn.Module):
    def __init__(self, num_classes):
        super(UltraHybridClassifier, self).__init__()
        self.roberta = RobertaModel.from_pretrained('roberta-base')
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.lstm = nn.LSTM(input_size=768 + 768, hidden_size=512, num_layers=3,
                            bidirectional=True, batch_first=True, dropout=0.4)
        self.attention = MultiHeadContextAttention(1024)
        self.classifier = nn.Sequential(
            nn.Linear(1024 + 768 + 768, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, r_ids, r_mask, b_ids, b_mask):
        r_out = self.roberta(input_ids=r_ids, attention_mask=r_mask)
        b_out = self.bert(input_ids=b_ids, attention_mask=b_mask)
        combined_seq = torch.cat((r_out.last_hidden_state, b_out.last_hidden_state), dim=2)
        r_cls, b_cls = r_out.pooler_output, b_out.pooler_output
        lstm_out, _ = self.lstm(combined_seq)
        attn_out = self.attention(lstm_out)
        final_features = torch.cat((attn_out, r_cls, b_cls), dim=1)
        return self.classifier(final_features)