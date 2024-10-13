import torch.nn as nn
import torch
import torch.nn.functional as F
from transformers import BertModel
from transformers import RobertaModel

class EncoderModel(nn.Module):
    def __init__(self, model_name_or_path):
        super(EncoderModel, self).__init__()
        self.model = BertModel.from_pretrained(model_name_or_path)
        # self.model = RobertaModel.from_pretrained(model_name_or_path)
        self.temperature = 0.1  # Temperature scaling parameter
        self.FFNetwork = nn.Sequential(
            nn.Linear(self.model.config.hidden_size, 128),
            nn.ReLU(),
            # nn.Linear(256, 128),
            # nn.ReLU(),
            nn.Linear(128, 1)
        )
        
        self.sigmoid = nn.Sigmoid()
    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        cls_outputs = outputs.pooler_output
        
        logits = self.FFNetwork(cls_outputs)
        
        # prob = self.sigmoid(logits)
        
        return logits
        
        