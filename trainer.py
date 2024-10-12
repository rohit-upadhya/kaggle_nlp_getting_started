import pandas as pd
from dataclasses import dataclass, fields
from typing import Text, List, Dict
from transformers import BertModel, BertTokenizer
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import torch
import json

from encoder import EncoderModel


@dataclass
class NLPTrainer:
    train_file: Text = ""
    test_file: Text = ""
    epochs: int = 5
    learning_rate: float = 3e-4
    device='cpu',
    def __post_init__(self):
        self.train_data = self.load_data(self.train_file)
        self.test_data = self.load_data(self.test_file)
        self.device = torch.device('cuda' if torch.cuda.is_available() and 'cuda' in self.device else 'cpu')
        pass
    
    def load_data(self, file):
        df = pd.read_csv(file)
        df = df.fillna("")
        rows_as_dict_list = df.to_dict(orient='records')
        
        train_dataset = []
        for item in rows_as_dict_list:
            text = f"{item['keyword']} {item['location']} {item['text']}"
            if "test" in file:
                train_dataset.append(
                    {
                        "id": item["id"],
                        "text": text
                    }
                )
            else:
                train_dataset.append(
                    {
                        "id": item["id"],
                        "text": text,
                        "label": item['target']
                    }
                )
        return train_dataset
    
    def tokenizer(self):
        self.tokenizer = BertTokenizer.from_pretrained("FacebookAI/roberta-base")
    
    
    def tokenize_inputs(self, data):
        tokenized_datapoints = []
        
        for item in data:
            inputs = self.tokenizer(
                    item["text"],
                    padding='max_length',
                    truncation=True,
                    max_length=512,
                    return_tensors='pt'
                )
            
            tokenized_datapoints.append(
                {
                    "id": inputs["id"],
                    "input_ids": inputs["input_ids"],
                    "attention_mask": inputs["attention_mask"],
                    "label": item["label"]
                }
            )
        return tokenized_datapoints
        
    def trainer(self):
        self.tokenizer()
        tokenized_data = self.tokenize_inputs(self.train_data)
        train_loader = DataLoader(
            tokenized_data,
            batch_size=32,
            shuffle=True
        )
        model = EncoderModel("FacebookAI/roberta-base").to(self.device)
        optimizer = optim.AdamW(model.parameters(), lr=self.learning_rate)
        criterion = nn.BCEWithLogitsLoss()
        
        for epoch in range(self.epochs):
            model.train()
            total_loss = 0
            for batch in train_loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["label"].to(self.device)
                outputs = model(input_ids, attention_mask)
                
                loss = criterion(outputs, labels)
                total_loss += loss.item()
                loss.backward()
                optimizer.step()
            average_loss = total_loss/len(train_loader)
            print(f"loss in epoch {epoch} : {average_loss}")
            
        self.run_inference(model)
    
    def run_inference(self, model):
        model.eval()
        tokenized_data = self.tokenize_inputs(self.test_data)
        prediction = []
        for item in tokenized_data:
            id = item["id"]
            input_ids = item["input_ids"]
            attention_mask = item["attention_mask"]
            
            outputs = model(input_ids, attention_mask)
            
            prediction_value = torch.sigmoid(outputs)
            predicted_label = 1 if prediction_value >= 0.5 else 0
            
            prediction.append(
                {
                    "id": id,
                    "target": predicted_label
                }
            )
            
            with open("predictions.json", "w+") as file:
                json.dump(prediction, file, indent=4)

if __name__ == "__main__":
    trainer = NLPTrainer(
        train_file="train.csv",
        test_file="test.csv",
        device='cuda:0',
    )