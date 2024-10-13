import pandas as pd
from dataclasses import dataclass, fields
from typing import Text, List, Dict
from transformers import BertModel, BertTokenizer, RobertaTokenizer
from transformers import RobertaModel, RobertaTokenizer
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import torch
import json
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import os
from encoder import EncoderModel


@dataclass
class NLPTrainer:
    train_file: Text = ""
    test_file: Text = ""
    epochs: int = 5
    learning_rate: float = 1e-5
    device: str ='cpu'
    
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
            text = f"{item['keyword']}\n{item['text']}"
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
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        # self.tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    
    
    def tokenize_inputs(self, data, test = False):
        tokenized_datapoints = []
        
        for item in data:
            inputs = self.tokenizer(
                    item["text"],
                    padding='max_length',
                    truncation=True,
                    max_length=512,
                    return_tensors='pt'
                )
            input_ids = inputs["input_ids"].squeeze(0)  # Ensure shape is (sequence_length,)
            attention_mask = inputs["attention_mask"].squeeze(0)
            if test:
                tokenized_datapoints.append(
                    {
                        "id": item["id"],
                        "input_ids": input_ids,
                        "attention_mask": attention_mask
                    }
                )
            else:
                tokenized_datapoints.append(
                    {
                        "id": item["id"],
                        "input_ids": input_ids,
                        "attention_mask": attention_mask,
                        "label": item["label"]
                    }
                )
        return tokenized_datapoints
        
    def trainer(self):
        self.tokenizer()
        tokenized_data = self.tokenize_inputs(self.train_data)
        
        train_data, val_data = train_test_split(
                                    tokenized_data, test_size=0.10, random_state=42)
        train_loader = DataLoader(
            train_data,
            batch_size=64,
            shuffle=True
        )
        
        val_loader = DataLoader(
            val_data,
            batch_size=64,
            shuffle=True
        )
        # model = EncoderModel("roberta-base").to(self.device)
        model = EncoderModel("bert-base-uncased").to(self.device)
        optimizer = optim.AdamW(model.parameters(), lr=self.learning_rate)
        criterion = nn.BCEWithLogitsLoss()
        
        for epoch in range(self.epochs):
            total_loss = 0
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.epochs}", leave=False)
            model.train()
            for batch in progress_bar:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["label"].to(self.device).unsqueeze(1).float()
                outputs = model(input_ids, attention_mask)
                
                loss = criterion(outputs, labels)
                total_loss += loss.item()
                loss.backward()
                optimizer.step()
                progress_bar.set_postfix(loss=loss.item())
            average_loss = total_loss/len(train_loader)
            print(f"train loss in epoch {epoch} : {average_loss}")
            test_progress_bar = tqdm(val_loader, desc=f"val for Epoch {epoch+1}/{self.epochs}", leave=False)
            model.eval()
            total_val_loss = 0
            correct_predictions = 0
            total_samples = 0
            
            with torch.no_grad():  # Disable gradient tracking
                test_progress_bar = tqdm(val_loader, desc=f"Validation Epoch {epoch+1}/{self.epochs}", leave=False)
                
                for batch in test_progress_bar:
                    input_ids = batch["input_ids"].to(self.device)
                    attention_mask = batch["attention_mask"].to(self.device)
                    labels = batch["label"].to(self.device).unsqueeze(1).float()
                    
                    # Forward pass (no gradient calculation)
                    outputs = model(input_ids, attention_mask)
                    
                    # Calculate loss
                    loss = criterion(outputs, labels)
                    total_val_loss += loss.item()
                    
                    # Calculate predictions and accumulate accuracy metrics
                    predictions = torch.sigmoid(outputs) > 0.5  # For binary classification
                    correct_predictions += (predictions == labels).sum().item()
                    total_samples += labels.size(0)
            
            average_val_loss = total_val_loss / len(val_loader)
            accuracy = correct_predictions / total_samples
            
            print(f"Validation loss in epoch {epoch+1}: {average_val_loss}")
            print(f"Validation accuracy in epoch {epoch+1}: {accuracy:.4f}")
        
        
        self.run_inference(model)
    
    def run_inference(self, model):
        model.eval()
        tokenized_data = self.tokenize_inputs(self.test_data, test=True)
        prediction = []
        # progress_bar = tqdm(tokenized_data, desc=f"inference", leave=False)
        for item in tqdm(tokenized_data):
            id = item["id"]
            input_ids = item["input_ids"].unsqueeze(0).to(self.device)  # Ensure shape is (sequence_length,)
            attention_mask = item["attention_mask"].unsqueeze(0).to(self.device)
            with torch.no_grad():
                outputs = model(input_ids, attention_mask)
            
            prediction_value = torch.sigmoid(outputs)
            predicted_label = 1 if prediction_value >= 0.5 else 0
            
            prediction.append(
                {
                    "id": id,
                    "target": predicted_label
                }
            )
            df_predictions = pd.DataFrame(prediction)
            
            # Save predictions to CSV
            df_predictions.to_csv("predictions.csv", index=False)
            with open("predictions.json", "w+") as file:
                json.dump(prediction, file, indent=4)



if __name__ == "__main__":
    
    trainer = NLPTrainer(
        train_file="train.csv",
        test_file="test.csv",
        device='cuda:0',
        learning_rate=1e-6,
        epochs=10
    )
    
    trainer.trainer()