from dataclasses import dataclass
import pandas as pd
from transformers import BertModel, BertTokenizer, RobertaTokenizer
import json
from tqdm import tqdm
import torch
from encoder import EncoderModel
from typing import Text

@dataclass
class Inference:
    model_name_or_path: str = 'bert-base-uncased'
    test_file: Text = ""
    device: str ='cpu'
    
    def __post_init__(self):
        self.test_data = self.load_data(self.test_file)
        self.device = torch.device('cuda' if torch.cuda.is_available() and 'cuda' in self.device else 'cpu')
    
    def load_data(self, file):
        df = pd.read_csv(file)
        df = df.fillna("")
        rows_as_dict_list = df.to_dict(orient='records')
        
        dataset = []
        for item in rows_as_dict_list:
            text = f"{item['keyword']}\n{item['text']}"
            dataset.append(
                {
                    "id": item["id"],
                    "text": text
                }
            )
        return dataset
    
    def tokenize(self):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
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
            input_ids = inputs["input_ids"].squeeze(0)  # Ensure shape is (sequence_length,)
            attention_mask = inputs["attention_mask"].squeeze(0)
            tokenized_datapoints.append(
                {
                    "id": item["id"],
                    "input_ids": input_ids,
                    "attention_mask": attention_mask
                }
            )
        return tokenized_datapoints
    
        
    def load_model(self):
        self.model = EncoderModel("bert-base-uncased").to(self.device)
    
    def inference(self):
        self.tokenize()
        self.load_model()
        self.model.eval()
        tokenized_data = self.tokenize_inputs(self.test_data)
        prediction = []
        # progress_bar = tqdm(tokenized_data, desc=f"inference", leave=False)
        for item in tqdm(tokenized_data):
            id = item["id"]
            input_ids = item["input_ids"].unsqueeze(0).to(self.device)  # Ensure shape is (sequence_length,)
            attention_mask = item["attention_mask"].unsqueeze(0).to(self.device)
            with torch.no_grad():
                outputs = self.model(input_ids, attention_mask)
            
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

if __name__=="__main__":
    inference = Inference(
        test_file="test.csv",
        device='cuda:1',
    )
    
    inference.inference()