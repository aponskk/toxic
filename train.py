import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import Dataset
from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments
import evaluate
import numpy as np
import torch
import os
from preprocessing import load_and_preprocess_data
from evaluate import compute_metrics

os.environ["WANDB_DISABLED"] = "true"

def train_model():
    data = pd.read_csv('/kaggle/input/russian-language-toxic-comments/labeled.csv') 
    data.columns = ['text', 'label']
    data['label'] = data['label'].astype(int)
    
    train, test = train_test_split(data, test_size=0.3)
    train = Dataset.from_pandas(train)
    test = Dataset.from_pandas(test)
    
    tokenized_train, tokenized_test = load_and_preprocess_data(train, test)
    
    model = AutoModelForSequenceClassification.from_pretrained(
        'SkolkovoInstitute/russian_toxicity_classifier',
        num_labels=2
    )
    
    training_args = TrainingArguments(
        output_dir='test_trainer_log',
        eval_strategy='epoch',
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=3,
        report_to=None,
        logging_steps=10,
        save_steps=500,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_test,
        compute_metrics=compute_metrics
    )
    
    trainer.train()
    
    model = model.to('cpu')
    torch.save(model.state_dict(), 'model_weights.pth')

if __name__ == "__main__":
    train_model()