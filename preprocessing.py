from transformers import AutoTokenizer
from datasets import Dataset

def load_and_preprocess_data(train: Dataset, test: Dataset):
    tokenizer = AutoTokenizer.from_pretrained('SkolkovoInstitute/russian_toxicity_classifier')
    
    def tokenize_function(examples):
        return tokenizer(examples['text'], padding='max_length', truncation=True)
    
    tokenized_train = train.map(tokenize_function)
    tokenized_test = test.map(tokenize_function)
    
    return tokenized_train, tokenized_test