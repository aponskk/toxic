import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def load_model(model_path='model_weights.pth'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = AutoModelForSequenceClassification.from_pretrained(
        'SkolkovoInstitute/russian_toxicity_classifier',
        num_labels=2
    )
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    
    model.to(device)
    model.eval()
    
    tokenizer = AutoTokenizer.from_pretrained('SkolkovoInstitute/russian_toxicity_classifier')
    
    return model, tokenizer
