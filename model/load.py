from transformers import AutoTokenizer, AutoModelForSequenceClassification

REPO_ID = "aponskk/toxicAI" 

def load_model():
    model = AutoModelForSequenceClassification.from_pretrained(REPO_ID)
    tokenizer = AutoTokenizer.from_pretrained(REPO_ID)
    model.eval()
    return model, tokenizer
