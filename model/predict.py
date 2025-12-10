import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import gdown

MODEL_PATH = "model_weights.pth"
GOOGLE_DRIVE_ID = "1jul3j9ephqUAWJyzQeZWXCZ-gIiVWXps" 

if not os.path.exists(MODEL_PATH):
    url = f"https://drive.google.com/uc?id={GOOGLE_DRIVE_ID}"
    print("Модель не найдена, скачиваем с Google Drive...")
    gdown.download(url, MODEL_PATH, quiet=False)
    print("Модель загружена.")

def load_model(model_path=MODEL_PATH):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = AutoModelForSequenceClassification.from_pretrained(
        "SkolkovoInstitute/russian_toxicity_classifier", num_labels=2
    )
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(
        "SkolkovoInstitute/russian_toxicity_classifier"
    )

    return model, tokenizer


class ToxicityClassifier:
    def __init__(self, model_path=MODEL_PATH):
        self.model, self.tokenizer = load_model(model_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def predict(self, texts):
        if isinstance(texts, str):
            texts = [texts]
            single_input = True
        else:
            single_input = False

        inputs = self.tokenizer(
            texts, padding=True, truncation=True, max_length=512, return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1)
            predictions = torch.argmax(probs, dim=1)

        results = []
        for pred, prob in zip(predictions, probs):
            label = "toxic" if pred.item() == 1 else "neutral"
            score = prob[pred.item()].item()
            results.append({"label": label, "score": round(score, 4)})

        return results[0] if single_input else results


classifier = ToxicityClassifier()

def predict_toxicity(texts):
    return classifier.predict(texts)
