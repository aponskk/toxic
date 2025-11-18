import torch
from .load import load_model

class ToxicityClassifier:
    def __init__(self, model_path='model_weights.pth'): 
        self.model, self.tokenizer = load_model(model_path)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
    
    def predict(self, texts):
        if isinstance(texts, str):
            texts = [texts]
            single_input = True
        else:
            single_input = False
        
        inputs = self.tokenizer(
            texts, 
            padding=True, 
            truncation=True, 
            max_length=512, 
            return_tensors="pt"
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1)
            predictions = torch.argmax(probs, dim=1)
        
        results = []
        for i, (pred, prob) in enumerate(zip(predictions, probs)):
            label = "toxic" if pred.item() == 1 else "neutral"
            score = prob[pred.item()].item()
            
            result = {
                "label": label,
                "score": round(score, 4)
            }
            results.append(result)
        
        return results[0] if single_input else results

classifier = ToxicityClassifier()

def predict_toxicity(texts):
    return classifier.predict(texts)
