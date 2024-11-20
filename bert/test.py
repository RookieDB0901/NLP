from transformers import BertTokenizer, BertForSequenceClassification
import torch

def predict(text):
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=256)
    with torch.no_grad():
        logits = model(**inputs).logits 
    predictions = torch.softmax(logits, dim=-1)
    return {f'{i + 1} stars': float(predictions[0][i]) for i in range(5)}

text = input()
print(predict(text))
