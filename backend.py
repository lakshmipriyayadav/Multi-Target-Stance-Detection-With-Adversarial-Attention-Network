from flask import Flask, request, jsonify
import torch
from transformers import BertTokenizer
import torch.nn as nn
from model import BERT_MTAAN  # Import the model from a separate file

app = Flask(__name__)

# Load Model and Tokenizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BERT_MTAAN().to(device)
model.load_state_dict(torch.load("stance_model.pth", map_location=device))
model.eval()

# Prediction Function
def predict_stance(text, target):
    encoding = tokenizer(
        text, target, padding='max_length', truncation=True,
        max_length=128, return_tensors="pt"
    )
    input_ids, attention_mask = encoding["input_ids"].to(device), encoding["attention_mask"].to(device)

    with torch.no_grad():
        logits = model(input_ids, attention_mask)
        probs = torch.softmax(logits, dim=1)
        predicted_class = torch.argmax(probs, dim=1).item()

    stance_map = {0: "Against", 1: "Favor", 2: "Neutral"}
    return stance_map[predicted_class]

# Flask API Route
@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    text, target = data.get("text"), data.get("target")

    if not text or not target:
        return jsonify({"error": "Text and Target are required"}), 400

    stance = predict_stance(text, target)
    return jsonify({"stance": stance})

if __name__ == "__main__":
    app.run(debug=True)
