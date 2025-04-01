from flask import Flask, render_template, request, jsonify
from flask_cors import CORS  # Import CORS
from transformers import ViTImageProcessor, ViTForImageClassification
from PIL import Image
import torch

app = Flask(__name__)
CORS(app)  

model_dir = r"C:\Users\anant\Desktop\DEEPFAKE MODEL\notebook\vit-base-patch16-224-in21k"  

model = ViTForImageClassification.from_pretrained(model_dir, local_files_only=True)
processor = ViTImageProcessor.from_pretrained(model_dir)

print("Model and processor loaded successfully")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400
        
        file = request.files['image']
        image = Image.open(file.stream).convert("RGB")

        inputs = processor(images=image, return_tensors="pt")

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            predicted_class_idx = logits.argmax(-1).item()

        predicted_label = model.config.id2label[predicted_class_idx]

        return jsonify({
            "prediction": predicted_label,
            "confidence": torch.softmax(logits, dim=-1)[0][predicted_class_idx].item()
        }), 200  # Explicitly return HTTP 200 status
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
