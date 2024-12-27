from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle

app = Flask(__name__)

# Enable CORS for all domains on all routes
CORS(app, resources={r"/predict": {"origins": "*"}})

# Load the model, vectorizer, and label encoder
with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)
with open('vectorizer.pkl', 'rb') as vec_file:
    vectorizer = pickle.load(vec_file)
with open('label_encoder.pkl', 'rb') as le_file:
    label_encoder = pickle.load(le_file)

# Print the available categories
categories = list(label_encoder.classes_)
print("Available categories:", categories)

@app.route('/predict', methods=['POST'])
def predict():
    content = request.json
    if not content or 'text' not in content:
        return jsonify({'error': 'Bad Request', 'message': 'No text provided'}), 400

    complaint_text = content['text']
    vectorized_text = vectorizer.transform([complaint_text])
    # Convert sparse matrix to dense
    #dense_vectorized_text = vectorized_text.toarray()
    prediction = model.predict(vectorized_text)
    print("Prediction:", prediction)
    # Map the prediction to the appropriate category
    if prediction == 0:
        category = 'Academic'
    elif prediction == 1:
        category = 'Finance'
    elif prediction == 2:
        category = 'Equipment'
    else:
        category = 'Unknown'  # Handle any unexpected predictions

    return jsonify({'category': category})
    #category = label_encoder.inverse_transform(prediction)[0]
    #return jsonify({'category': category})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')

