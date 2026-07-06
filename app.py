from flask import Flask, request, jsonify
import numpy as np
import re

app = Flask(__name__)

# ============================
# Load Model
# ============================

model = np.load("model.npz")

W1 = model["W1"]
b1 = model["b1"]
W2 = model["W2"]
b2 = model["b2"]

# Load Vocabulary
vocab = np.load("vocab.npy", allow_pickle=True).item()


# ============================
# Activation Functions
# ============================

def relu(x):
    return np.maximum(0, x)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# ============================
# Text Cleaning
# ============================

def clean_text(text):

    text = text.lower()

    text = re.sub(r"[^a-zA-Z\s]", "", text)

    text = re.sub(r"\s+", " ", text)

    return text.strip()


# ============================
# Convert Message to Vector
# ============================

def vectorize(message):

    x = np.zeros((1, len(vocab)))

    words = clean_text(message).split()

    for word in words:

        if word in vocab:

            x[0, vocab[word]] = 1

    return x


# ============================
# Forward Propagation
# ============================

def forward(X):

    Z1 = np.dot(X, W1) + b1

    A1 = relu(Z1)

    Z2 = np.dot(A1, W2) + b2

    A2 = sigmoid(Z2)

    return A2


# ============================
# Home Route
# ============================

@app.route("/")
def home():

    return jsonify({
        "message": "Spam Detection API is running."
    })


# ============================
# Prediction Route
# ============================

@app.route("/predict", methods=["POST"])
def predict():

    data = request.get_json()

    if not data:
        return jsonify({"error": "No JSON received"}), 400

    if "message" not in data:
        return jsonify({"error": "Missing 'message' field"}), 400

    message = data["message"]

    X = vectorize(message)

    probability = forward(X)[0][0]

    prediction = "Spam"

    if probability < 0.5:
        prediction = "Ham"

    return jsonify({

        "message": message,

        "prediction": prediction,

        "probability": round(float(probability), 4)

    })


# ============================
# Run Server
# ============================

if __name__ == "__main__":
    app.run(debug=True)