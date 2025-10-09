from flask import Flask, render_template, request
import numpy as np
import torch
from utils import load_pickle
from qnn_model import HybridQuantumClassifier

app = Flask(__name__)

# Load models and transformers
nb = load_pickle("bernoulli.sav")
svc = load_pickle("svc.sav")
qnn_state = load_pickle("qnn_state.pth")
tfidf = load_pickle("tfidf.sav")
pca = load_pickle("pca.sav")
le_verified = load_pickle("le_verified.sav")
le_cat = load_pickle("le_category.sav")

# QNN setup
qnn_input_size = 4  # same as training
qnn = HybridQuantumClassifier(num_qubits=qnn_input_size, num_layers=2)
if qnn_state:
    qnn.load_state_dict(qnn_state)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    # Get form inputs
    text = request.form.get("text", "")
    rating = float(request.form.get("rating", 3))
    verified = str(request.form.get("verified", "Y"))
    category = str(request.form.get("category", "General"))

    # Text vectorization
    if tfidf:
        x_text = tfidf.transform([text]).toarray()
    else:
        x_text = np.zeros((1, 100))

    # Label encodings
    ver = le_verified.transform([verified]) if le_verified else np.array([0])
    cat = le_cat.transform([category]) if le_cat else np.array([0])

    # Combine features
    X = np.hstack([[rating], ver, cat, x_text[0]]).reshape(1, -1)

    # PCA transform
    if pca:
        X = pca.transform(X)

    # Predictions
    probs = {}

    if nb:
        probs["nb"] = float(nb.predict_proba(X)[0, 1])
    if svc:
        probs["svc"] = float(svc.predict_proba(X)[0, 1])

    # QNN prediction
    n_features = X.shape[1]
    if n_features < qnn_input_size:
        Xq = np.hstack([X, np.zeros((1, qnn_input_size - n_features))])
    else:
        Xq = X[:, :qnn_input_size]

    qnn.eval()
    with torch.no_grad():
        q_prob = float(torch.sigmoid(qnn(torch.tensor(Xq, dtype=torch.float32)))[0].item())
    probs["qnn"] = q_prob

    # Combine predictions (optional: average all models)
    avg_prob = np.mean(list(probs.values()))

    # Determine final result
    label = "Fake Review" if avg_prob >= 0.5 else "Real Review"

    return render_template(
        "results.html",
        text=text,
        rating=rating,
        verified=verified,
        category=category,
        result=label,
        avg_prob=round(avg_prob, 3),
        probs=probs
    )


if __name__ == "__main__":
    app.run(debug=True)
