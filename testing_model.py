import pickle
import time

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from tensorflow.keras.models import load_model


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}")


# Load model and encoder
log("Loading model and label encoder...")
model = load_model("class_model.h5")
with open("label_encoder.pkl", "rb") as f:
    le = pickle.load(f)

log("Loading transformer...")
transformer = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

# Load input CSV
log("Loading products.csv...")
df = pd.read_csv("products2.csv", sep=";", encoding="utf-8", on_bad_lines="skip")
df = df.dropna(subset=['name'])

# Encode texts
log("Encoding product names...")
X_input = transformer.encode(df['name'].tolist(), batch_size=32, show_progress_bar=True)

# Predict
log("Predicting...")
y_pred_probs = model.predict(X_input)
y_pred = np.argmax(y_pred_probs, axis=1)
categories = le.inverse_transform(y_pred)

# Print results
log("Prediction results:")
for name, category in zip(df['name'], categories):
    print(f"{name}  -->  {category}")
