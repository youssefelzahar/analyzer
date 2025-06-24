import joblib
from pathlib import Path

model_path = Path("models/model.pkl")
vectorizer_path = Path("models/model_vectorizer.pkl")
label_encoder_path = Path("models/model_label_encoder.pkl")

model = joblib.load(model_path)
vectorizer = joblib.load(vectorizer_path)

label_encoder = None
if label_encoder_path.exists():
    label_encoder = joblib.load(label_encoder_path)

test_descriptions = [
    "Netflix monthly subscription",
    "Uber ride to the airport",
    "Lunch at McDonald's",
    "Salary from company",
    "Electricity bill payment",
    "Grocery shopping at Walmart",
    "Amazon online purchase",
    "Doctor visit for checkup",
    "Apple iTunes charge",
    "Flight ticket from Cairo to Dubai"
]

X_test = vectorizer.transform(test_descriptions)
predictions = model.predict(X_test)

print("\n--- Prediction Results ---")
for desc, label in zip(test_descriptions, predictions):
    if label_encoder:
        decoded_label = label_encoder.inverse_transform([label])[0]
        print(f"{desc} => Predicted Category: {decoded_label}")
    else:
        print(f"{desc} => Predicted Category ID: {label}")
