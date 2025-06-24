import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from config import TRANSACTIONS_CLEANED_PATH
from sklearn.metrics import classification_report


class Predict:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = None
        self.model = None
        self.vectorizer = None

    def load_data(self):
        self.data = pd.read_csv(self.file_path)

        return self.data

    def vectorize_data(self):
        self.vectorizer = TfidfVectorizer()
        X = self.vectorizer.fit_transform(self.data['Description'])

        y = self.data['Category_encoded']

        return X, y

    def split_data(self, X, y):
        return train_test_split(X, y, test_size=0.2, random_state=42)

    def train_model(self, X_train, y_train):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model.fit(X_train, y_train)
       
        return self.model

    def save_model(self, model_path):
        joblib.dump(self.model, model_path)
        joblib.dump(self.vectorizer, model_path.replace('.pkl', '_vectorizer.pkl'))
        print(f"✅ Model and vectorizer saved to {model_path}")


if __name__ == "__main__":
    predictor = Predict(TRANSACTIONS_CLEANED_PATH)

    predictor.load_data()

    X, y = predictor.vectorize_data()

    X_train, X_test, y_train, y_test = predictor.split_data(X, y)
    model = predictor.train_model(X_train, y_train)
    score = model.score(X_test, y_test)
    print(f"Model accuracy: {score:.2f}")

    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))

    

    predictor.save_model("models/model.pkl")
