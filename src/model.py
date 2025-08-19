"""
model.py
Defines and trains a lightweight classifier for spoken digit recognition.
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import joblib

class DigitClassifier:
    def __init__(self, model_type='logreg'):
        if model_type == 'logreg':
            self.model = LogisticRegression(max_iter=200)
        elif model_type == 'svm':
            self.model = SVC(kernel='linear', probability=True)
        else:
            raise ValueError('Unsupported model type')
        self.scaler = StandardScaler()

    def fit(self, X, y):
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)

    def predict(self, X):
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

    def predict_proba(self, X):
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)

    def save(self, path):
        joblib.dump({'model': self.model, 'scaler': self.scaler}, path)

    def load(self, path):
        obj = joblib.load(path)
        self.model = obj['model']
        self.scaler = obj['scaler']
