"""
main.py
Entrypoint for training and evaluating spoken digit classifier.
"""
import os
import numpy as np
from data import load_fsdd_parquet
from features import extract_mfcc
from model import DigitClassifier
from evaluate import evaluate_model, plot_confusion_matrix

TRAIN_PATH = "free-spoken-digit-dataset/data/train-00000-of-00001.parquet"
TEST_PATH = "free-spoken-digit-dataset/data/test-00000-of-00001.parquet"
MODEL_PATH = "digit_classifier.joblib"

if __name__ == "__main__":
    print("Loading dataset from Parquet files...")
    train_data, test_data = load_fsdd_parquet(TRAIN_PATH, TEST_PATH)
    print(f"Loaded {len(train_data)} train and {len(test_data)} test samples.")

    print("Extracting features...")
    X_train = np.array([extract_mfcc(y, sr=8000, n_fft=512) for y, _ in train_data])
    y_train = np.array([label for _, label in train_data])
    X_test = np.array([extract_mfcc(y, sr=8000, n_fft=512) for y, _ in test_data])
    y_test = np.array([label for _, label in test_data])

    print("Training model...")
    clf = DigitClassifier(model_type='logreg')
    clf.fit(X_train, y_train)
    clf.save(MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")

    print("Evaluating...")
    acc, cm = evaluate_model(clf, X_test, y_test)
    print(f"Accuracy: {acc:.4f}")
    import matplotlib.pyplot as plt
    plot_confusion_matrix(cm, classes=[str(i) for i in range(10)])
    plt.savefig('confusion_matrix.png')

    from sklearn.metrics import classification_report
    print("\nClassification Report:\n")
    print(classification_report(y_test, clf.predict(X_test)))
