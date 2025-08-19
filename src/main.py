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

    print("Training Logistic Regression model...")
    clf_logreg = DigitClassifier(model_type='logreg')
    clf_logreg.fit(X_train, y_train)
    clf_logreg.save("digit_classifier_logreg.joblib")
    print("Evaluating Logistic Regression...")
    acc_logreg, cm_logreg = evaluate_model(clf_logreg, X_test, y_test)
    print(f"Logistic Regression Accuracy: {acc_logreg:.4f}")
    import matplotlib.pyplot as plt
    plot_confusion_matrix(cm_logreg, classes=[str(i) for i in range(10)])
    plt.savefig('confusion_matrix_logreg.png')
    from sklearn.metrics import classification_report
    print("\nLogistic Regression Classification Report:\n")
    print(classification_report(y_test, clf_logreg.predict(X_test)))

    print("\nTraining Random Forest model...")
    clf_rf = DigitClassifier(model_type='rf')
    clf_rf.fit(X_train, y_train)
    clf_rf.save("digit_classifier_rf.joblib")
    print("Evaluating Random Forest...")
    acc_rf, cm_rf = evaluate_model(clf_rf, X_test, y_test)
    print(f"Random Forest Accuracy: {acc_rf:.4f}")
    plot_confusion_matrix(cm_rf, classes=[str(i) for i in range(10)])
    plt.savefig('confusion_matrix_rf.png')
    print("\nRandom Forest Classification Report:\n")
    print(classification_report(y_test, clf_rf.predict(X_test)))
