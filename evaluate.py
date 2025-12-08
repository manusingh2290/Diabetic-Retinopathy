# evaluate.py
import tensorflow as tf
import numpy as np
import os
from utils import create_generators
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

MODEL_PATH = "models/final.h5"
DATASET_PATH = "data/DR"

def plot_confusion(cm, class_names):
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_names, yticklabels=class_names, cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()

def plot_roc(y_true, y_prob, class_names):
    n_classes = y_prob.shape[1]
    plt.figure(figsize=(8,6))
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve((y_true==i).astype(int), y_prob[:,i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{class_names[i]} (AUC = {roc_auc:.3f})")
    plt.plot([0,1],[0,1],'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    plt.title('ROC curves (one-vs-rest)')
    plt.show()

def main():
    print("Loading model:", MODEL_PATH)
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)

    _, _, test_gen = create_generators(DATASET_PATH, target_size=(380,380), batch_size=8)
    class_names = list(test_gen.class_indices.keys())
    print("Detected classes:", class_names)

    print("Evaluating...")
    loss, acc, auc_metric = model.evaluate(test_gen, verbose=1)
    print(f"Test Loss: {loss:.4f}  Test Acc: {acc:.4f}  Test AUC: {auc_metric:.4f}")

    # Predictions
    y_true = test_gen.classes
    y_prob = model.predict(test_gen, verbose=1)
    y_pred = np.argmax(y_prob, axis=1)

    print("Classification report:")
    print(classification_report(y_true, y_pred, target_names=class_names))

    cm = confusion_matrix(y_true, y_pred)
    plot_confusion(cm, class_names)
    plot_roc(y_true, y_prob, class_names)

if __name__ == "__main__":
    main()
