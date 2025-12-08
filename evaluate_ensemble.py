# evaluate_ensemble.py
import tensorflow as tf
import numpy as np
import os
from utils import create_generators
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns

DATASET_PATH = "data/DR"
MODELS_DIR = "models"
MODEL_FILES = ["efficientnetv2.h5", "final1.h5", "best_model.h5"]  # change/order as needed

def load_models(model_files):
    models = []
    for f in model_files:
        path = os.path.join(MODELS_DIR, f)
        if os.path.exists(path):
            models.append(tf.keras.models.load_model(path, compile=False))
        else:
            print("Warning: model not found:", path)
    return models

def ensemble_predict(models, x):
    probs = None
    for m in models:
        p = m.predict(x, verbose=0)
        probs = p if probs is None else probs + p
    probs /= len(models)
    return probs

def main():
    models = load_models(MODEL_FILES)
    if not models:
        raise SystemExit("No models found - check MODEL_FILES list and models folder")

    _, _, test_gen = create_generators(DATASET_PATH, target_size=(380,380), batch_size=8)
    class_names = list(test_gen.class_indices.keys())
    y_true = test_gen.classes

    # Predict on all test images (generator)
    probs_list = []
    # iterate through test_gen batches
    for x_batch, y_batch in test_gen:
        probs_batch = ensemble_predict(models, x_batch)
        probs_list.append(probs_batch)
        if len(probs_list) * test_gen.batch_size >= test_gen.samples:
            break

    y_prob = np.vstack(probs_list)[:test_gen.samples]
    y_pred = np.argmax(y_prob, axis=1)

    print("Classification report:")
    print(classification_report(y_true, y_pred, target_names=class_names))

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_names, yticklabels=class_names, cmap='Blues')
    plt.xlabel("Predicted"); plt.ylabel("True"); plt.title("Confusion matrix")
    plt.show()

    # ROC per class
    n_classes = y_prob.shape[1]
    plt.figure(figsize=(8,6))
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve((y_true==i).astype(int), y_prob[:,i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{class_names[i]} (AUC={roc_auc:.3f})")
    plt.plot([0,1],[0,1],'k--')
    plt.xlabel("FPR"); plt.ylabel("TPR"); plt.legend(); plt.title("ROC curves")
    plt.show()

if __name__ == "__main__":
    main()
