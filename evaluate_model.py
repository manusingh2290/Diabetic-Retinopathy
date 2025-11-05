import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
from utils import create_generators


MODEL_PATH = "models/final1.h5"
DATASET_PATH = "data/DR"

print(f"\n Loading model from: {MODEL_PATH}")
model = tf.keras.models.load_model(MODEL_PATH)


_, _, test_loader = create_generators(DATASET_PATH, target_size=(380, 380), batch_size=8)
class_names = list(test_loader.class_indices.keys())
print(f"\n Detected Classes: {class_names}")



print("\n Evaluating model...")
loss, acc, auc = model.evaluate(test_loader)
print(f"\n Test Accuracy: {acc*100:.2f}%")
print(f" Test Loss: {loss:.4f}")
print(f" Test AUC: {auc:.4f}")


y_true = test_loader.classes
y_pred = np.argmax(model.predict(test_loader), axis=1)


print("\n Classification Report:")
print(classification_report(y_true, y_pred, target_names=class_names))


cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=class_names, yticklabels=class_names)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix — Diabetic Retinopathy Classification")
plt.show()
