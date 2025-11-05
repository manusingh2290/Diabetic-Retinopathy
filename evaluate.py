import argparse
import os
import numpy as np
from tensorflow.keras.models import load_model
from utils import create_generators
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import tensorflow as tf

def parse_args():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', required=True)
    p.add_argument('--model_path', required=True)
    p.add_argument('--batch_size', type=int, default=16)
    p.add_argument('--num_classes', type=int, default=2)
    return p.parse_args()

def main():
    args = parse_args()
    model = load_model(args.model_path, compile=False)

    train_loader, val_loader, test_loader = create_generators(args.dataset, target_size=(512,512), batch_size=args.batch_size, num_classes=args.num_classes)
    # Evaluate on test set
    steps = int(np.ceil(test_loader.samples / args.batch_size))
    preds = model.predict(test_loader, steps=steps, verbose=1)
    if args.num_classes == 2:
        y_pred = (preds.flatten() > 0.5).astype(int)
    else:
        y_pred = np.argmax(preds, axis=1)

    y_true = test_loader.classes[:len(y_pred)]

    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))
    print("Classification Report:")
    print(classification_report(y_true, y_pred, digits=4))

    # ROC-AUC for binary
    if args.num_classes == 2:
        try:
            auc = roc_auc_score(y_true, preds.flatten())
            print("ROC-AUC:", auc)
        except Exception as e:
            print("Could not compute ROC-AUC:", e)

if __name__ == '__main__':
    main()
