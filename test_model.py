import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd


MODEL_PATH = "models/final1.h5"
DATASET_TRAIN_PATH = "data/DR/train"   # Used to extract class names
IMG_PATH = r"Mild DR/Mild_DR_65.png"   # Single test image path
TEST_FOLDER = r"Proliferate DR"               # 🔁 Change this to test a full folder
EXPORT_CSV = True                      # Save predictions to CSV if True


print(f"\n Loading model from: {MODEL_PATH}")
model = tf.keras.models.load_model(MODEL_PATH)


class_names = sorted([
    d for d in os.listdir(DATASET_TRAIN_PATH)
    if os.path.isdir(os.path.join(DATASET_TRAIN_PATH, d))
])
print(f" Class labels detected: {class_names}")


def predict_single_image(img_path):
    img = image.load_img(img_path, target_size=(380, 380))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = x / 255.0

    pred = model.predict(x)
    class_idx = np.argmax(pred)
    confidence = np.max(pred) * 100
    predicted_class = class_names[class_idx]

    print("\n Prediction Results:")
    for i, c in enumerate(class_names):
        print(f"{c:20s}: {pred[0][i]*100:.2f}%")
    print(f"\n🔍 Final Prediction: {predicted_class} ({confidence:.2f}% confidence)")

    # Show image + bar chart
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(image.load_img(img_path))
    plt.title(f"Predicted: {predicted_class}\n({confidence:.1f}% Confidence)")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    bars = plt.barh(class_names, pred[0] * 100, color="skyblue")
    bars[class_idx].set_color("orange")
    plt.xlabel("Confidence (%)")
    plt.title("Class Probability Distribution")
    plt.xlim(0, 100)
    plt.tight_layout()
    plt.show()


def predict_folder(folder_path):
    results = []
    print(f"\n Predicting all images in folder: {folder_path}")

    for file in os.listdir(folder_path):
        if file.lower().endswith((".png", ".jpg", ".jpeg")):
            img_path = os.path.join(folder_path, file)
            img = image.load_img(img_path, target_size=(380, 380))
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x = x / 255.0

            pred = model.predict(x, verbose=0)
            class_idx = np.argmax(pred)
            confidence = np.max(pred) * 100
            predicted_class = class_names[class_idx]

            results.append({
                "Filename": file,
                "Predicted_Class": predicted_class,
                "Confidence (%)": round(confidence, 2)
            })

    df = pd.DataFrame(results)
    print("\n Prediction Summary:")
    print(df.head())

    if EXPORT_CSV:
        csv_path = os.path.join(folder_path, "predictions.csv")
        df.to_csv(csv_path, index=False)
        print(f"\n Predictions saved to: {csv_path}")

    return df


mode = input("\nChoose mode — (1) Single image  or  (2) Folder batch: ").strip()

if mode == "1":
    predict_single_image(IMG_PATH)
elif mode == "2":
    df = predict_folder(TEST_FOLDER)
else:
    print("Invalid selection. Please choose 1 or 2.")
