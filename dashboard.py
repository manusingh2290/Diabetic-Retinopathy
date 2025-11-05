import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
from PIL import Image
from gradcam import get_img_array, make_gradcam_heatmap, save_and_display_gradcam


st.set_page_config(page_title="Diabetic Retinopathy Classifier", layout="centered")
st.title("👁️ Benchmarking Neuro-Symbolic AI Proposal in Diabetic Retinopathy")


model_path = st.text_input(" Enter model path:", "models/final1.h5")

if os.path.exists(model_path):
    model = load_model(model_path, compile=False)
    st.success(f" Model loaded successfully: {os.path.basename(model_path)}")
else:
    st.warning(" Model path not found! Please check the file path.")
    model = None


CLASS_NAMES = ["0_Healthy", "1_Mild DR", "2_Moderate DR", "3_Proliferate DR", "4_Severe DR"]


uploaded_file = st.file_uploader("📸 Upload a retinal fundus image", type=['png', 'jpg', 'jpeg'])

if uploaded_file and model is not None:
    # Load image
    img = Image.open(uploaded_file).convert('RGB')
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Save temporary file
    tmp_path = "temp_input.jpg"
    img.save(tmp_path)

    # Get input size from model
    input_shape = model.input_shape[1:3]  # e.g., (380, 380)
    st.write(f"Model expects input size: {input_shape}")

    # Preprocess for model
    img = image.load_img(tmp_path, target_size=input_shape)
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0) / 255.0

    # Predict
    preds = model.predict(x)
    class_idx = int(np.argmax(preds[0]))
    confidence = float(np.max(preds[0])) * 100

    # Display results
    st.markdown("### Prediction Results")
    for i, c in enumerate(CLASS_NAMES):
        st.write(f"{c:20s}: **{preds[0][i]*100:.2f}%**")

    st.success(f"Final Prediction: **{CLASS_NAMES[class_idx]} ({confidence:.2f}% confidence)**")


    st.markdown("---")
    st.subheader("Grad-CAM Visualization")

    # Find last conv layer
    last_conv_layer_name = None
    for layer in reversed(model.layers):
        if len(layer.output_shape) == 4:  # convolution layer
            last_conv_layer_name = layer.name
            break

    if last_conv_layer_name is None:
        st.error("No convolutional layer found for Grad-CAM.")
    else:
        st.write(f"Using last conv layer: `{last_conv_layer_name}`")

        # Generate heatmap
        heatmap = make_gradcam_heatmap(x, model, last_conv_layer_name, pred_index=class_idx)
        cam_img = save_and_display_gradcam(
            tmp_path,
            heatmap,
            cam_path="gradcam_output.jpg",
            alpha=0.4,
            size=input_shape
        )

        # Normalize image for display
        cam_img = np.asarray(cam_img)
        if cam_img.dtype != np.uint8:
            cam_img = np.uint8(255 * (cam_img - cam_img.min()) / (cam_img.max() - cam_img.min()))

        st.image(cam_img, caption="Grad-CAM Overlay", use_column_width=True)
        st.info("Bright red regions indicate the most influential areas for this prediction.")
