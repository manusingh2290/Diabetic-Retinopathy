# 👁️ Diabetic Retinopathy Detection (Deep Learning + Grad-CAM)

A Streamlit-based web application that detects diabetic retinopathy (DR) stages from retinal fundus images using deep learning models.  
The app supports **automatic model download**, **Grad-CAM visualization**, and **PDF report generation** for clinical use.

---

# 📌 Overview

Diabetic Retinopathy (DR) is a leading cause of blindness among diabetic patients.  
Early detection is essential, and deep learning models can help clinicians analyze retinal fundus images quickly and accurately.

This project provides:

✔ Automatic DR Stage Classification  
✔ Grad-CAM heatmaps for model explainability  
✔ Multi-image batch processing  
✔ PDF reports for patients  
✔ Streamlit UI for easy usage  
✔ Models automatically downloaded from Google Drive  

---

# ⭐ Features

- ✔ Upload multiple retina images  
- ✔ Automatically download trained DL models from Google Drive  
- ✔ Predict DR severity: **Healthy → Severe DR**  
- ✔ Grad-CAM heatmaps for medical interpretability  
- ✔ Generate PDF reports with patient details  
- ✔ Clean, responsive UI built using Streamlit  

---

# 🧠 Model Architecture

This project uses transfer learning with:

- EfficientNet  
- ResNet  
- Custom CNN models  

Training pipeline included:

- Image normalization  
- Data augmentation  
- Class imbalance handling  
- Softmax classification  

### Models Included  
- best_model.h5
- final_model.h5
- final1.h5

  
The app automatically downloads them during execution.

---

# 📂 Project Structure
- diabetic-retinopathy/
 - │
 - ├── dashboard.py
 - ├── gradcam.py
 - ├── requirements.txt
 - ├── runtime.txt
 - ├── README.md
 - └── models/ # auto-created when model downloads


---

# 🔧 Installation & Setup (Local)

### 1️⃣ Clone Repository
```bash
git clone https://github.com/manusingh2290/Diabetic-Retinopathy
cd Diabetic-Retinopathy
```
###2️⃣ Create Environment
```bash
python -m venv venv
```
###3️⃣ Activate (Windows)
```bash
venv\Scripts\activate
```
###4️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```
###5️⃣ Run the App
```bash
streamlit run dashboard.py
```

---

# 🤖 Model Auto-Download

The app automatically downloads .h5 models from Google Drive via gdown.

- 🔹 best_model.h5: https://drive.google.com/file/d/1yKtQdHNaVFIq6g-j6Mn0RlqYTS5RjXre/view?usp=sharing

- 🔹 final_model.h5: https://drive.google.com/file/d/1T2cgXyayzJ4eskeSX2oE4_QLyeNU8BjM/view?usp=sharing

- 🔹 final1.h5: https://drive.google.com/file/d/1zV42a1RjybxQ3dnAmT6alizpwOED6PmB/view?usp=sharing

- Stored in:

   - models/
     - ├── best_model.h5
     - ├── final_model.h5
     - └── final1.h5

---

# 🔥 Why Grad-CAM?
Grad-CAM helps clinicians understand:

- Where the model is focusing

- Whether the prediction corresponds to real pathology

- How confident the model is

- If the network is attending to medical features or noise

This improves interpretability and trust in AI-driven diagnosis.

---

# 🖼 Grad-CAM Visualization
For each image, the app:

1. Predicts DR class

2. Generates Grad-CAM heatmap

3. Displays original + heatmap side-by-side

4. Includes the heatmap in the final PDF report

---

# 📄 PDF Report Generation
The PDF includes:

- Patient name, age, gender
- Notes (optional)
- Prediction table
- Confidence scores
- Side-by-side image + Grad-CAM
- Doctor signature field

Fully downloadable with 1 click.

---

# 📦 Requirements
Here is the exact requirements.txt:
``` bash
tensorflow-cpu==2.13.0
numpy==1.24.3
pandas
opencv-python-headless
matplotlib
scikit-learn
tqdm
streamlit
pillow
scipy
gdown
fpdf
```

---

# 🌐 Deployment (Streamlit Cloud)
Required files:
``` bash
requirements.txt
runtime.txt
dashboard.py
```
runtime.txt:
```bash
python-3.10
```
Steps:

1. Push repo to GitHub

2. Open https://share.streamlit.io

3. Select repository

4. Set main file →
```
dashboard.py
```

5. Deploy 🚀

Streamlit Cloud installs everything automatically.

---

# 🧪 DR Classification Labels
```
0 — Healthy
1 — Mild DR
2 — Moderate DR
3 — Proliferate DR
4 — Severe DR
```

---

# 🤝 Contributing
Pull requests are welcome!

For major changes, open an issue first to discuss improvements.

---

# 👤 Author

Manendra Singh

GitHub → https://github.com/manusingh2290

---

# ⭐ Support

If you like this project, please ⭐ star the repository!
Your support helps improve and grow this work.
