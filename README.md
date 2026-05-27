# 🔍 Pehchan AI — Real-Time Age & Gender Prediction System

> A deep learning system that detects faces from a live webcam feed and predicts **age group** and **gender** in real time using a custom-trained Convolutional Neural Network.

![Python](https://img.shields.io/badge/Python-3.10-blue?style=flat-square&logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13-orange?style=flat-square&logo=tensorflow)
![OpenCV](https://img.shields.io/badge/OpenCV-4.13-green?style=flat-square&logo=opencv)
![Accuracy](https://img.shields.io/badge/Gender%20Accuracy-90%25-brightgreen?style=flat-square)
![Accuracy](https://img.shields.io/badge/Age%20Accuracy-73%25-yellow?style=flat-square)

---

## 📌 What This Project Does

| Feature | Detail |
|---|---|
| 🎥 Input | Live webcam feed |
| 👤 Face Detection | OpenCV Haar Cascade |
| 🧠 Model | Custom CNN (TensorFlow/Keras) |
| 🚻 Gender Prediction | Male / Female — **90% accuracy** |
| 🎂 Age Group Prediction | 5 classes — **73% accuracy** |
| ⚡ Inference | Real-time, per frame |

---

## 🏆 Results at a Glance

### Gender Prediction
| Class | Precision | Recall | F1-Score |
|---|---|---|---|
| Male | 0.90 | 0.91 | **0.91** |
| Female | 0.90 | 0.89 | **0.90** |
| **Overall** | | | **90%** |

### Age Group Prediction
| Class | Precision | Recall | F1-Score |
|---|---|---|---|
| Child (0–12) | 0.92 | 0.90 | **0.91** |
| Teen (13–19) | 0.43 | 0.25 | 0.31 |
| Young Adult (20–35) | 0.75 | 0.90 | **0.82** |
| Adult (36–55) | 0.53 | 0.55 | 0.54 |
| Senior (56+) | 0.87 | 0.44 | 0.58 |
| **Overall** | | | **73%** |

> Teen/Adult boundary confusion is expected — a known challenge even in published research due to high visual similarity between age groups 13–35.

---

## 🗂️ Project Structure

```
pehchan-ai/
├── datasets/               # UTKFace images (23,700+ files)
├── models/
│   ├── pehchan_model.keras # Trained model
│   ├── X_train.npy         # Preprocessed training data
│   └── X_test.npy          # Preprocessed test data
├── notebook/
│   ├── 01_explore_data.ipynb    # Data exploration & visualization
│   ├── 02_preprocess.ipynb      # Image preprocessing & splitting
│   ├── 03_train_model.ipynb     # Model architecture & training
│   └── 04_evaluate.ipynb        # Evaluation & confusion matrices
├── src/
│   └── webcam.py           # Real-time webcam inference script
└── README.md
```

---

## 🧠 Model Architecture

```
Input (64×64×3)
     │
     ▼
Conv2D(32) → BatchNorm → MaxPool
     │
Conv2D(64) → BatchNorm → MaxPool
     │
Conv2D(128) → BatchNorm → MaxPool
     │
Flatten → Dense(256) → Dropout(0.4)
     │
   ┌─┴─┐
   ▼   ▼
Gender  Age Group
(sigmoid) (softmax)
Male/Female  5 classes
```

**Why two output heads?**
A single shared CNN extracts facial features once, then two separate branches specialize — one for gender (binary), one for age group (5-class). This is more efficient than training two separate models.

---

## 📊 Dataset

- **Source:** [UTKFace Dataset — Kaggle](https://www.kaggle.com/datasets/jangedoo/utkface-new)
- **Size:** 23,708 images after cleaning
- **Labels:** Encoded in filename — `age_gender_race_timestamp.jpg`
- **Gender split:** Male: 12,391 | Female: 11,317 (balanced ✅)
- **Age range:** 1–116 years, grouped into 5 classes

| Age Group | Label | Count |
|---|---|---|
| Child | 0–12 | ~3,400 |
| Teen | 13–19 | ~1,100 |
| Young Adult | 20–35 | ~10,600 |
| Adult | 36–55 | ~5,100 |
| Senior | 56+ | ~3,500 |

---

## ⚙️ Setup & Installation

### 1. Clone the repository
```bash
git clone https://github.com/namansachdeva18/Pehchan-ai.git
cd pehchan-ai
```

### 2. Create virtual environment
```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install tensorflow==2.13.0 opencv-python numpy pandas matplotlib scikit-learn Pillow
```

### 4. Download the dataset
Download UTKFace from [Kaggle](https://www.kaggle.com/datasets/jangedoo/utkface-new) and place all `.jpg` images into the `datasets/` folder.

---

## 🚀 Run the Project

### Option A — Real-time webcam (main demo)
```bash
python src/webcam.py
```
A window opens with your webcam. Detected faces are labeled with predicted gender and age group + confidence score. Press **Q** to quit.

### Option B — Step through notebooks
Open notebooks in order inside VS Code or Jupyter:
```
01_explore_data.ipynb   → Understand the dataset
02_preprocess.ipynb     → Prepare images for training
03_train_model.ipynb    → Build and train the CNN
04_evaluate.ipynb       → Evaluate with metrics and visualizations
```

---

## 🛠️ Tech Stack

| Category | Tools |
|---|---|
| Language | Python 3.10 |
| Deep Learning | TensorFlow 2.13, Keras |
| Computer Vision | OpenCV 4.13 |
| Data Processing | NumPy, Pandas, PIL |
| Visualization | Matplotlib |
| Evaluation | scikit-learn |
| IDE | VS Code + Jupyter |

---

## 📈 Training Summary

| Parameter | Value |
|---|---|
| Image size | 64 × 64 px |
| Training samples | 18,966 |
| Test samples | 4,742 |
| Batch size | 64 |
| Optimizer | Adam |
| Epochs trained | 12 (EarlyStopping) |
| Best model saved | `models/pehchan_model.keras` |

Training stopped at epoch 12 via EarlyStopping — no overfitting observed.

---

## 🔮 Future Improvements

- [ ] Replace Haar Cascade with **MTCNN** or **YOLOv8** for better face detection
- [ ] Add **race/ethnicity** as a third prediction head
- [ ] Train on higher resolution images (128×128) for improved age accuracy
- [ ] Deploy as a **web app** using Flask or Streamlit
- [ ] Export model to **TensorFlow Lite** for mobile deployment

---

## 👨‍💻 Author

**Naman Sachdeva**
B.Tech Information Technology | Manipal University Jaipur

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?style=flat-square&logo=linkedin)](https://www.linkedin.com/in/naman-sachdeva18/)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-black?style=flat-square&logo=github)](https://github.com/namansachdeva18)

---


