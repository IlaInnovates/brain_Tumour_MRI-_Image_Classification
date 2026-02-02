🧠 Brain Tumor MRI Image Classification using Deep Learning
📌 Project Overview

Brain tumor detection from MRI images is a critical task in medical image analysis. This project aims to classify brain MRI images into four categories using Deep Learning (CNN and Transfer Learning) techniques and deploy the trained model using a Streamlit web application.

The system allows users to upload an MRI image and receive:

Predicted tumor type

Confidence scores for all classes

🎯 Objectives

Analyze and preprocess brain MRI images

Build a Custom CNN model from scratch

Apply Transfer Learning (EfficientNetB0)

Compare model performances

Deploy the best-performing model using Streamlit

Provide an intuitive and interactive user interface

🧬 Dataset Information

Dataset Name: Tumour (Updated)
Classes (4):

Glioma

Meningioma

No Tumor

Pituitary

Directory Structure:

Tumour/
│── train/
│   ├── glioma/
│   ├── meningioma/
│   ├── no_tumor/
│   └── pituitary/
│── val/
│── test/

🔁 Project Workflow
1️⃣ Dataset Understanding

Verified class distribution

Visualized sample images

Checked image formats and resolutions

2️⃣ Data Preprocessing

Resized images to 224 × 224

Normalized pixel values to [0,1]

Converted images to RGB format

3️⃣ Data Augmentation

Applied transformations to improve generalization:

Rotation

Zoom

Horizontal flip

Brightness adjustment

4️⃣ Model Building – Custom CNN

Convolution + MaxPooling layers

Batch Normalization

Dropout for regularization

Dense layers with Softmax output

Result:
✅ Achieved ~86% accuracy

5️⃣ Transfer Learning – EfficientNetB0

Loaded ImageNet pretrained weights

Replaced top layers with custom classifier

Fine-tuned selected layers

Result:
⚠️ Achieved ~54% accuracy
(Lower due to limited dataset size)

6️⃣ Model Training

Optimizer: Adam

Loss: Categorical Crossentropy

Epochs: 20

ModelCheckpoint used to save best model

EarlyStopping removed for full training

7️⃣ Model Evaluation

Accuracy

Precision, Recall, F1-score

Confusion Matrix

Training & Validation Loss plots

8️⃣ Model Comparison
Model	Accuracy	Observation
Custom CNN	~86%	Best performance
EfficientNetB0	~54%	Needs larger dataset

✅ Custom CNN selected for deployment

9️⃣ Streamlit Application Deployment

Features:

Upload MRI image (jpg / png / jpeg)

Displays:

Primary prediction

Secondary possible class

Confidence scores

Warns when prediction confidence is low

🖥️ Streamlit App Usage
Run the app:
streamlit run app.py

App Output:

Predicted tumor type

Probability distribution across all classes

📂 Project Structure
project5/
│── app.py
│── custom_cnn_best.keras
│── README.md
│── requirements.txt

🛠️ Technologies Used

---> Python

---> TensorFlow / Keras

---> NumPy

---> Matplotlib

---> Streamlit

---> PIL (Image Processing)



✅ Conclusion

This project successfully demonstrates the application of deep learning for medical image classification.
The Custom CNN model proved more effective than transfer learning due to dataset size constraints.
The Streamlit deployment enables easy real-time testing and visualization.

📌 Future Enhancements

Increase dataset size

Apply Grad-CAM for explainability

Improve EfficientNet fine-tuning

Add user authentication to Streamlit app

👩‍⚕️ Disclaimer

This project is intended for educational purposes only and should not be used for clinical diagnosis.
