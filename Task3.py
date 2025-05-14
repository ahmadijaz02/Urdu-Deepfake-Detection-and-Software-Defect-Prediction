import streamlit as st
import numpy as np
import librosa
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import pickle
import tensorflow as tf
import warnings

# Suppress warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)

# Load Task 1 models and preprocessing objects
task1_scaler = joblib.load("task1_scaler.pkl")
task1_svm_model = joblib.load("task1_svm_model.pkl")
task1_lr_model = joblib.load("task1_lr_model.pkl")
task1_dnn_model = tf.keras.models.load_model("task1_dnn_model.h5")

# Load Task 2 models and preprocessing objects
with open("task2_tfidf_vectorizer.pkl", "rb") as f:
    task2_vectorizer = pickle.load(f)
task2_lr_model = joblib.load("task2_lr_model.pkl")
task2_svm_model = joblib.load("task2_svm_model.pkl")
task2_dnn_model = tf.keras.models.load_model("task2_dnn_model.h5")

# Task 1 labels (binary classification)
task1_labels = ["Bonafide", "Deepfake"]

# Task 2 labels (multi-label classification)
task2_labels = ['type_blocker', 'type_regression', 'type_bug', 'type_documentation', 
                'type_enhancement', 'type_dependency_upgrade']

# Function to preprocess audio file for Task 1
def preprocess_audio(audio_file):
    max_timesteps = 100  # Must match Task 1
    try:
        audio, sr = librosa.load(audio_file, sr=16000)
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
        mfccs = mfccs.T
        if mfccs.shape[0] > max_timesteps:
            mfccs = mfccs[:max_timesteps, :]
        else:
            pad_width = max_timesteps - mfccs.shape[0]
            mfccs = np.pad(mfccs, ((0, pad_width), (0, 0)), mode='constant')
        mfccs_flattened = mfccs.flatten()
        mfccs_flattened = mfccs_flattened.reshape(1, -1)  # Reshape for scaler
        mfccs_scaled = task1_scaler.transform(mfccs_flattened)
        return mfccs_scaled
    except Exception as e:
        st.error(f"Error processing audio: {e}")
        return None

# Function to predict for Task 1 (Deepfake Detection)
def predict_deepfake(audio_features, model_type):
    if audio_features is None:
        return None, None
    
    if model_type == "SVM":
        pred = task1_svm_model.predict(audio_features)[0]
        prob = task1_svm_model.predict_proba(audio_features)[0][1]  # Probability of Deepfake
    elif model_type == "Logistic Regression":
        pred = task1_lr_model.predict(audio_features)[0]
        prob = task1_lr_model.predict_proba(audio_features)[0][1]
    else:  # DNN
        prob = task1_dnn_model.predict(audio_features, verbose=0)[0][0]
        pred = 1 if prob > 0.5 else 0
    
    label = task1_labels[pred]
    confidence = prob if pred == 1 else 1 - prob  # Confidence for the predicted class
    return label, confidence

# Function to preprocess text for Task 2
def preprocess_text(text):
    try:
        text_tfidf = task2_vectorizer.transform([text]).toarray()
        return text_tfidf
    except Exception as e:
        st.error(f"Error processing text: {e}")
        return None

# Function to predict for Task 2 (Defect Prediction)
def predict_defects(text_features, model_type):
    if text_features is None:
        return None, None
    
    if model_type == "SVM":
        pred = task2_svm_model.predict(text_features)[0]
        # SVM with MultiOutputClassifier doesn't provide predict_proba directly
        confidence = [0.5] * len(pred)  # Placeholder confidence
    elif model_type == "Logistic Regression":
        pred = task2_lr_model.predict(text_features)[0]
        confidence = [0.5] * len(pred)  # Placeholder confidence
    else:  # DNN
        prob = task2_dnn_model.predict(text_features, verbose=0)[0]
        pred = (prob > 0.5).astype(int)
        confidence = prob  # Probabilities for each label
    
    # Map predictions to labels
    predicted_labels = [task2_labels[i] for i in range(len(pred)) if pred[i] == 1]
    if not predicted_labels:
        predicted_labels = ["No defects predicted"]
    
    # For confidence, use the probability for predicted labels
    if model_type == "DNN":
        confidence_dict = {task2_labels[i]: prob[i] for i in range(len(prob)) if pred[i] == 1}
    else:
        confidence_dict = {label: 0.5 for label in predicted_labels}  # Placeholder
    
    return predicted_labels, confidence_dict

# Streamlit App
st.title("Deepfake Audio and Software Defect Prediction App")

# Section 1: Deepfake Audio Detection
st.header("Deepfake Audio Detection")
st.write("Upload an audio file to detect if it's Bonafide (real) or Deepfake.")
audio_model_type = st.selectbox("Select Model for Deepfake Detection:", 
                                ["SVM", "Logistic Regression", "DNN"],
                                key="audio_model")

uploaded_audio = st.file_uploader("Upload Audio File (WAV format)", type=["wav"])
if uploaded_audio is not None:
    st.audio(uploaded_audio, format="audio/wav")
    if st.button("Predict Deepfake"):
        audio_features = preprocess_audio(uploaded_audio)
        label, confidence = predict_deepfake(audio_features, audio_model_type)
        if label is not None:
            st.success(f"Prediction: **{label}**")
            st.write(f"Confidence: **{confidence:.2f}**")

# Section 2: Software Defect Prediction
st.header("Software Defect Prediction")
st.write("Enter a software defect report to predict defect types.")
defect_model_type = st.selectbox("Select Model for Defect Prediction:", 
                                 ["SVM", "Logistic Regression", "DNN"],
                                 key="defect_model")

report_text = st.text_area("Enter Defect Report:", 
                           placeholder="Enter the software defect report here...")
if st.button("Predict Defects"):
    if report_text:
        text_features = preprocess_text(report_text)
        predicted_labels, confidence_dict = predict_defects(text_features, defect_model_type)
        if predicted_labels is not None:
            st.success("Predicted Defect Types:")
            for label in predicted_labels:
                confidence = confidence_dict.get(label, 0.5)
                st.write(f"- **{label}** (Confidence: {confidence:.2f})")
    else:
        st.error("Please enter a defect report.")