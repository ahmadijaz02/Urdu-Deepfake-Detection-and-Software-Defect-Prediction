import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras import layers, models
import librosa
import os
import glob
from tqdm import tqdm
import warnings
import joblib
import pickle

tf.get_logger().setLevel('ERROR')  
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  
warnings.filterwarnings('ignore', category=Warning)  # Suppress all warnings

def load_and_preprocess_data():
    dataset_dir = r"C:\Users\Ahmad\Desktop\DS-A4\deepfake_detection_dataset_urdu"
    audio_features = []
    labels = []
    max_timesteps = 100  # Fixed length for MFCC timesteps
    
    # Load Bonafide (real) audio files (label = 0)
    bonafide_files = glob.glob(os.path.join(dataset_dir, "Bonafide", "**", "*.wav"), recursive=True)
    print(f"Processing Bonafide files... ({len(bonafide_files)} files)")
    for audio_path in tqdm(bonafide_files, desc="Bonafide"):
        try:
            audio, sr = librosa.load(audio_path, sr=16000)
            mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
            mfccs = mfccs.T
            if mfccs.shape[0] > max_timesteps:
                mfccs = mfccs[:max_timesteps, :]
            else:
                pad_width = max_timesteps - mfccs.shape[0]
                mfccs = np.pad(mfccs, ((0, pad_width), (0, 0)), mode='constant')
            mfccs_flattened = mfccs.flatten()
            audio_features.append(mfccs_flattened)
            labels.append(0)  # 0 for Bonafide (real)
        except Exception as e:
            print(f"Error processing {audio_path}: {e}")
            continue
    
    # Load Spoofed_TTS (deepfake) audio files (label = 1)
    spoofed_tts_files = glob.glob(os.path.join(dataset_dir, "Spoofed_TTS", "**", "*.wav"), recursive=True)
    print(f"Processing Spoofed_TTS files... ({len(spoofed_tts_files)} files)")
    for audio_path in tqdm(spoofed_tts_files, desc="Spoofed_TTS"):
        try:
            audio, sr = librosa.load(audio_path, sr=16000)
            mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
            mfccs = mfccs.T
            if mfccs.shape[0] > max_timesteps:
                mfccs = mfccs[:max_timesteps, :]
            else:
                pad_width = max_timesteps - mfccs.shape[0]
                mfccs = np.pad(mfccs, ((0, pad_width), (0, 0)), mode='constant')
            mfccs_flattened = mfccs.flatten()
            audio_features.append(mfccs_flattened)
            labels.append(1)  # 1 for deepfake
        except Exception as e:
            print(f"Error processing {audio_path}: {e}")
            continue
    
    # Load Spoofed_Tacotron (deepfake) audio files (label = 1)
    spoofed_tacotron_files = glob.glob(os.path.join(dataset_dir, "Spoofed_Tacotron", "**", "*.wav"), recursive=True)
    print(f"Processing Spoofed_Tacotron files... ({len(spoofed_tacotron_files)} files)")
    for audio_path in tqdm(spoofed_tacotron_files, desc="Spoofed_Tacotron"):
        try:
            audio, sr = librosa.load(audio_path, sr=16000)
            mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
            mfccs = mfccs.T
            if mfccs.shape[0] > max_timesteps:
                mfccs = mfccs[:max_timesteps, :]
            else:
                pad_width = max_timesteps - mfccs.shape[0]
                mfccs = np.pad(mfccs, ((0, pad_width), (0, 0)), mode='constant')
            mfccs_flattened = mfccs.flatten()
            audio_features.append(mfccs_flattened)
            labels.append(1)  # 1 for deepfake
        except Exception as e:
            print(f"Error processing {audio_path}: {e}")
            continue
    
    audio_features = np.array(audio_features)
    labels = np.array(labels)
    
    print(f"\nTotal samples processed: {len(audio_features)}")
    print(f"Bonafide samples: {len(bonafide_files)}")
    print(f"Spoofed_TTS samples: {len(spoofed_tts_files)}")
    print(f"Spoofed_Tacotron samples: {len(spoofed_tacotron_files)}")
    
    return audio_features, labels

# Load and preprocess data
print("Loading and preprocessing data...")
X, y = load_and_preprocess_data()

# Split the data
print("Splitting data into training and test sets...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")

# Scale the data
print("Scaling the data...")
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Save the scaler for Task 3
joblib.dump(scaler, "task1_scaler.pkl")

# Model 1: Support Vector Machine (SVM)
print("Training SVM...")
svm = SVC(probability=True)
svm.fit(X_train, y_train)
print("SVM training completed. Making predictions...")
svm_pred = svm.predict(X_test)
svm_prob = svm.predict_proba(X_test)[:, 1]
joblib.dump(svm, "task1_svm_model.pkl")

# Model 2: Logistic Regression
print("Training Logistic Regression...")
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, y_train)
print("Logistic Regression training completed. Making predictions...")
lr_pred = lr.predict(X_test)
lr_prob = lr.predict_proba(X_test)[:, 1]
joblib.dump(lr, "task1_lr_model.pkl")

# Model 3: Single-Layer Perceptron
print("Training Perceptron...")
perceptron = Perceptron()
perceptron.fit(X_train, y_train)
print("Perceptron training completed. Making predictions...")
perceptron_pred = perceptron.predict(X_test)
joblib.dump(perceptron, "task1_perceptron_model.pkl")

# Model 4: Deep Neural Network (DNN)
print("Building and training DNN...")
dnn = models.Sequential([
    layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    layers.Dropout(0.3),
    layers.Dense(32, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(1, activation='sigmoid')
])
dnn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
dnn.fit(X_train, y_train, epochs=10, batch_size=32, verbose=1)
print("DNN training completed. Making predictions...")
dnn_prob = dnn.predict(X_test, verbose=0)
dnn_pred = (dnn_prob > 0.5).astype(int).flatten()
dnn.save("task1_dnn_model.h5")

# Evaluation function
def evaluate_model(y_true, y_pred, y_prob=None):
    metrics = {
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred),
        'Recall': recall_score(y_true, y_pred),
        'F1-Score': f1_score(y_true, y_pred)
    }
    if y_prob is not None:
        metrics['AUC-ROC'] = roc_auc_score(y_true, y_prob)
    return metrics

# Evaluate all models
print("\nEvaluating models...")
print("SVM Results:", evaluate_model(y_test, svm_pred, svm_prob))
print("Logistic Regression Results:", evaluate_model(y_test, lr_pred, lr_prob))
print("Perceptron Results:", evaluate_model(y_test, perceptron_pred))
print("DNN Results:", evaluate_model(y_test, dnn_pred, dnn_prob))
print("Evaluation completed.")