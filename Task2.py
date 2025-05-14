import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, hamming_loss
from skmultilearn.model_selection import iterative_train_test_split
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf
from tensorflow.keras import layers, models
import warnings
import joblib
import pickle

# Suppress TensorFlow deprecation warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)

# Load the dataset
def load_data():
    dataset_path = r"C:\Users\Ahmad\Desktop\DS-A4\dataset.csv"
    data = pd.read_csv(dataset_path)
    X = data['report']  # Text descriptions
    y = data[['type_blocker', 'type_regression', 'type_bug', 'type_documentation', 
             'type_enhancement', 'type_dependency_upgrade']].values  # Multi-label targets
    return X, y, data

# Preprocess the text data
def preprocess_data(X, y):
    vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
    X_tfidf = vectorizer.fit_transform(X).toarray()
    return X_tfidf, y, vectorizer

# Check for labels with insufficient class diversity
def check_class_diversity(y, labels):
    valid_indices = []
    valid_labels = []
    for i, label in enumerate(labels):
        unique_classes = np.unique(y[:, i])
        if len(unique_classes) > 1:  # Ensure at least two classes
            valid_indices.append(i)
            valid_labels.append(label)
        else:
            print(f"Skipping label '{label}' because it has only one class: {unique_classes}")
    return valid_indices, valid_labels

# Load and preprocess data
X, y, data = load_data()

# Print class distribution for each label in the full dataset
all_labels = ['type_blocker', 'type_regression', 'type_bug', 'type_documentation', 
              'type_enhancement', 'type_dependency_upgrade']
print("Class distribution in the full dataset:")
for label in all_labels:
    print(f"\nLabel: {label}")
    print(data[label].value_counts())

X_tfidf, y, vectorizer = preprocess_data(X, y)

# Save the vectorizer for Task 3
with open("task2_tfidf_vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

# Split the data with stratification for multi-label
X_train, y_train, X_test, y_test = iterative_train_test_split(X_tfidf, y, test_size=0.2)

# Check class diversity in training data
valid_indices, valid_labels = check_class_diversity(y_train, all_labels)

# Filter the training and test labels to only include valid ones
if valid_indices:
    y_train_filtered = y_train[:, valid_indices]
    y_test_filtered = y_test[:, valid_indices]
else:
    raise ValueError("No labels have sufficient class diversity to train the model.")

# Use the filtered training data directly (relying on class weighting)
X_train_resampled, y_train_resampled = X_train, y_train_filtered

# Model 1: Logistic Regression with MultiOutputClassifier and class weighting
lr = MultiOutputClassifier(LogisticRegression(max_iter=1000, class_weight='balanced'))
lr.fit(X_train_resampled, y_train_resampled)
lr_pred = lr.predict(X_test)
joblib.dump(lr, "task2_lr_model.pkl")

# Model 2: SVM with MultiOutputClassifier and class weighting
svm = MultiOutputClassifier(SVC(probability=True, class_weight='balanced'))
svm.fit(X_train_resampled, y_train_resampled)
svm_pred = svm.predict(X_test)
joblib.dump(svm, "task2_svm_model.pkl")

# Model 3: Deep Neural Network (DNN) without class weighting
dnn = models.Sequential([
    layers.Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    layers.Dense(64, activation='relu'),
    layers.Dense(len(valid_indices), activation='sigmoid')  # Multi-label output for valid labels
])
dnn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
dnn.fit(X_train_resampled, y_train_resampled, epochs=10, batch_size=32, verbose=0)
dnn_pred = (dnn.predict(X_test, verbose=0) > 0.5).astype(int)
dnn.save("task2_dnn_model.h5")

# Evaluation function with multi-label metrics
def evaluate_model(y_true, y_pred, model_name, labels, k=2):
    print(f"\n{model_name} Results:")
    
    # Hamming Loss
    hamming = hamming_loss(y_true, y_pred)
    print(f"Hamming Loss: {hamming:.4f}")
    
    # Micro-F1
    micro_f1 = f1_score(y_true, y_pred, average='micro', zero_division=0)
    print(f"Micro-F1: {micro_f1:.4f}")
    
    # Macro-F1
    macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    print(f"Macro-F1: {macro_f1:.4f}")
    
    # Precision@k
    # For each sample, sort predicted labels by confidence (here, we use prediction values as proxies for confidence)
    # Since SVM and LR don't provide probabilities, we'll compute Precision@k based on the predicted 1s
    precision_at_k = []
    for i in range(y_true.shape[0]):
        true_labels = np.where(y_true[i] == 1)[0]  # Indices of true labels
        pred_labels = np.where(y_pred[i] == 1)[0]  # Indices of predicted labels
        if len(pred_labels) == 0:  # If no labels predicted, precision is 0
            precision_at_k.append(0)
            continue
        # Take top-k predicted labels (or fewer if less than k predicted)
        top_k_pred = pred_labels[:min(k, len(pred_labels))]
        if len(true_labels) == 0:  # If no true labels, precision is 0
            precision_at_k.append(0)
            continue
        # Compute precision for top-k predictions
        correct = len(set(top_k_pred).intersection(set(true_labels)))
        precision_at_k.append(correct / min(k, len(pred_labels)))
    avg_precision_at_k = np.mean(precision_at_k)
    print(f"Precision@{k}: {avg_precision_at_k:.4f}")
    
    # Per-label metrics (retaining the original evaluation)
    for i, label in enumerate(labels):
        print(f"\nLabel: {label}")
        print(f"Accuracy: {accuracy_score(y_true[:, i], y_pred[:, i]):.4f}")
        print(f"Precision: {precision_score(y_true[:, i], y_pred[:, i], zero_division=0):.4f}")
        print(f"Recall: {recall_score(y_true[:, i], y_pred[:, i], zero_division=0):.4f}")
        print(f"F1-Score: {f1_score(y_true[:, i], y_pred[:, i], zero_division=0):.4f}")
        print(f"ROC-AUC: {roc_auc_score(y_true[:, i], y_pred[:, i]):.4f}")

# Evaluate all models on valid labels
evaluate_model(y_test_filtered, lr_pred, "Logistic Regression", valid_labels, k=2)
evaluate_model(y_test_filtered, svm_pred, "SVM", valid_labels, k=2)
evaluate_model(y_test_filtered, dnn_pred, "DNN", valid_labels, k=2)