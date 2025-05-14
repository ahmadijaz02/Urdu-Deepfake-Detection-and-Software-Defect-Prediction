# Urdu Deepfake Detection and Software Defect Prediction

Welcome to the repository for my machine learning project, developed as part of a university assignment. This project comprises two main tasks: detecting deepfake Urdu audio and predicting multiple software defect types, complemented by an interactive Streamlit web application for real-time predictions.

## Table of Contents
- [Overview](#overview)
- [Tasks](#tasks)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Files and Structure](#files-and-structure)
- [Dependencies](#dependencies)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Overview
This project demonstrates the application of machine learning techniques to real-world problems:
- **Task 1**: Binary classification to distinguish between Bonafide (real) and Deepfake (synthetic) Urdu audio files using MFCC features and models like SVM, Logistic Regression, and Deep Neural Networks (DNN).
- **Task 2**: Multi-label classification to predict software defect types (e.g., `type_bug`, `type_blocker`) from text reports using Logistic Regression, SVM, and DNN.
- **Task 3**: An interactive Streamlit app allowing users to upload audio files or input defect reports for predictions with confidence scores.

The project highlights skills in audio processing, text feature extraction, multi-label classification, and web-based deployment.

## Tasks
1. **Task 1: Urdu Deepfake Audio Detection**
   - Extracts MFCC features from Urdu audio files.
   - Trains SVM, Logistic Regression, and DNN models to classify audio as Bonafide or Deepfake.
   - Achieves high AUC-ROC scores (e.g., ~0.90 for DNN).

2. **Task 2: Multi-Label Defect Prediction**
   - Preprocesses text reports using TF-IDF vectorization.
   - Trains multi-label classifiers (Logistic Regression, SVM, DNN) with class weighting to handle imbalance.
   - Evaluates performance using Hamming Loss, Micro-F1, Macro-F1, and Precision@k metrics.

3. **Task 3: Interactive Streamlit App**
   - Provides a user-friendly interface for both tasks.
   - Supports model selection (SVM, Logistic Regression, DNN) and displays predictions with confidence scores.

## Features
- Audio preprocessing with MFCC extraction for deepfake detection.
- Text preprocessing with TF-IDF for defect prediction.
- Multi-label classification with handling for imbalanced datasets.
- Real-time predictions via a Streamlit web app.
- Model performance summaries based on evaluation metrics.

## Installation
### Prerequisites
- Python 3.11
- Virtual environment (recommended)

