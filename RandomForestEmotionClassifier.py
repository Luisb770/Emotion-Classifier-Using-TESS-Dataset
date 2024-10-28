from google.colab import drive
drive.mount('/content/drive')

import os
import librosa
import pandas as pd
import numpy as np  
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
from tsfresh import extract_features
from tsfresh.feature_extraction.settings import ComprehensiveFCParameters

# Function to extract features from audio files
def extract_audio_features(file_path):
    y, sr = librosa.load(file_path, duration=3)
    features = {
        'mean': [y.mean()],
        'std': [y.std()],
        'zero_crossing_rate': [librosa.feature.zero_crossing_rate(y).mean()],
        'spectral_centroid': [librosa.feature.spectral_centroid(y=y, sr=sr).mean()],
        'spectral_rolloff': [librosa.feature.spectral_rolloff(y=y, sr=sr).mean()],
        'mfcc_mean': [librosa.feature.mfcc(y=y, sr=sr).mean()],
        'mfcc_std': [librosa.feature.mfcc(y=y, sr=sr).std()]
    }
    return pd.DataFrame(features)

# Load data and extract features and emotion labels
def load_data(base_dir):
    emotions = []
    features_list = []
    for folder in os.listdir(base_dir):
        folder_path = os.path.join(base_dir, folder)
        if os.path.isdir(folder_path):
            for file in os.listdir(folder_path):
                if file.endswith('.wav'):
                    file_path = os.path.join(folder_path, file)
                    features = extract_audio_features(file_path)
                    features_list.append(features)
                    emotions.append(folder.split('_')[1])  # Extract emotion from folder name
    return pd.concat(features_list, ignore_index=True), np.array(emotions)

# Convert labels (emotions) to numeric
def encode_labels(labels):
    unique_labels = np.unique(labels)
    label_map = {label: idx for idx, label in enumerate(unique_labels)}
    return np.array([label_map[label] for label in labels]), label_map

# Train and evaluate Random Forest model
def train_random_forest(X_train, y_train, X_test, y_test):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy * 100:.2f}%")
    
    # Display classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Feature importance
    feature_importances = model.feature_importances_
    feature_names = X_train.columns
    importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importances})
    importance_df = importance_df.sort_values(by='Importance', ascending=False)
    
    print("\nTop 10 Important Features:")
    print(importance_df.head(10))
    
    return model, y_pred

# Plot confusion matrix
def plot_confusion_matrix(y_true, y_pred, labels):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.title('Confusion Matrix')
    plt.ylabel('True Labels')
    plt.xlabel('Predicted Labels')
    plt.show()

# Main function
def main():
    base_dir = '/content/drive/My Drive/Colab Notebooks/TESS Toronto emotional speech set data/'
    features, emotions = load_data(base_dir)

    emotion_labels, label_map = encode_labels(emotions)
    X_train, X_test, y_train, y_test = train_test_split(features, emotion_labels, test_size=0.2, random_state=42)

    # Train Random Forest Classifier
    model, y_pred = train_random_forest(X_train, y_train, X_test, y_test)

    # Visualize Confusion Matrix
    plot_confusion_matrix(y_test, y_pred, labels=list(label_map.keys()))

if __name__ == "__main__":
    main()
