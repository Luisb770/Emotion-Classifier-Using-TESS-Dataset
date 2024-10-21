
from google.colab import drive
drive.mount('/content/drive')

# Import necessary libraries
import os
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.manifold import TSNE

# Function to load audio and convert to spectrogram
def audio_to_spectrogram(file_path, n_mels=128, fixed_length=128):
    y, sr = librosa.load(file_path, duration=3)  # Load audio file
    spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)  # Convert to Mel-spectrogram
    log_spectrogram = librosa.power_to_db(spectrogram, ref=np.max)  # Convert to log scale (decibels)

    # Pad or truncate to fixed length (time axis)
    if log_spectrogram.shape[1] < fixed_length:
        # Pad with zeros if shorter than the fixed length
        pad_width = fixed_length - log_spectrogram.shape[1]
        log_spectrogram = np.pad(log_spectrogram, ((0, 0), (0, pad_width)), mode='constant')
    else:
        # Truncate if longer than the fixed length
        log_spectrogram = log_spectrogram[:, :fixed_length]

    return log_spectrogram

# Load data and extract spectrograms and emotion labels
def load_data(base_dir, fixed_length=128):
    emotions = []
    spectrograms = []
    for folder in os.listdir(base_dir):
        folder_path = os.path.join(base_dir, folder)
        if os.path.isdir(folder_path):
            for file in os.listdir(folder_path):
                if file.endswith('.wav'):
                    file_path = os.path.join(folder_path, file)
                    spec = audio_to_spectrogram(file_path, fixed_length=fixed_length)
                    spectrograms.append(spec)
                    emotions.append(folder.split('_')[1])  # Extract emotion from folder name
    return np.array(spectrograms), np.array(emotions)

# Convert labels (emotions) to numeric
def encode_labels(labels):
    unique_labels = np.unique(labels)
    label_map = {label: idx for idx, label in enumerate(unique_labels)}
    return np.array([label_map[label] for label in labels]), label_map

# Define CNN model
def build_cnn(input_shape, num_classes):
    model = models.Sequential([
        layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(128, kernel_size=(3, 3), activation='relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# Plot confusion matrix
def plot_confusion_matrix(y_true, y_pred, labels):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.title('Confusion Matrix')
    plt.ylabel('True Labels')
    plt.xlabel('Predicted Labels')
    plt.show()

# Plot t-SNE
def plot_tsne(X, y, labels):
    tsne = TSNE(n_components=2, random_state=42)
    X_2d = tsne.fit_transform(X)

    plt.figure(figsize=(10, 7))
    sns.scatterplot(x=X_2d[:, 0], y=X_2d[:, 1], hue=y, palette='coolwarm', s=100, alpha=0.7, legend='full')
    plt.title('t-SNE plot of Audio Features')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.legend(title='Emotion', labels=labels)
    plt.show()

# Main function
def main():
    base_dir = '/content/drive/My Drive/Colab Notebooks/TESS Toronto emotional speech set data/'
    spectrograms, emotions = load_data(base_dir)
    spectrograms = np.expand_dims(spectrograms, axis=-1)

    emotion_labels, label_map = encode_labels(emotions)
    X_train, X_test, y_train, y_test = train_test_split(spectrograms, emotion_labels, test_size=0.2, random_state=42)

    input_shape = (X_train.shape[1], X_train.shape[2], 1)
    num_classes = len(label_map)
    model = build_cnn(input_shape, num_classes)
    model.summary()

    history = model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test), batch_size=32)

    # Output emotion counts
    emotion_counts = {emotion: list(emotions).count(emotion) for emotion in label_map.keys()}
    most_common_emotion = max(emotion_counts, key=emotion_counts.get)
    print(f"Emotion counts: {emotion_counts}")
    print(f"The most frequent emotion is: {most_common_emotion}")

    # Plot emotion distribution
    sns.barplot(x=list(emotion_counts.keys()), y=list(emotion_counts.values()))
    plt.title('Emotion Distribution in the TESS Dataset')
    plt.xlabel('Emotion')
    plt.ylabel('Count')
    plt.show()

    # Plot training and validation accuracy
    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label='val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0, 1])
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')
    plt.show()

    # Visualizations
    y_pred = np.argmax(model.predict(X_test), axis=1)  # Updated prediction method
    plot_confusion_matrix(y_test, y_pred, labels=list(label_map.keys()))

    spectrograms_flattened = np.array([spec.flatten() for spec in X_test])
    plot_tsne(spectrograms_flattened, y_test, list(label_map.keys()))

if __name__ == "__main__":
    main()
