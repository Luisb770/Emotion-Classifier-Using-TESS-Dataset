# Emotion Classifier Using TESS Dataset

## Overview

This project is an emotion classifier built using the Toronto Emotional Speech Set (TESS) dataset. The goal of this project is to classify emotions from audio files using convolutional neural networks (CNN). The TESS dataset consists of high-quality audio recordings from two female actors portraying seven different emotions: anger, disgust, fear, happiness, pleasant surprise, sadness, and neutral. The dataset is organized into folders by actor and emotion, and each audio file contains one of 200 target words spoken in the context of the phrase "Say the word _."

## Dataset

The TESS dataset contains:
- 2800 audio files (WAV format)
- Two actresses (aged 26 and 64)
- Seven emotions: anger, disgust, fear, happiness, pleasant surprise, sadness, and neutral

You can find the TESS dataset [here on Kaggle](https://www.kaggle.com/datasets/ejlok1/toronto-emotional-speech-set-tess).

## Project Structure

### Code Overview

- **Data Loading and Preprocessing**: Audio files are converted into Mel-spectrograms using the `librosa` library. The spectrograms are then padded or truncated to a fixed length for uniformity.
  
- **Model Architecture**: A CNN is used for classifying emotions. The model includes multiple convolutional and pooling layers followed by a fully connected layer for final classification.
  
- **Evaluation and Visualization**: The model's performance is evaluated using confusion matrices and t-SNE plots for visualizing emotion separation in the feature space.

### Key Functions

1. **audio_to_spectrogram**: Converts audio files to Mel-spectrograms and ensures consistent length.
2. **load_data**: Loads the audio files, extracts spectrograms, and labels the emotions.
3. **encode_labels**: Converts emotion labels to numeric values for model training.
4. **build_cnn**: Defines the CNN architecture for emotion classification.
5. **plot_confusion_matrix**: Visualizes model performance using a confusion matrix.
6. **plot_tsne**: Generates t-SNE plots to visualize feature separation.

### Model Training

- The model is trained using 80% of the dataset, with the remaining 20% used for validation.
- Training history, including accuracy and validation accuracy, is plotted to track model performance.
  
### Visualizations

- **Confusion Matrix**: Displays how well the model predicts each emotion.
- **t-SNE Plot**: A 2D representation of the spectrogram features, showing clusters for each emotion.

### Emotion Distribution

The dataset is slightly imbalanced, with some emotions appearing more frequently than others. This distribution is visualized using a bar plot.

## How to Run

1. Clone the repository to your local machine or open it in Google Colab.
2. Download the TESS dataset from [Kaggle](https://www.kaggle.com/datasets/ejlok1/toronto-emotional-speech-set-tess) and upload it to your Google Drive.
3. Modify the `base_dir` in the `main()` function to point to the correct directory in your Google Drive where the dataset is stored.
4. Run the `main()` function to load the data, train the model, and visualize the results.

## Libraries Used

- `librosa` for audio processing
- `tensorflow` and `keras` for building the CNN
- `scikit-learn` for data splitting and evaluation metrics
- `matplotlib` and `seaborn` for visualizations

## Acknowledgements

We would like to thank the University of Toronto for providing the TESS dataset. This high-quality dataset has been instrumental in training a robust emotion classifier.

---

Feel free to explore the code, experiment with different models, or use the dataset for your own emotion classification tasks!
