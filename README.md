# Emotion Classifier Using TESS Dataset

## Overview

This project is an emotion classifier built using the Toronto Emotional Speech Set (TESS) dataset. The goal of this project is to classify emotions from audio files using two different approaches: Convolutional Neural Networks (CNN) and Random Forests. The TESS dataset consists of high-quality audio recordings from two female actors portraying seven different emotions: anger, disgust, fear, happiness, pleasant surprise, sadness, and neutral. The dataset is organized into folders by actor and emotion, and each audio file contains one of 200 target words spoken in the context of the phrase "Say the word _."

## Dataset

The TESS dataset contains:
- 2800 audio files (WAV format)
- Two actresses (aged 26 and 64)
- Seven emotions: anger, disgust, fear, happiness, pleasant surprise, sadness, and neutral

You can find the TESS dataset [here on Kaggle](https://www.kaggle.com/datasets/ejlok1/toronto-emotional-speech-set-tess).

## Project Structure

### Code Overview

- **Data Loading and Preprocessing**: Audio files are processed to extract features for both the CNN and Random Forest models. The preprocessing varies depending on the model approach.

### Approach 1: Convolutional Neural Network (CNN)

- **Model Architecture**: 
  - The CNN approach processes audio files by converting them into Mel-spectrograms using the `librosa` library. These spectrograms are visual representations of the audio signal.
  - The model includes multiple convolutional and pooling layers, followed by fully connected layers that classify the emotions based on learned patterns from the spectrograms.
  
- **Evaluation and Visualization**: 
  - The performance of the CNN model is evaluated using confusion matrices and t-SNE plots, which help visualize how well the model separates different emotions in the feature space.
  - Additional visualizations include bar plots showing the distribution of emotions in the dataset and training/validation accuracy plots to track model performance over epochs.

### Approach 2: Random Forest

- **Model Architecture**: 
  - To address overfitting issues observed with the CNN model, a Random Forest approach was introduced. 
  - Instead of relying on spectrograms, this model uses statistical features extracted from the audio (e.g., MFCCs, zero-crossing rate, spectral properties). 
  - The Random Forest model is trained using these features, which helps it learn patterns across different emotions without requiring the same level of data volume as the CNN.

- **Evaluation and Visualization**: 
  - The Random Forest model's performance is evaluated using confusion matrices and classification reports that provide precision, recall, and F1-scores for each class. 
  - Feature importance plots are generated to show which features contribute the most to the classification, helping to understand the model's decision-making process.

### Key Functions

#### For CNN:
1. `audio_to_spectrogram`: Converts audio files to Mel-spectrograms and ensures consistent length.
2. `load_data`: Loads the audio files, extracts spectrograms, and labels the emotions.
3. `encode_labels`: Converts emotion labels to numeric values for model training.
4. `build_cnn`: Defines the CNN architecture for emotion classification.
5. `plot_confusion_matrix`: Visualizes CNN model performance using a confusion matrix.
6. `plot_tsne`: Generates t-SNE plots to visualize feature separation in the CNN approach.
7. `plot_training_accuracy`: Displays training and validation accuracy over epochs.
8. `plot_emotion_distribution`: Shows a bar plot of the distribution of emotions in the dataset.

#### For Random Forest:
1. `extract_audio_features`: Extracts statistical features (e.g., MFCCs, spectral properties) for use in the Random Forest model.
2. `load_data`: Loads the audio files, extracts features, and labels the emotions.
3. `encode_labels`: Converts emotion labels to numeric values for model training.
4. `train_random_forest`: Trains a Random Forest model for emotion classification.
5. `plot_confusion_matrix`: Visualizes Random Forest model performance using a confusion matrix.
6. `plot_feature_importance`: Shows the importance of different features in the Random Forest model.
7. `classification_report`: Provides precision, recall, and F1-scores for each class, offering a detailed evaluation of model performance.

### Model Training

#### CNN Training
- The CNN model is trained using 80% of the dataset, with the remaining 20% used for validation.
- Training history, including accuracy and validation accuracy, is plotted to track model performance over epochs.
  
#### Random Forest Training
- The Random Forest model is trained on statistical features extracted from the audio data. 
- By averaging the results of multiple decision trees, the model reduces overfitting, making it more robust for smaller datasets.

### Visualizations

#### For CNN:
- **Confusion Matrix**: Displays how well the CNN model predicts each emotion.
- **t-SNE Plot**: A 2D representation of the spectrogram features, showing clusters for each emotion.
- **Training and Validation Accuracy**: Plots showing the model's accuracy over epochs.
- **Emotion Distribution**: Bar plot showing the distribution of emotions in the dataset.

#### For Random Forest:
- **Confusion Matrix**: Displays how well the Random Forest model predicts each emotion.
- **Classification Report**: Provides a detailed breakdown of precision, recall, and F1-scores for each emotion class.
- **Feature Importance**: Highlights the most influential features used in the Random Forest model’s decision-making process.

### Emotion Distribution

The dataset is slightly imbalanced, with some emotions appearing more frequently than others. This distribution is visualized using a bar plot to help understand potential biases in classification.

## How to Run

1. Clone the repository to your local machine or open it in Google Colab.
2. Download the TESS dataset from [Kaggle](https://www.kaggle.com/datasets/ejlok1/toronto-emotional-speech-set-tess) and upload it to your Google Drive.
3. Modify the `base_dir` in the `main()` function to point to the correct directory in your Google Drive where the dataset is stored.
4. Choose to run the CNN or Random Forest model by executing the corresponding code sections.
5. Run the `main()` function to load the data, train the selected model, and visualize the results.

## Libraries Used

### For CNN:
- `os` for file handling
- `librosa` and `librosa.display` for audio processing and spectrogram visualization
- `numpy` for numerical operations
- `matplotlib` and `seaborn` for visualizations (confusion matrix, bar plots, accuracy plots, t-SNE)
- `tensorflow` and `keras` for building and training the CNN
- `sklearn.model_selection` for train-test splitting
- `sklearn.metrics` for generating confusion matrices
- `sklearn.manifold.TSNE` for t-SNE visualizations

### For Random Forest:
- `os` for file handling
- `librosa` for audio processing
- `pandas` for data handling and feature storage
- `numpy` for numerical operations
- `tsfresh` for extracting comprehensive statistical features from audio signals
- `tsfresh.feature_extraction.settings.ComprehensiveFCParameters` for configuring feature extraction
- `sklearn.ensemble.RandomForestClassifier` for building and training the Random Forest
- `sklearn.model_selection` for train-test splitting
- `sklearn.metrics` for generating accuracy scores, confusion matrices, and classification reports
- `matplotlib` and `seaborn` for visualizations (confusion matrix, feature importance)

## Acknowledgements

I would like to thank the University of Toronto for providing the TESS dataset. This high-quality dataset has been instrumental in training robust emotion classifiers.

---

Feel free to explore the code, experiment with different models, or use the dataset for your own emotion classification tasks! 
