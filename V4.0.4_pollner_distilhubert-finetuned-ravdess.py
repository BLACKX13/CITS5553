# !pip install transformers datasets torchaudio librosa pandas

import os
import librosa
import pandas as pd
import torch
from transformers import Wav2Vec2FeatureExtractor, AutoModelForAudioClassification

# ---------------------------
# Emotion Recognition
# ---------------------------
class PretrainedEmotionModel:
    """
    Use the pre-trained DistilHuBERT model from Hugging Face for emotion classification.
    """
    def __init__(self, model_name):
        # Load the pre-trained DistilHuBERT model and feature extractor
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
        self.model = AutoModelForAudioClassification.from_pretrained(model_name)

    def predict_all_labels(self, audio_file_path):
        """
        Predict all possible emotion labels with their respective confidence scores.
        """
        # Load and preprocess the audio
        speech_array, sampling_rate = librosa.load(audio_file_path, sr=16000)
        inputs = self.feature_extractor(
            speech_array, return_tensors="pt", sampling_rate=sampling_rate, padding=True
        )
        
        # Make predictions
        with torch.no_grad():
            logits = self.model(**inputs).logits
            probabilities = torch.softmax(logits, dim=-1).squeeze().numpy()

        # Map label IDs to emotion labels
        id2label = self.model.config.id2label

        # Collect the predictions and their associated confidence scores
        results = []
        for label_id, confidence in enumerate(probabilities):
            emotion = id2label[label_id]
            results.append({
                'audio_file': os.path.basename(audio_file_path),
                'emotion': emotion.capitalize(),
                'confidence': confidence
            })

        # Sort results by confidence in descending order
        df = pd.DataFrame(results)
        df = df.sort_values(by='confidence', ascending=False).reset_index(drop=True)

        return df


class EmotionPipeline:
    """
    Orchestrates the workflow of downloading audio data from Google Drive and performing emotion classification.
    """
    def __init__(self, model_name="pollner/distilhubert-finetuned-ravdess"):
        self.model_name = model_name
        self.model = PretrainedEmotionModel(self.model_name)

    def load_and_predict(self, audio_file_paths):
        for audio_file_name, audio_file_path in audio_file_paths.items():
            # Perform prediction using the pre-trained model
            result_df = self.model.predict_all_labels(audio_file_path)
            return result_df


# Main function to execute the pipeline
def main():
  # ---------------------------
  # Emotion Recognition
  # ---------------------------
  audio_model_name = "pollner/distilhubert-finetuned-ravdess"

  # Initialize the pipeline with the pre-trained model
  emotion_pipeline = EmotionPipeline(model_name=audio_model_name)

  # Provide local paths to audio files
  audio_file_paths = {
      'audio1.mp3': './audio1.mp3'
      }

  # Run the emotion pipeline with local files
  audio_results_df = emotion_pipeline.load_and_predict(audio_file_paths)

  # Output the audio results
  print("Audio Emotion Results:")
  print(audio_results_df)

if __name__ == "__main__":
    main()
