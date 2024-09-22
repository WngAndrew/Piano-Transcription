# Piano Transcription Using CRNNs

This project aims to transcribe piano songs from audio data into sheet music using deep learning techniques. The transcription pipeline converts WAV audio files into mel spectrograms, which are then passed in as input data to a Convolutional Recurrent Neural Network (CRNN). The CRNN architecture combines convolutional layers to capture spatial features (e.g., pitch and timbre) with LSTM layers to capture the temporal dependencies in the music. The final model predicts MIDI information for pitch, velocity, and note timing.

## Data Preprocessing
1. **WAV to Mel Spectrogram**: Each audio file is converted into a mel spectrogram to represent the frequency domain visually, which is used as the input for the neural network.
2. **MIDI to PrettyMIDI**: Corresponding MIDI files are processed using PrettyMIDI for proper alignment with the audio data. This MIDI data represents the ground truth for training the model.
3. **Shuffling and Batching**: The dataset consists of over 200 hours of piano music, shuffled and split into manageable batches. The large batches (3GB each) are further divided into smaller chunks (100-500MB) for training in Google Colab, avoiding memory limitations.
4. **Padding and Normalization**: To ensure uniformity across the dataset, padding is applied based on the 80th percentile of song length, while the data is normalized for consistency during training.

## Model Architecture
The model is built around a CRNN architecture:
- **Convolutional Layers (CNN)**: Used to extract local audio features like pitch and timbre from the mel spectrogram. The CNN architecture includes three convolutional layers with max-pooling to reduce the temporal dimensions while capturing relevant audio features.
- **LSTM Layers (RNN)**: Two LSTM layers are used to capture the sequential nature of music and predict note events over time, including pitch, note onset/offset, and velocity.
- **Output Layer**: The model outputs MIDI predictions, which are further processed to generate sheet music using the `music21` library. This includes predicting pitch, velocity, and timing for each note in the sequence.

## Challenges and Considerations
- **Batch Size**: Optimized the batch sizes based on memory limitations in Colab, balancing efficiency and storage constraints. This is by far the most significant challenge that is still a limitation on the project's completion.
- **Loss of Temporal Features**: Considered the potential downside of aggressive max-pooling, which could reduce the temporal granularity of the spectrogram, but managed this by using asymmetric pooling and fine-tuning LSTM layers.
- **CRNN Architecture**: Chose CRNN over pure CNN models for its ability to handle the sequential nature of music, with potential improvements from tuning layer configurations and experimenting with different architectures like GRUs and LSTMs.

## Next Steps
- Finalize the architecture with proper tuning of convolutional filters and LSTM units.
- Test on real-world piano data and use `music21` to generate sheet music predictions.
- Optimize the model's performance by experimenting with different hyperparameters, architectures, and data augmentation techniques.
- Build out a UI for users to upload .wav files of their piano music, which will be converted to mel spectrogram data for the model to generate a MIDI prediction.
- The UI will then automatically feed the prediction to music21 which will produce sheet music for the users. 
