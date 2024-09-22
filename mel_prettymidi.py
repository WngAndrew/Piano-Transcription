import os
import numpy as np
import librosa
import pretty_midi

#process data with log mel spectrogram and pretty midi for wav and midi files respectively

def adjust_lengths(mel_spectrogram, midi_data, pad_value=-1):
    mel_len = mel_spectrogram.shape[1]
    midi_len = midi_data.shape[1]
    
    if mel_len > midi_len:
        padding = np.full((midi_data.shape[0], mel_len - midi_len), pad_value)
        midi_data = np.concatenate((midi_data, padding), axis=1)
    elif midi_len > mel_len:
        padding = np.full((mel_spectrogram.shape[0], midi_len - mel_len), pad_value)
        mel_spectrogram = np.concatenate((mel_spectrogram, padding), axis=1)
    
    return mel_spectrogram, midi_data

def process_batch(batch_index, batch_folder, save_folder, sr=22050, n_mels=128, hop_length=512, pad_value=-1):
    pairs = []
    for root, dirs, files in os.walk(batch_folder):
        for file in files:
            if file.endswith('.midi'):
                midi_path = os.path.join(root, file)
                wav_path = midi_path.replace('.midi', '.wav')
                if os.path.exists(wav_path):
                    pairs.append((wav_path, midi_path))
    
    mel_spectrograms = []
    midi_data_objects = []

    frame_rate = sr / hop_length

    for wav_path, midi_path in pairs:
        try:
            wav_data, sr = librosa.load(wav_path, sr=sr)
            if len(wav_data) == 0:
                print(f"Empty wav data for file: {wav_path}")
                continue
            
            mel_spectrogram = librosa.feature.melspectrogram(y=wav_data, sr=sr, n_mels=n_mels, hop_length=hop_length)
            log_mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)
            midi_data = pretty_midi.PrettyMIDI(midi_path).get_piano_roll(fs=frame_rate)

            log_mel_spectrogram, midi_data = adjust_lengths(log_mel_spectrogram, midi_data, pad_value)
            
            mel_spectrograms.append(log_mel_spectrogram)
            midi_data_objects.append(midi_data)

            # Verify lengths
            mel_len = log_mel_spectrogram.shape[1]
            midi_len = midi_data.shape[1]
            if mel_len != midi_len:
                print(f"Length mismatch for mel and midi: difference = {mel_len - midi_len}")
                print(f"duration difference = {mel_duration-midi_duration}")

        except Exception as e:
            print(f"Error processing pair (wav: {wav_path}, midi: {midi_path}): {e}")
            continue
    
    np.savez_compressed(os.path.join(save_folder, f'log_mel_spectrograms_batch_{batch_index}.npz'), *mel_spectrograms)
    np.savez_compressed(os.path.join(save_folder, f'midi_data_batch_{batch_index}.npz'), *midi_data_objects)

def main():
    parent_batch_folder = 'D:\\maestro_batches'
    save_folder = 'D:\\processed_maestro_dataset_pre_padding'
    os.makedirs(save_folder, exist_ok=True)
    
    for batch_index in range(43):
        batch_folder = os.path.join(parent_batch_folder, f'batch_{batch_index}')
        process_batch(batch_index, batch_folder, save_folder)
        print(f'Processed batch {batch_index} without padding')

if __name__ == "__main__":
    main()
    