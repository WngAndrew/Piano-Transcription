import numpy as np
import matplotlib.pyplot as plt

#vizualize what the data looks like after being processed with a log mel spectrogram and pretty midi
#displays what the first song's mel spectrogram and midi data looks like

def visualize_mel_spectrogram(file_path, key):
    data = np.load(file_path)
    mel_spectrogram = data[key]

    plt.figure(figsize=(10, 4))
    plt.imshow(mel_spectrogram, aspect='auto', origin='lower')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel Spectrogram')
    plt.xlabel('Time')
    plt.ylabel('Mel Frequency Bands')
    plt.show()

def visualize_midi_piano_roll(file_path, key):
    data = np.load(file_path)
    midi_data = data[key]

    plt.figure(figsize=(10, 4))
    plt.imshow(midi_data, aspect='auto', origin='lower', cmap='gray_r')
    plt.colorbar()
    plt.title('MIDI Piano Roll')
    plt.xlabel('Time')
    plt.ylabel('MIDI Note Number')
    plt.show()

def main():
    # Adjust these paths and keys as necessary
    mel_spectrogram_file = 'D:\\processed_maestro_dataset_pre_padding\\log_mel_spectrograms_batch_0.npz'
    midi_data_file = 'D:\\processed_maestro_dataset_pre_padding\\midi_data_batch_0.npz'
    mel_spectrogram_key = 'arr_0'  # Key of the array to visualize
    midi_data_key = 'arr_0'  # Key of the array to visualize

    visualize_mel_spectrogram(mel_spectrogram_file, mel_spectrogram_key)
    visualize_midi_piano_roll(midi_data_file, midi_data_key)

if __name__ == "__main__":
    main()
