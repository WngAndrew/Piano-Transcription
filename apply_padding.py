import numpy as np
import os

"""
Calculates length to truncate data to, then splits the dataset accordingly
downsamples midi data and pads mel data
then normalizes it all
"""

def downsample_midi(midi_data, target_length=512):
    original_length = midi_data.shape[1]
    downsampled_data = np.zeros((128, target_length))
    for note in range(128):
        downsampled_data[note] = np.interp(
            np.linspace(0, original_length, target_length),
            np.arange(original_length),
            midi_data[note]
        )
    return downsampled_data

def collect_lengths(save_folder, num_batches):
    lengths = []
    for batch_index in range(num_batches):
        mel_file = os.path.join(save_folder, f'log_mel_spectrograms_batch_{batch_index}.npz')
        mel_data = np.load(mel_file, allow_pickle=True)
        lengths.extend([mel_data[key].shape[1] for key in mel_data])
    return lengths

def calculate_percentile(lengths, percentile=95):
    return int(np.percentile(lengths, percentile))

def normalize_mel(mel, mean, std):
    return (mel - mean) / std

def normalize_midi(midi):
    return midi / 127.0

def calculate_mel_stats(save_folder, num_batches):
    mel_files = [os.path.join(save_folder, f'log_mel_spectrograms_batch_{batch_index}.npz') for batch_index in range(num_batches)]
    all_mel_data = []

    for mel_file in mel_files:
        mel_data = np.load(mel_file, allow_pickle=True)
        for key in mel_data:
            all_mel_data.append(mel_data[key])

    all_mel_data = np.concatenate(all_mel_data, axis=1)
    mean = np.mean(all_mel_data)
    std = np.std(all_mel_data)
    return mean, std

save_folder = 'D:\\processed_maestro_dataset_pre_padding'
num_batches = 15
lengths = collect_lengths(save_folder, num_batches)
fixed_length = 32768
print(f'Fixed length (80th percentile): {fixed_length}')

mean, std = calculate_mel_stats(save_folder, num_batches)

def pad_sequences(sequences, max_length, pad_value=-999):
    padded_sequences = []
    for seq in sequences:
        if seq.shape[1] <= max_length:
            padding = np.full((seq.shape[0], max_length - seq.shape[1]), pad_value)
            padded_seq = np.concatenate((seq, padding), axis=1)
        padded_sequences.append(padded_seq)
    return np.array(padded_sequences, dtype=object)

def save_chunked_data(data, chunk_size, output_folder, base_filename):
    num_chunks = len(data) // chunk_size + (1 if len(data) % chunk_size != 0 else 0)
    for i in range(num_chunks):
        chunk = data[i * chunk_size:(i + 1) * chunk_size]
        np.savez_compressed(os.path.join(output_folder, f"{base_filename}_chunk_{i}.npz"), *chunk)

def filter_and_save_songs(input_folder, num_batches, percentile_length, mean, std, output_folder, chunk_size=100):
    training_mel_files = []
    training_midi_files = []
    testing_mel_files = []
    testing_midi_files = []

    training_chunk_count = 0
    testing_chunk_count = 0

    for batch_index in range(num_batches):
        mel_file = os.path.join(input_folder, f'log_mel_spectrograms_batch_{batch_index}.npz')
        midi_file = os.path.join(input_folder, f'midi_data_batch_{batch_index}.npz')

        mel_data = np.load(mel_file, allow_pickle=True)
        midi_data = np.load(midi_file, allow_pickle=True)

        for key in mel_data:
            if mel_data[key].shape[1] <= percentile_length:
                normalized_mel = normalize_mel(mel_data[key], mean, std)
                padded_mel = pad_sequences([normalized_mel], fixed_length)
                training_mel_files.append(padded_mel[0])

                downsampled_midi = downsample_midi(midi_data[key])
                normalized_midi = normalize_midi(downsampled_midi)
                training_midi_files.append(normalized_midi)

                if len(training_mel_files) >= chunk_size:
                    save_chunked_data(training_mel_files, chunk_size, output_folder, f'training_mel_{training_chunk_count}')
                    save_chunked_data(training_midi_files, chunk_size, output_folder, f'training_midi_{training_chunk_count}')
                    training_mel_files = []
                    training_midi_files = []
                    training_chunk_count += 1
            else:
                normalized_mel = normalize_mel(mel_data[key], mean, std)
                testing_mel_files.append(normalized_mel)

                normalized_midi = normalize_midi(midi_data[key])
                testing_midi_files.append(normalized_midi)

                if len(testing_mel_files) >= chunk_size:
                    save_chunked_data(testing_mel_files, chunk_size, output_folder, f'testing_mel_{testing_chunk_count}')
                    save_chunked_data(testing_midi_files, chunk_size, output_folder, f'testing_midi_{testing_chunk_count}')
                    testing_mel_files = []
                    testing_midi_files = []
                    testing_chunk_count += 1

    if training_mel_files:
        save_chunked_data(training_mel_files, chunk_size, output_folder, f'training_mel_{training_chunk_count}')
        save_chunked_data(training_midi_files, chunk_size, output_folder, f'training_midi_{training_chunk_count}')

    if testing_mel_files:
        save_chunked_data(testing_mel_files, chunk_size, output_folder, f'testing_mel_{testing_chunk_count}')
        save_chunked_data(testing_midi_files, chunk_size, output_folder, f'testing_midi_{testing_chunk_count}')

# Main script
output_folder = 'D:\\processed_maestro_dataset_padded'
os.makedirs(output_folder, exist_ok=True)

filter_and_save_songs(save_folder, num_batches, fixed_length, mean, std, output_folder)
