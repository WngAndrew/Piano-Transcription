import os
import random
import shutil


def load_dataset_info(root_dir):
    pairs = []
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith('.midi'):
                midi_path = os.path.join(root, file)
                wav_path = midi_path.replace('.midi', '.wav')
                if os.path.exists(wav_path):
                    pairs.append((midi_path, wav_path))
    return pairs

def shuffle_and_save_chunk(data, chunk_index, output_dir):
    chunk_dir = os.path.join(output_dir, f'chunk_{chunk_index}')
    os.makedirs(chunk_dir, exist_ok=True)
    for midi_path, wav_path in data:
        shutil.copy(midi_path, chunk_dir)
        shutil.copy(wav_path, chunk_dir)
    print(f'Shuffled and saved chunk {chunk_index + 1}')

# Main script
root_dir = 'D:\\maestro-v3.0.0\\maestro-v3.0.0'
chunk_size = 50
output_chunk_dir = 'D:\\maestro_chunks'

# Load dataset info
dataset = load_dataset_info(root_dir)

# Shuffle the dataset
random.shuffle(dataset)

# Split and save in chunks
for chunk_index in range(0, len(dataset), chunk_size):
    chunk = dataset[chunk_index:chunk_index + chunk_size]
    shuffle_and_save_chunk(chunk, chunk_index // chunk_size, output_chunk_dir)
