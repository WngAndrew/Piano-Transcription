import os
import random
import shutil

def load_dataset_info(chunk_dir, pairs):
    for root, dirs, files in os.walk(chunk_dir):
        for file in files:
            if file.endswith('.midi'):
                midi_path = os.path.join(root, file)
                wav_path = midi_path.replace('.midi', '.wav')
                if os.path.exists(wav_path):
                    pairs.append((midi_path, wav_path))
    return pairs


def merge_chunks(output_dir, final_dir, batch_size):
    chunk_dirs = [os.path.join(output_dir, d) for d in os.listdir(output_dir) if d.startswith('chunk_')]
    all_files = []
    
    for chunk_dir in chunk_dirs:
        load_dataset_info(chunk_dir, all_files)
    
    random.shuffle(all_files)
    batches = list(create_batches(all_files, batch_size))
    
    os.makedirs(final_dir, exist_ok=True)
    for batch_index, batch in enumerate(batches):
        batch_dir = os.path.join(final_dir, f'batch_{batch_index}')
        os.makedirs(batch_dir, exist_ok=True)
        for midi_path, wav_path in batch:
            shutil.copy(midi_path, batch_dir)
            shutil.copy(wav_path, batch_dir)
        print(f'Processed batch {batch_index + 1}/{len(batches)}')

def create_batches(data, batch_size):
    for i in range(0, len(data), batch_size):
        yield data[i:i + batch_size]

# Directory paths
output_chunk_dir = 'D:\\maestro_chunks'  # Directory where initial chunks are stored
final_batch_dir = 'D:\\maestro_batches'  # Directory where final batches will be stored
batch_size = 30  # Number of pairs per batch, calculated using average file size needed to reach 3gb batches

# Merge chunks and create final batches
merge_chunks(output_chunk_dir, final_batch_dir, batch_size)
