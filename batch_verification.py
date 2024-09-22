import os

def count_missing_pairs(batch_directory):
    wav_files = {f.replace('.wav', '') for f in os.listdir(batch_directory) if f.endswith('.wav')}
    midi_files = {f.replace('.midi', '') for f in os.listdir(batch_directory) if f.endswith('.midi')}
    
    missing_pairs = 0
    
    for wav_file in wav_files:
        if wav_file not in midi_files:
            missing_pairs += 1
    
    for midi_file in midi_files:
        if midi_file not in wav_files:
            missing_pairs += 1
    
    return missing_pairs

def check_all_batches(base_directory):
    batch_directories = [os.path.join(base_directory, d) for d in os.listdir(base_directory) if os.path.isdir(os.path.join(base_directory, d))]
    
    batch_missing_pairs = {}
    
    for batch_dir in batch_directories:
        missing_pairs = count_missing_pairs(batch_dir)
        batch_missing_pairs[os.path.basename(batch_dir)] = missing_pairs
    
    return batch_missing_pairs

# Example usage
base_directory = 'D:\\maestro_batches'  # Replace with your actual base directory path
batch_missing_pairs = check_all_batches(base_directory)

for batch, missing_pairs in batch_missing_pairs.items():
    print(f"{batch}: {missing_pairs} missing pairs")
