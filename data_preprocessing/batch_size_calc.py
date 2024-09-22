import os
import random

def estimate_average_pair_size(root_dir, sample_size=10):
    midi_files = []
    wav_files = []
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith('.midi'):
                midi_files.append(os.path.join(root, file))
            elif file.endswith('.wav'):
                wav_files.append(os.path.join(root, file))
    
    print(f"Found {len(midi_files)} MIDI files and {len(wav_files)} WAV files.")
    
    if len(midi_files) == 0 or len(wav_files) == 0:
        print("Error: No MIDI or WAV files found in the specified directory.")
        return 0

    sample_midi_files = random.sample(midi_files, min(sample_size, len(midi_files)))
    sample_wav_files = random.sample(wav_files, min(sample_size, len(wav_files)))
    
    total_midi_size = sum(os.path.getsize(f) for f in sample_midi_files)
    total_wav_size = sum(os.path.getsize(f) for f in sample_wav_files)
    
    average_midi_size = total_midi_size / len(sample_midi_files)
    average_wav_size = total_wav_size / len(sample_wav_files)
    
    average_pair_size = average_midi_size + average_wav_size
    return average_pair_size

# Directory path
root_dir = 'D:\\maestro-v3.0.0\\maestro-v3.0.0'

# Print directory structure to verify
print("Directory structure:")
for root, dirs, files in os.walk(root_dir):
    level = root.replace(root_dir, '').count(os.sep)
    indent = ' ' * 4 * (level)
    print('{}{}/'.format(indent, os.path.basename(root)))
    subindent = ' ' * 4 * (level + 1)
    for f in files:
        print('{}{}'.format(subindent, f))

# Estimate average pair size
average_pair_size = estimate_average_pair_size(root_dir)
if average_pair_size > 0:
    average_pair_size_mb = average_pair_size / (1024 * 1024)  # Convert to MB
    print(f"Estimated average pair size: {average_pair_size_mb:.2f} MB")
else:
    print("Unable to estimate average pair size due to missing files.")



