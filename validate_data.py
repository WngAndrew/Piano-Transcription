import numpy as np
import matplotlib.pyplot as plt

#verify our procesed and padded data is correctly formatted, and ready to be handled in colab

# Load the npz file with allow_pickle=True
file_path = 'D:\\processed_maestro_dataset_padded\\padded_training_mel.npz'
data = np.load(file_path, allow_pickle=True)

# Print the keys in the npz file
print("Keys in the npz file:", data.keys())

# Access and print each array
for key in data:
    print(f"\nKey: {key}")
    array = data[key]

    if isinstance(array, np.ndarray) and array.dtype == 'object':
        # Convert object array to a regular float array if necessary
        array = np.array([np.asarray(item) for item in array], dtype=np.float32)

    print(f"Shape: {array.shape}")
    # Print a small part of the array to avoid large output
    print(array[:5, :5])  # Print the first 5x5 part of the array

    # Mask the padding values (-999) for visualization
    masked_array = np.ma.masked_where(array == -999, array)

    # Visualize the mel spectrogram if the array is suitable for plotting
    if array.ndim == 2:
        plt.figure(figsize=(10, 4))
        plt.imshow(masked_array, aspect='auto', origin='lower', cmap='viridis')
        plt.colorbar(format='%+2.0f dB')
        plt.title(f"Mel Spectrogram: {key}")
        plt.xlabel('Time')
        plt.ylabel('Mel Frequency Bands')
        plt.show()
