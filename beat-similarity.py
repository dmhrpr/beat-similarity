import librosa
import numpy as np
import os

# Define the audio path
audio_path = 'path/to/audio/file.wav'

# Define the output path for the beat files
beat_output_path = 'path/to/output/beat/files'

# Define the output path for the concatenated audio file
concat_output_path = 'path/to/output/concatenated/audio/file.wav'

# Load the audio file
y, sr = librosa.load(audio_path)

# Get the tempo and beats
tempo, beats = librosa.beat.beat_track(y=y, sr=sr)

# Create a list to store the beat files
beat_files = []

# Iterate over the beats and extract the corresponding audio
for i, beat in enumerate(beats):
    # Set the start and end times for the beat audio
    start = beat if i == 0 else beats[i-1]
    end = beat

    # Extract the beat audio
    beat_audio = y[start:end]

    # Get the MFCCs for the beat audio
    mfccs = librosa.feature.mfcc(beat_audio, sr=sr)

    # Save the MFCCs to a file
    beat_file_path = os.path.join(beat_output_path, f'beat_{i}.npy')
    np.save(beat_file_path, mfccs)

    # Add the beat file path to the list
    beat_files.append(beat_file_path)

# Create a list to store the ordered beat files
ordered_beat_files = []

# Add the first beat file to the ordered beat files
ordered_beat_files.append(beat_files[0])

# Remove the first beat file from the list of available beat files
available_beat_files = beat_files[1:]

# Iterate over the remaining beat files and add them in order of similarity
while len(available_beat_files) > 0:
    # Load the MFCCs for the current beat file
    current_mfccs = np.load(ordered_beat_files[-1])

    # Calculate the distances between the current beat file and the available beat files
    distances = []
    for beat_file in available_beat_files:
        mfccs = np.load(beat_file)
        distance = np.linalg.norm(current_mfccs - mfccs)
        distances.append(distance)

    # Get the index of the beat file with the minimum distance
    min_index = np.argmin(distances)

    # Add the beat file with the minimum distance to the ordered beat files
    ordered_beat_files.append(available_beat_files[min_index])

    # Remove the beat file with the minimum distance from the list of available beat files
    available_beat_files.pop(min_index)

# Concatenate the ordered beat files into a single audio file
concatenated_audio = None
for beat_file in ordered_beat_files:
    mfccs = np.load(beat_file)
    beat_audio = librosa.feature.inverse.mfcc_to_audio(mfccs, sr=sr)
    if concatenated_audio is None:
        concatenated_audio = beat_audio
    else:
        concatenated_audio = np.concatenate((concatenated_audio, beat_audio))

# Save the concatenated audio to a file
librosa.output.write_wav(concat_output_path, concatenated_audio, sr=sr)
