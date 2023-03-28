import os
import librosa
import numpy as np
from scipy.spatial.distance import euclidean
import soundfile as sf

# Load the audio file and extract its beats
audio_path = 'file.wav'
y, sr = librosa.load(audio_path)
tempo, beats = librosa.beat.beat_track(y=y, sr=sr)

# Extract the MFCC features for the first beat
beat_audio_path = 'path/to/output/beat_0.wav'
beat_start, beat_end = librosa.frames_to_time(beats[0], sr=sr)
beat_y = y[int(beat_start * sr):int(beat_end * sr)]
librosa.output.write_wav(beat_audio_path, beat_y, sr=sr)
beat_mfccs = librosa.feature.mfcc(y=beat_y, sr=sr)

# Create a list to store the Euclidean distances and file paths of the beat files
distances = [(0, beat_audio_path)]  # Add the first beat file to the list
for i, beat in enumerate(beats[1:], start=1):
    # Extract the MFCC features for the current beat
    beat_audio_path = f'path/to/output/beat_{i}.wav'
    beat_start, beat_end = librosa.frames_to_time(beat, sr=sr)
    beat_y = y[int(beat_start * sr):int(beat_end * sr)]
    librosa.output.write_wav(beat_audio_path, beat_y, sr=sr)
    beat_mfccs = librosa.feature.mfcc(y=beat_y, sr=sr)

    # Calculate the Euclidean distance between the first beat and the current beat
    dist = euclidean(beat_mfccs.flatten(), distances[0][1].flatten())

    # Add the Euclidean distance and file path to the distances list
    distances.append((dist, beat_audio_path))

# Sort the distances list in ascending order of Euclidean distance
distances.sort(key=lambda x: x[0])

# Concatenate the beat files in the order of similarity
output_audio_path = 'output.wav'
output_y = np.concatenate([librosa.load(beat_file)[0] for _, beat_file in distances])
SF.write(output_audio_path, output_y, sr)

# Print the sorted list of beat files in order of similarity
print(f"Beat files similar to {os.path.basename(distances[0][1])}")
for i, (dist, beat_file) in enumerate(distances[1:]):
    print(f"{i+1}. {os.path.basename(beat_file)} - Euclidean distance: {dist:.2f}")
