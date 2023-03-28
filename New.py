import librosa
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances

# Load audio file
audio_file = 'example_audio.wav'
y, sr = librosa.load(audio_file)

# Calculate tempo and beats
tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
beat_times = librosa.frames_to_time(beats, sr=sr)

# Extract MFCCs for each beat
mfccs = []
for i, beat in enumerate(beats[:-1]):
    start = beat
    end = beats[i+1]
    mfcc = librosa.feature.mfcc(y[start:end], sr=sr, n_mfcc=13)
    mfccs.append(np.ravel(mfcc.T))

# Calculate similarity between each beat and the first beat
distances = euclidean_distances([mfccs[0]], mfccs)
order = np.argsort(distances)

# Reorder beat array based on similarity
new_order = [0]
for i in range(len(beats)-2):
    current = new_order[-1]
    remaining = [x for x in range(len(beats)-1) if x not in new_order]
    similarity = order[current][remaining]
    similarity_flat = np.ravel(similarity)
    next_beat = remaining[np.argmin(similarity_flat)]
    new_order.append(next_beat)

# Concatenate beats in new order
new_beats = np.concatenate([beats[i:i+1] for i in new_order])

# Save new audio file
y_new = np.concatenate([y[new_beats[i]:new_beats[i+1]] for i in range(len(new_beats)-1)])
librosa.output.write_wav('example_audio_reordered.wav', y_new, sr)
