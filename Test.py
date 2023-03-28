import wave
import struct
import numpy as np
from scipy.fftpack import fft
from scipy.signal import find_peaks

def get_onset_times(audio_data, sample_rate):
    # Compute onset strength envelope
    hop_length = 512
    window = np.hanning(hop_length)
    audio_data = audio_data / np.max(np.abs(audio_data))
    audio_data = np.pad(audio_data, int(hop_length // 2), mode='reflect')
    audio_data = np.abs(np.lib.stride_tricks.as_strided(audio_data, shape=(len(audio_data) - hop_length + 1, hop_length), strides=(audio_data.itemsize, audio_data.itemsize)))
    audio_data = audio_data.T * window
    audio_data = np.sqrt(audio_data.sum(axis=0))
    audio_data /= audio_data.max()

    # Compute onset times
    onset_threshold = 0.3
    onset_times, _ = find_peaks(audio_data, height=onset_threshold, distance=2 * sample_rate / hop_length)

    return onset_times * hop_length / sample_rate

def get_mfcc_features(data, frame_rate):
    n_fft = 2048
    hop_length = 512
    num_mfcc = 13

    # Compute the Mel filterbank
    low_freq_mel = 0
    high_freq_mel = (2595 * np.log10(1 + (frame_rate / 2) / 700))  # Convert Hz to Mel
    mel_points = np.linspace(low_freq_mel, high_freq_mel, num_mfcc + 2)  # Equally spaced in Mel scale
    hz_points = (700 * (10**(mel_points / 2595) - 1))  # Convert Mel to Hz
    bin = np.floor((n_fft + 1) * hz_points / frame_rate)

    fbank = np.zeros((num_mfcc, int(np.floor(n_fft / 2 + 1))))
    for m in range(1, num_mfcc + 1):
        f_m_minus = int(bin[m - 1])   # left
        f_m = int(bin[m])             # center
        f_m_plus = int(bin[m + 1])    # right

        for k in range(f_m_minus, f_m):
            fbank[m - 1, k] = (k - bin[m - 1]) / (bin[m] - bin[m - 1])
        for k in range(f_m, f_m_plus):
            fbank[m - 1, k] = (bin[m + 1] - k) / (bin[m + 1] - bin[m])

    # Compute MFCCs
    mfccs = np.zeros((num_mfcc, len(data)))
    for i in range(0, len(data), hop_length):
        window = np.hamming(n_fft) * data[i:i + n_fft]
        spectrum = np.fft.rfft(window)
        spectrum[spectrum <= 0] = np.finfo(float).eps
        mel_spectrum = np.dot(fbank, np.abs(spectrum) ** 2)
        log_mel_spectrum = np.log(mel_spectrum)
        dct_spectrum = scipy.fftpack.dct(log_mel_spectrum, type=2)
        mfccs[:, i:i + hop_length] = dct_spectrum[1:num_mfcc + 1, :]
    
    return mfccs


def get_distance(feature_vector1, feature_vector2):
    distance = np.sqrt(np.sum(np.square(feature_vector1 - feature_vector2)))
    return distance


def main():
    # Load audio file
    file_name = "audio_file.wav"
    with wave.open(file_name, 'rb') as wave_file:
        frame_rate = wave_file.getframerate()
        num_frames = wave_file.getnframes()
        num_channels = wave_file.getnchannels()
        sample_width = wave_file.getsampwidth()

        # Read audio data
        data = wave_file.readframes(num_frames)
        samples = np.frombuffer(data, dtype=get_numpy_format(sample_width, num_channels))
        samples = samples.reshape(-1, num_channels)

        # Get onset times
        onset_times = get_onset_times(samples[:, 0], frame_rate)

        # Get MFCC features
        feature_vectors = []
        for i in range(len(onset_times)):
            onset_sample = int(onset_times[i] * frame_rate)
            offset_sample = int(onset_times[i+1] * frame_rate) if i < len(onset_times)-1 else None
            segment = samples[onset_sample:offset_sample, 0]
            feature_vectors.append(get_mfcc_features(segment, frame_rate))

        # Get distances
        distances = []
        for i in range(1, len(feature_vectors)):
            distance = get_distance(feature_vectors[0], feature_vectors[i])
            distances.append((distance, i))

        # Sort by distance
        distances.sort(key=lambda x: x[0])

        # Concatenate segments in order
        output = np.concatenate((samples[int(onset_times[0]*frame_rate):], samples[int(onset_times[distances[0][1]]*frame_rate):]))
        for i in range(1, len(distances)):
            output = np.concatenate((output, samples[int(onset_times[distances[i][1]]*frame_rate):]))

        # Write output to file
        with wave.open("output.wav", 'wb') as output_file:
            output_file.setnchannels(num_channels)
            output_file.setsampwidth(sample_width)
            output_file.setframerate(frame_rate)
            output_file.writeframes(output.tobytes())

if __name__ == "__main__":
    main()
