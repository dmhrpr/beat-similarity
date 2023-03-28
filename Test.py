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
        data = wave_file.readframes(num_frames)
        data = np.array(struct.unpack('{n}h'.format(n=num_frames), data))
    
    # Get onset times
    onset_times = get_onset_times(data, frame_rate)

    # Split audio file into beats
    beats = []
    for i in range(len(onset_times)-1):
        start = int(onset_times[i] * frame_rate)
        end = int(onset_times[i+1] * frame_rate)
        beat = data[start:end]
        beats.append(beat)

    # Compute MFCC features for each beat
    mfccs = []
    for i in range(len(beats)):
        mfcc = get_mfcc_features(beats[i], frame_rate)
        mfccs.append(mfcc)
    
    # Calculate distance between each beat and the first beat
    distances = []
    for i in range(len(mfccs)):
        distance = get_distance(mfccs[0], mfccs[i])
        distances.append(distance)
    
    # Sort beats by their similarity to the first beat
    sorted_indices = np.argsort(distances)
    sorted_beats = [beats[i] for i in sorted_indices]

    # Concatenate beats in order of similarity to the first beat
    output_data = np.concatenate(sorted_beats)
    
    # Save the concatenated audio file
    output_file_name = "output_audio_file.wav"
    with wave.open(output_file_name, 'wb') as output_file:
        output_file.setnchannels(1)
        output_file.setsampwidth(2)
        output_file.setframerate(frame_rate)
        output_file.writeframes(output_data.astype(np.int16))

main()
