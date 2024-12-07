import yaml 
import librosa
import numpy as np
np.set_printoptions(suppress=True)
from scipy import signal
from scipy.spatial.distance import pdist, squareform, euclidean
from scipy.stats import zscore
from scipy.ndimage import uniform_filter1d
import soundfile as sf
import os
import json

from plotters import *


# PATHS
dir_path = os.path.dirname(os.path.realpath(__file__))
AUDIO_FOLDER = os.path.join(dir_path, 'audio_folder') 
ANALYSIS_FOLDER = os.path.join(dir_path, 'analysis_folder')
CONFIG_FILE = os.path.join(dir_path, 'config.yaml')


with open(CONFIG_FILE, 'r') as params_config:
    config = yaml.safe_load(params_config)

SR = config['target_sr']
N_FFT = config['n_fft']
HOP_LEN = config['hop_length']
SMOOTH_WL = config['smooth_wl']
MIN_DIST = config['min_dist']
THRESH = config['thresh']
STACK_MEM = config['stack_mem']
PLOT_FEAT = config['plot_feat']
PLOT_SPEC = config['plot_spec']



# LOADERS
def load_audio(file_path, norm=True):
    y, sr = librosa.load(file_path, mono=True)
    if sr != SR:
        y = librosa.core.resample(y, orig_sr=sr, target_sr=SR)
        sr = SR
    if norm: 
        y = librosa.util.normalize(y)
    return y, sr


def extract_stft(y, n_fft=4096, hop_length=2048):
    return np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))


def harmonic_percussive(stft):
    harmonic, percussive = librosa.decompose.hpss(stft)
    return harmonic, percussive



# FEATURES EXTRACTORS

def get_chromagram(stft, sr):
    chroma = librosa.feature.chroma_stft(S=stft, sr=sr, n_fft=N_FFT, hop_length=HOP_LEN)
    return chroma


def get_mfccs(stft, sr, n_mfcc=13):
    mfccs = librosa.feature.mfcc(S=stft, sr=sr, n_mfcc=n_mfcc)
    mfccs = mfccs + (min(mfccs.flatten()) * -1)
    mfccs = np.sqrt(mfccs)
    mfccs_norm = (mfccs - np.min(mfccs)) / (np.max(mfccs) - np.min(mfccs) + 1e-10)
    return mfccs_norm


def get_tempogram(stft, sr):
    oenv = librosa.onset.onset_strength(S=stft, sr=sr, hop_length=HOP_LEN)
    tempogram = librosa.feature.tempogram(onset_envelope=oenv, sr=sr, hop_length=HOP_LEN, win_length=25) #100 o default 384
    tempogram = tempogram[1:, :]
    return tempogram


def get_energy(y):
    energy = librosa.feature.rms(y=y, hop_length=HOP_LEN)[0]
    energy = np.convolve(energy, np.ones(7)/7, mode='same')
    energy_norm = (energy - np.min(energy)) / (np.max(energy) - np.min(energy))
    return energy_norm



# SELF-SIMILARITY MATRIX

def stack_memory(feature, n_steps=3):
    return librosa.feature.stack_memory(feature, n_steps=n_steps)


def get_SSM(feature, k=None):
    
    feature = zscore(feature, axis=1)
    distance_matrix = squareform(pdist(feature.T, metric='euclidean'))
    sigma = np.median(distance_matrix)
    SSM = np.exp(-distance_matrix**2 / (2 * sigma**2))

    if k is not None:
        k = min(k, feature.shape[1] - 1)  
        for i in range(SSM.shape[0]):
            row = SSM[i]
            sorted_indices = np.argsort(row)[::-1]
            row[sorted_indices[k+1:]] = 0

    return SSM



# NOVELTY FUNCTION

def get_novelty_function(recurrence_matrix, smoothing_passes=1):
    lag_matrix = librosa.segment.recurrence_to_lag(recurrence_matrix, pad=None)

    s1 = round(3*SR/HOP_LEN) 
    s2 = round(15*SR/HOP_LEN)
    pad_width = max(s1, s2) 
    sigma1 = (s1 - 1) / 2 
    sigma2 = (s2 - 1) / 3 
    gaussian_small = signal.windows.gaussian(s1, std=sigma1).reshape(s1, 1)
    gaussian_large = signal.windows.gaussian(s2, std=sigma2).reshape(s2, 1)
    gaussian_kernel = np.matmul(gaussian_small, gaussian_large.T)

    for i in range(smoothing_passes):
        lag_padded = np.pad(lag_matrix, ((0, 0), (pad_width, pad_width)), mode='constant')
        smoothed_pad = signal.convolve2d(lag_padded, gaussian_kernel, mode='same')
        lag_matrix = smoothed_pad[:, pad_width:-pad_width]

    novelty = np.linalg.norm(lag_matrix[:, 1:] - lag_matrix[:, :-1], axis=0)

    zero_array = np.zeros(pad_width)
    novelty[:pad_width] = zero_array
    novelty[-pad_width:] = zero_array
    return novelty


def smooth_novelty(novelty_function, window_length=50, exp=2):
    silence_threshold = 0.2
    novelty_function[novelty_function < silence_threshold] = 0
    smoothed_novelty = np.convolve(novelty_function, np.ones(window_length)/window_length, mode='same') ** exp
    return smoothed_novelty


def normalize_novelty(novelty_function):
    return (novelty_function - novelty_function.min()) / (novelty_function.max() - novelty_function.min())


def find_boundaries(peaks, duration):
        boundaries = np.concatenate(([0], peaks, [duration]))
        return boundaries



# PEAK DETECTION

def peak_detection(novelty_function, window_length=200, threshold=0.5):
    peaks_position = signal.find_peaks(novelty_function, height=threshold, distance=window_length*2, width=round(0.5*SR/HOP_LEN))[0]
    peaks_values = signal.find_peaks(novelty_function, height=threshold, distance=window_length*2, width=round(0.5*SR/HOP_LEN))[1]['peak_heights']
    return peaks_position, peaks_values


def merge_near_peaks(peaks, peak_values, window_length):
    sorted_indices = np.argsort(peaks)
    peaks = peaks[sorted_indices]
    peak_values = peak_values[sorted_indices]

    merged_peaks = []
    current_group = [peaks[0]]
    current_values = [peak_values[0]]

    for peak, value in zip(peaks[1:], peak_values[1:]):
        if peak - current_group[-1] <= window_length*2:
            current_group.append(peak)
            current_values.append(value)
        else:
            most_prominent_peak = current_group[np.argmax(current_values)]
            merged_peaks.append(most_prominent_peak)
            current_group = [peak]
            current_values = [value]

    if current_group:
        most_prominent_peak = current_group[np.argmax(current_values)]
        merged_peaks.append(most_prominent_peak)

    merged_peaks = np.array(merged_peaks)
    return np.array(merged_peaks)



# SEGMENTATION

def segment_and_mean(feature, boundaries, similarity_threshold=0.05):
    segment_means = []
    merged_boundaries = []

    for start, end in zip(boundaries[:-1], boundaries[1:]):
        segment = feature[:, start:end]
        segment_mean = np.mean(segment, axis=1)
        segment_means.append(segment_mean)
        merged_boundaries.append((start, end))

    i = 0
    while i < len(segment_means) - 1:
        current_mean = segment_means[i]
        next_mean = segment_means[i + 1]
        similarity = euclidean(current_mean, next_mean)

        if similarity <= similarity_threshold:
            merged_start = merged_boundaries[i][0]
            merged_end = merged_boundaries[i + 1][1]
            merged_segment = feature[:, merged_start:merged_end]
            merged_mean = np.mean(merged_segment, axis=1)

            segment_means[i] = merged_mean
            merged_boundaries[i] = (merged_start, merged_end)
            del segment_means[i + 1]
            del merged_boundaries[i + 1]
        else:
            i += 1

    unique_peaks = sorted(set([start for start, end in merged_boundaries] + [merged_boundaries[-1][1]]))
    segment_means = np.expand_dims(segment_means, axis=1)
    return segment_means, unique_peaks


def segment_and_mean_no_bound(feature, boundaries):
    segment_means = []

    for start, end in zip(boundaries[:-1], boundaries[1:]):
        segment = feature[:, start:end]
        segment_mean = np.mean(segment, axis=1)
        segment_means.append(segment_mean)
    
    segment_means = np.expand_dims(segment_means, axis=1)
    return segment_means


def compute_segment_similarities(segment_pcas):
    flattened_pcas = [pca.flatten() for pca in segment_pcas]
    euclidean_distances = pdist(flattened_pcas, metric='euclidean')
    distance_matrix = squareform(euclidean_distances)
    
    max_distance = np.max(distance_matrix)
    min_distance = np.min(distance_matrix)
    normalized_distances = (distance_matrix - min_distance) / (max_distance - min_distance)
    
    similarity_matrix = 1 - normalized_distances
    
    return similarity_matrix


def local_normalize_novelty(novelty_function):
    normalized_novelty = novelty_function
    for i in range(3, 0, -1):
        w = novelty_function.shape[0] // 2**i
        local_mean = uniform_filter1d(normalized_novelty, size=w)
        local_std = np.sqrt(uniform_filter1d((normalized_novelty - local_mean)**2, size=w))
        normalized_novelty = (normalized_novelty - local_mean) / (local_std + 1e-6)

    normalized_novelty = (normalized_novelty - np.min(normalized_novelty)) / (np.max(normalized_novelty) - np.min(normalized_novelty))   
    return normalized_novelty



# CLICKS AUDIO TRACK

def create_clicks_audio(y, peaks, save_path):
    click_duration_ms = 50
    click_duration_samples = int(SR * click_duration_ms / 1000)
    click = np.sin(2 * np.pi * 1000 * np.arange(click_duration_samples) / SR)

    right_channel = np.zeros_like(y)

    for peak in peaks:
        peak_sample = int(peak * HOP_LEN)
        if peak_sample + click_duration_samples < len(right_channel):
            right_channel[peak_sample:peak_sample + click_duration_samples] += click

    right_channel = np.clip(right_channel, -1, 1)
    stereo_audio = np.vstack((y, right_channel)).T

    sf.write(f"{save_path}/audio_plus_clicks.mp3", stereo_audio, SR, format='MP3')
    


# JSON

def save_json(file, peaks_ch, peaks_mf, peaks_temp, peaks_en, merged_peaks, save_path):

    piece_name = file[:-4]

    data = {
        'name': piece_name,
        'peaks_ch': peaks_ch.tolist(),
        'peaks_mf': peaks_mf.tolist(),
        'peaks_temp': peaks_temp.tolist(),
        'peaks_en': peaks_en.tolist(),
        'merged_peaks': merged_peaks.tolist()
    }

    json_file_path = os.path.join(save_path, "data.json")
    with open(json_file_path, 'w') as json_file:
        json.dump(data, json_file, indent=4)

    

# EXTRACTOR 

def extractors(piece_path, save_path, file, 
               plot_features=PLOT_FEAT, 
               plot_extracted_spectrograms=PLOT_SPEC):

    print('Loading audio...')
    y, sr = load_audio(piece_path)
    y_trimmed, _ = librosa.effects.trim(y, top_db=40)
    stft = extract_stft(y_trimmed, n_fft=N_FFT, hop_length=HOP_LEN)
    harmonic, percussive = harmonic_percussive(stft)

    if plot_extracted_spectrograms:
        plot_spectrogram(stft, sr, save_path, 'stft')
        plot_spectrogram(harmonic, sr, save_path, 'harmonic')
        plot_spectrogram(percussive, sr, save_path, 'percussive')

    print('Extracting features...')
    ch = get_chromagram(harmonic, sr)
    mf = get_mfccs(stft, sr)
    temp = get_tempogram(percussive, sr)
    en = get_energy(y_trimmed).reshape(1, -1)
    features = {
        'chroma': ch,
        'mfcc': mf,
        'tempogram': temp,
        'energy': en
    }

    if plot_features:
        for name, feature in features.items():
            if name != 'energy':
                plot_feature(feature, SR, HOP_LEN, save_path, name)
            else:
                plot_energy(y_trimmed, feature, SR, HOP_LEN, save_path, 'energy')

    print('Extracting recurrence matrices...')
    SSMs = {}
    for name, feature in features.items():
        stacked_feature = stack_memory(feature, n_steps=STACK_MEM)
        SSMs[name] = get_SSM(stacked_feature)
        plot_SSM(SSMs[name], SR, HOP_LEN, save_path, name)


    print('Extracting novelty functions...')
    novelty_functions = {}
    for name, rec_matrix in SSMs.items():
        novelty_function = get_novelty_function(rec_matrix)
        smoothed_novelty = smooth_novelty(novelty_function, window_length=SMOOTH_WL)
        novelty_functions[name] = local_normalize_novelty(smoothed_novelty)
    peaks = {}
    for name, novelty_function in novelty_functions.items():
        peaks[name], _ = peak_detection(novelty_function, window_length=MIN_DIST, threshold=THRESH)
    plot_multiple_novelty_peaks(list(novelty_functions.values()), list(peaks.values()), SR, HOP_LEN, save_path)

    print('Extracting segment similarities...')
    seg_SSMs = {}
    duration = len(features['chroma'][0])  # Assuming all features have the same shape
    for name, feature in features.items():
        reduction, newpeaks = segment_and_mean(feature, find_boundaries(peaks[name], duration))
        seg_SSMs[name] = compute_segment_similarities(reduction)
        plot_segment_similarity_matrix(seg_SSMs[name], newpeaks, SR, HOP_LEN, save_path, name)

    global_novelty = np.mean(list(novelty_functions.values()), axis=0)
    smoothed_global_novelty = smooth_novelty(global_novelty, window_length=SMOOTH_WL)
    smoothed_global_novelty = local_normalize_novelty(smoothed_global_novelty)
    global_peaks, _ = peak_detection(smoothed_global_novelty, window_length=MIN_DIST, threshold=THRESH)
    plot_summed_novelty_peaks(smoothed_global_novelty, global_peaks, SR, HOP_LEN, save_path)
    global_seg_SSMs = {}
    global_peaks = find_boundaries(global_peaks, duration)
    for name, feature in features.items():
        reduction = segment_and_mean_no_bound(feature, global_peaks)
        global_seg_SSMs[name] = compute_segment_similarities(reduction)
    sum_similarities = np.sum(list(global_seg_SSMs.values()), axis=0)
    plot_segment_similarity_matrix(sum_similarities, global_peaks, SR, HOP_LEN, save_path, 'global')

    print('Creating audio with clicks...')
    create_clicks_audio(y_trimmed, global_peaks, save_path)

    print('Saving data to JSON...')
    save_json(file, peaks['chroma'], peaks['mfcc'], peaks['tempogram'], peaks['energy'], global_peaks, save_path)



# MAIN

def main():
    print('-'*50)
    print(f'ANALYZING {len(os.listdir(AUDIO_FOLDER))} FILES')
    for i, file in enumerate(os.listdir(AUDIO_FOLDER)):
        if file.endswith('.wav') or file.endswith('.mp3'):
            print('-'*50)
            print(f'STEP {i + 1}/{len(os.listdir(AUDIO_FOLDER))} - ANALYZING FILE {file}')

            foldername = file.split('.')[0]
            folder_name = os.path.join(ANALYSIS_FOLDER, foldername)

            if not os.path.exists(folder_name):
                os.makedirs(folder_name)

            extractors(os.path.join(AUDIO_FOLDER, file), folder_name, file)
            print('---')
            print(f'File {file} analyzed!')



if __name__ == '__main__':

    main()
    print('-'*50)
    print('All files analyzed!')