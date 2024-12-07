# PLOTTERS

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import librosa.display
from matplotlib.collections import LineCollection
from mpl_toolkits.axes_grid1 import make_axes_locatable


def plot_feature(feature, sr, hop_length, path, name):
    match name:
        case 'chroma':
            plot_title = 'Chroma Features'
            y_ax = 'chroma'
            cmap = 'coolwarm'
            tempo_min = None
            tempo_max = None
        case 'mfcc':
            plot_title = 'Mel-Frequency Cepstral Coefficients'
            y_ax = None
            cmap = 'coolwarm'
            tempo_min = None
            tempo_max = None
        case 'tempogram':
            plot_title = 'Tempo Features'
            y_ax = 'tempo'
            cmap = 'coolwarm'
            tempo_min = 16
            tempo_max = 290
    
    plt.figure(figsize=(5, 4))
    librosa.display.specshow(feature, sr=sr, hop_length=hop_length, 
                             x_axis='time', y_axis=y_ax, 
                             cmap=cmap, 
                             tempo_min=tempo_min, tempo_max=tempo_max)
    plt.title(plot_title)
    if name == 'mfcc':
        plt.yticks(np.arange(feature.shape[0]), np.arange(1, feature.shape[0] + 1))
        plt.ylabel('Coefficients')
    plt.tight_layout()
    plt.colorbar()
    plt.savefig(f'{path}/feature_{name}.png', dpi=200)
    plt.close()


def plot_spectrogram(spectrogram, sr, save_path, name):
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(librosa.amplitude_to_db(spectrogram, ref=np.max), sr=sr, x_axis='time', y_axis='log')
    plt.title(f'{name} spectrogram')
    plt.tight_layout()
    plt.savefig(f'{save_path}/spectrogram_{name}.png', dpi=200)
    plt.close()


def plot_energy(y, energy, sr, hop_length, save_path, name):
    y = np.maximum(y, 0)
    y = y / np.max(np.abs(y))
    energy = energy / np.max(energy)
    energy_db = 20 * np.log10(np.maximum(energy, 1e-10))

    plt.figure(figsize=(5, 4))
    times = np.arange(len(y)) / sr
    plt.plot(times, y, alpha=0.4, label='Waveform')

    energy_times = np.arange(energy.shape[1]) * hop_length // sr
    cmap = plt.get_cmap('coolwarm') 
    norm = plt.Normalize(vmin=-30, vmax=0)
    points = np.array([energy_times, energy.flatten()]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    lc = LineCollection(segments, cmap=cmap, norm=norm)
    lc.set_array(energy_db.flatten())  
    lc.set_linewidth(3)
    plt.gca().add_collection(lc)

    plt.title('Root-Mean-Square + Waveform')

    librosa.display.specshow(np.zeros((1, len(y) // hop_length)), sr=sr, hop_length=hop_length, x_axis='time')
    plt.xlabel('Time')
    plt.ylabel('Normalized Amplitude')
    plt.yticks(np.linspace(0, 1, num=5))
    plt.xlim([0, times[-1]])
    plt.ylim([0, 1]) 
    cbar = plt.colorbar(lc)
    cbar.set_label('Amplitude (dB)')

    plt.tight_layout()
    plt.savefig(f'{save_path}/feature_{name}.png', dpi=200)
    plt.close()


def plot_SSM(SSM, sr, hop_length, path, name):
    plt.figure(figsize=(8, 8))
    img = librosa.display.specshow(SSM, cmap='coolwarm', x_axis='time', y_axis='time', sr=sr, hop_length=hop_length)
    match name:
        case 'chroma':
            plot_title = 'Chroma'
        case 'mfcc':    
            plot_title = 'MFCCs'
        case 'tempogram':
            plot_title = 'Tempogram'
        case 'energy':
            plot_title = 'Energy'    
    plt.title(f'{plot_title} SSM')
    ax = plt.gca()
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.25)
    plt.colorbar(img, cax=cax)
    plt.savefig(f'{path}/self_similarity_matrix_{name}.png', dpi=100)
    plt.close()


def plot_multiple_novelty_peaks(novelty_functions, peaks_positions, sr, hop_length, save_path):
    sns.set_theme(style="whitegrid")
    sns.set_context("talk")

    plt.figure(figsize=(16, 4))
    colors = ['blue', 'red', 'purple', 'orange']  
    feature_names = ['Chroma', 'MFCCs', 'Tempogram', 'Energy']

    for i, (novelty_function, peaks_position) in enumerate(zip(novelty_functions, peaks_positions)):
        frames = range(len(novelty_function))
        time_in_minutes = [(frame * hop_length // sr) / 60 for frame in frames] 
        peaks_time_in_minutes = [(peak * hop_length // sr) / 60 for peak in peaks_position]  
        sns.lineplot(x=time_in_minutes, y=novelty_function, label=feature_names[i], linewidth=3, color=colors[i], alpha=0.7)
        sns.scatterplot(x=peaks_time_in_minutes, y=[novelty_function[peak] for peak in peaks_position], color=colors[i], s=170, zorder=5)

    plt.xlabel('Time (minutes)')
    plt.ylabel('Novelty')
    plt.title('Novelty Functions and Peaks')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{save_path}/combined_novelty_peaks.png', dpi=200)
    plt.close()


def plot_summed_novelty_peaks(summed_novelty, peaks_summed, sr, hop_length, save_path):
    sns.set_theme(style="whitegrid")
    sns.set_context("talk")

    frames = range(len(summed_novelty))
    time_in_minutes = [(frame * hop_length // sr) / 60 for frame in frames]  
    peaks_time_in_minutes = [(peak * hop_length // sr) / 60 for peak in peaks_summed]  

    plt.figure(figsize=(12, 5))
    sns.lineplot(x=time_in_minutes, y=summed_novelty, linewidth=4)
    sns.scatterplot(x=peaks_time_in_minutes, y=summed_novelty[peaks_summed], color='red', s=300, zorder=5)

    plt.xlabel('Time (minutes)')
    plt.ylabel('Novelty')
    plt.title('Summed Novelty Function and Detected Peaks')
    plt.savefig(f'{save_path}/summed_novelty_peaks.png', dpi=200)
    plt.close()



def plot_segment_similarity_matrix(similarity_matrix, merged_peaks, sr, hop_length, savepath, name):
    extended_peaks = [peak * hop_length // sr for peak in merged_peaks]
    num_segments = len(extended_peaks) - 1
    segment_durations = [(extended_peaks[i+1] - extended_peaks[i]) for i in range(num_segments)]
    total_length_seconds = int(extended_peaks[-1])
    rescaled_ssm = np.zeros((total_length_seconds, total_length_seconds))

    start_x = 0
    for i in range(num_segments):
        end_x = start_x + int(segment_durations[i])
        start_y = 0
        for j in range(num_segments):
            end_y = start_y + int(segment_durations[j])
            rescaled_ssm[start_x:end_x, start_y:end_y] = similarity_matrix[i, j]
            start_y = end_y
        start_x = end_x

    def seconds_to_minsec(seconds):
        minutes = int(seconds // 60)
        seconds = int(seconds % 60)
        return f'{minutes}:{seconds:02d}'
    cumulative_durations = np.cumsum([0] + segment_durations)

    rescaled_ssm = np.flipud(rescaled_ssm)
    fig, ax = plt.subplots(figsize=(8, 8))
    img = ax.imshow(rescaled_ssm, cmap='coolwarm', aspect='equal', extent=[0, total_length_seconds, 0, total_length_seconds])
    ax = plt.gca()
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.25)
    cbar = plt.colorbar(img, cax=cax)
    cbar.set_label('Similarity')
    ax.set_xticks(cumulative_durations)
    ax.set_yticks(cumulative_durations)
    time_labels = [seconds_to_minsec(t) for t in cumulative_durations]
    ax.set_xticklabels(time_labels, rotation=45)
    ax.set_yticklabels(time_labels)
    ax.set_xlabel('Time (minutes:seconds)')
    ax.set_ylabel('Time (minutes:seconds)')
    ax.grid(which='both', color='white', linewidth=6)
    match name:
        case 'chroma':
            plot_title = 'Chroma'
        case 'mfcc':    
            plot_title = 'MFCCs'
        case 'tempogram':
            plot_title = 'Tempogram'
        case 'energy':
            plot_title = 'Energy'    
        case 'global':
            plot_title = 'Global'
    ax.set_title(f'{plot_title} SegSM')
    plt.tight_layout()
    plt.savefig(f'{savepath}/segment_similarity_matrix_{name}.png', dpi=100)
    plt.close()
