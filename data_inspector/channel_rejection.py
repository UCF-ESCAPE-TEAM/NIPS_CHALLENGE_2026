import numpy as np
from scipy.fft import fft, fftfreq

def detect_high_rms_channels(sample, n):
    """
    Detect bad EMG channels based on the RMS from a window of EMG data
    Args:
        sample: numpy array of shape (window_size, n_channels)
        n: number of standard deviations to add to the mean
    Returns:
        numpy array of shape (n_channels,)
            Indices of the bad channels
    """
    if sample.shape[0] < sample.shape[1]:
        sample = sample.T
    rms = np.sqrt(np.mean(np.square(sample), axis=0))
    threshold = np.mean(rms) + n * np.std(rms)
    return np.where(rms > threshold)[0]


def detect_powerline_channels(sample, first_seg=200, fs=2000):
    if sample.shape[0] < sample.shape[1]:
        sample = sample.T
    first_seg_sample = sample[:first_seg, :]
    fft_result = fft(first_seg_sample, axis=0)
    freq = fftfreq(first_seg_sample.shape[0], 1 / fs)
    fft_result = np.abs(fft_result[:first_seg // 2, :])
    freq = freq[:first_seg // 2]

    energy = np.zeros(sample.shape[1])
    for target in [50, 100]:
        idx = np.argmin(np.abs(freq - target))
        energy += fft_result[idx, :]

    threshold = np.mean(energy) + 3 * np.std(energy)
    return np.where(energy > threshold)[0]

def arm_band_interpolate_channels(sample, bad_channels, total_channels):
    """
    Interpolate the bad channels in the EMG arm band, if the bad channels is the first channel then use the last channel and the second channel to interpolate the bad channel.
    If the bad channels is the last channel then use the first channel and the second to last channel to interpolate the bad channel.
    Args:
        sample: numpy array of shape (window_size, n_channels)
        bad_channels: numpy array of shape (n_bad_channels,)
        total_channels: int
    Returns:
    """
    if sample.shape[0] < sample.shape[1]:
        sample = sample.T
        flip_output = True
    else:
        flip_output = False
    for bad_channel in bad_channels:
        if bad_channel == 0:
            sample[:, bad_channel] = (sample[:, -1] + sample[:, 1]) / 2
        elif bad_channel == total_channels - 1:
            sample[:, bad_channel] = (sample[:, 0] + sample[:, -2]) / 2
        else:
            sample[:, bad_channel] = (sample[:, bad_channel - 1] + sample[:, bad_channel + 1]) / 2
    return sample if not flip_output else sample.T


if __name__ == "__main__":
    sample = np.random.randn(100, 16)
    bad_channels = detect_high_rms_channels(sample, 3)
    print(bad_channels)