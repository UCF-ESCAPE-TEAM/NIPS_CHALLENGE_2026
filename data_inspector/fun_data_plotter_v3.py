import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import mne
import pandas as pd
import pathlib
from scipy.ndimage import binary_dilation, uniform_filter1d, median_filter, minimum_filter1d
from scipy.signal import butter, sosfiltfilt, iirnotch, filtfilt, welch, spectrogram
from channel_rejection import detect_high_rms_channels, detect_powerline_channels, arm_band_interpolate_channels


# ----------------------------------------------------------------------------
# numpy / scipy signal processing (MNE is used only to read the BDF file)
# ----------------------------------------------------------------------------

def detect_transients(emg: np.ndarray,
                      sfreq: float,
                      step_ratio: float = 4.0,
                      step_win_s: float = 0.02,
                      rms_win_s: float = 0.2,
                      rail_frac: float = 0.98,
                      pad_s: float = 0.05) -> np.ndarray:
    """
    Flag broadband transients (electrode pops, contact loss, motion steps) on the RAW EMG.
    These are what turn into narrow-band "ringing" after band-pass / notch filtering, so they
    have to be found here, before any filter sees them.

    Detection is a baseline-STEP test normalised by the LOCAL EMG level: at each sample compare
    the mean of the next `step_win_s` against the mean of the previous `step_win_s`. Genuine EMG
    is zero-mean, so this stays ~0 no matter how strong the contraction; a pop / contact loss is
    a DC offset jump and shows up as a step much larger than the EMG riding on it.
    Args:
        emg: Raw EMG, shape (n_channels, n_samples), in volts.
        sfreq: Sampling frequency in Hz.
        step_ratio: Flag when |step| > step_ratio * local EMG RMS (plus a floor at the recording's
            median RMS so quiet rest periods do not fire on tiny drifts).
        step_win_s: Half-width of the two-sided mean used to measure the step.
        rms_win_s: Window for the local EMG RMS (on the 20-450 Hz content).
        rail_frac: Also flag samples at |x| >= rail_frac * max|x| (ADC saturation).
        pad_s: Each flagged sample is dilated by +/- this many seconds so the whole event is covered.
    Returns:
        Boolean mask, shape (n_channels, n_samples), True where the sample is an artifact.
    """
    w = max(int(round(step_win_s * sfreq)), 2)
    r = max(int(round(rms_win_s * sfreq)), 2)

    # local EMG level: RMS of the in-band signal (cheap IIR, zero-phase)
    sos = butter(4, [20, min(450, 0.95 * sfreq / 2)], btype="bandpass", fs=sfreq, output="sos")
    hp = sosfiltfilt(sos, emg, axis=1)
    local_rms = np.sqrt(uniform_filter1d(hp ** 2, size=r, axis=1, mode="nearest"))
    floor = np.median(local_rms, axis=1, keepdims=True)

    # two-sided step: mean of the next w samples minus mean of the previous w samples
    m = uniform_filter1d(emg, size=w, axis=1, mode="nearest")      # centred mean, width w
    half = w // 2
    right = np.roll(m, -half, axis=1)       # mean of [t, t+w)
    left = np.roll(m, half, axis=1)         # mean of [t-w, t)
    step = np.abs(right - left)
    step[:, :w] = 0
    step[:, -w:] = 0
    step_bad = step > step_ratio * (local_rms + floor)

    # ADC rails / saturation: a real clip sits at the rail for a RUN of samples, so only
    # count channels where at least `rail_min_samples` land there (a lone noise peak is not a clip)
    rail_min_samples = 5
    amax = np.max(np.abs(emg), axis=1, keepdims=True)
    rail = np.abs(emg) >= rail_frac * amax
    rail &= rail.sum(axis=1, keepdims=True) >= rail_min_samples

    mask = step_bad | rail
    k = int(round(pad_s * sfreq))
    if k > 0:
        mask = binary_dilation(mask, structure=np.ones((1, 2 * k + 1), dtype=bool))
    return mask


def blank_transients(emg: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Linearly interpolate across every flagged run, per channel, so the filter never sees a step.
    """
    x = emg.copy()
    idx = np.arange(x.shape[1])
    for c in range(x.shape[0]):
        bad = mask[c]
        if bad.any() and (~bad).sum() > 1:
            x[c, bad] = np.interp(idx[bad], idx[~bad], x[c, ~bad])
    return x


def narrowband_diagnostic(emg: np.ndarray, sfreq: float, ch: int,
                          t_bad: float, t_good: float, dur: float = 0.3,
                          fmin: float = 20, fmax: float = 450):
    """
    Compare the PSD of a suspect segment against a clean segment of the same channel and
    report where the extra power sits. Run this BEFORE deciding between a fixed notch and
    the sliding narrow-band detector.
    Args:
        emg: (n_channels, n_samples) array (raw or filtered, any unit).
        sfreq: Sampling frequency.
        ch: Channel index.
        t_bad: Start (s) of a segment that shows the oscillation.
        t_good: Start (s) of a segment that looks like normal EMG / rest.
        dur: Length of both segments in seconds.
    Returns:
        (f, psd_bad, psd_good, f_peak)
    """
    a, b = int(t_bad * sfreq), int((t_bad + dur) * sfreq)
    c, d = int(t_good * sfreq), int((t_good + dur) * sfreq)
    nper = min(512, b - a, d - c)
    f, p_bad = welch(emg[ch, a:b], fs=sfreq, nperseg=nper)
    _, p_good = welch(emg[ch, c:d], fs=sfreq, nperseg=nper)
    band = (f >= fmin) & (f <= fmax)
    ratio = p_bad[band] / (p_good[band] + 1e-30)
    f_peak = f[band][np.argmax(ratio)]
    print(f"ch{ch} @ {t_bad:.2f}s: artifact peak at {f_peak:.1f} Hz, "
          f"{10 * np.log10(ratio.max()):.1f} dB above the clean segment")
    return f, p_bad, p_good, f_peak


def detect_narrowband(emg: np.ndarray,
                      sfreq: float,
                      win_s: float = 0.25,
                      peak_ratio: float = 6.0,
                      avg_s: float = 0.4,
                      nb_hz: float = 25.0,
                      fmin: float = 20,
                      fmax: float = 450,
                      pad_s: float = 0.05,
                      persist_s: float = 0.3,
                      return_freq: bool = False):
    """
    Flag stretches whose spectrum is dominated by a single line (band resonance, vibration,
    cable/connector ringing). Broadband EMG has a peak-to-median PSD ratio of ~2-4 inside
    [fmin, fmax]; a near-pure tone gives >> 10. Such a tone sits inside the EMG passband, so
    neither the band-pass nor the mains notches can remove it; it has to be flagged in time.
    Args:
        emg: (n_channels, n_samples) array. Use the raw or band-passed signal, NOT the min-max
            normalised one (the ratio is scale-free, but normalisation does not matter either way).
        sfreq: Sampling frequency.
        win_s: Length of signal blanked around each hit (seconds).
        peak_ratio: Threshold on the local spectral prominence (bin / median of its +/-nb_hz
            neighbourhood, excluding the bin and its two neighbours) after time-averaging.
            Noise and real EMG ~1-3, a tone >> 5. Defaults (6, avg 0.4 s, +/-25 Hz, persist
            0.3 s) gave zero false positives on coloured synthetic EMG while catching tones.
        avg_s: Time-averaging of the spectrogram before the prominence test.
        nb_hz: Half-width of the spectral neighbourhood used as the local baseline.
        fmin, fmax: Band inside which the ratio is evaluated.
        pad_s: Dilate each flagged run by +/- this many seconds.
        persist_s: A spectral line must stay above peak_ratio in the SAME bin for at least this
            long to count; this is what separates a resonance from noise extremes.
        return_freq: Also return the dominant frequency per (channel, window) where flagged.
    Returns:
        mask (n_channels, n_samples) bool; optionally also (t_centres, f_peak) with f_peak
        shape (n_channels, n_windows), NaN where not flagged.
    """
    # Short FFTs (nfft_s) averaged over win_s: a single periodogram has only ~2 dof per bin, so
    # its max/median is ~7 even for white noise; averaging ~15 of them brings noise down to ~2
    # while a tone stays >> 10.
    # The statistic is LOCAL prominence: each bin divided by the median of its spectral
    # neighbourhood (+/- ~40 Hz, excluding the bin itself and its two neighbours). Real EMG
    # has a smooth, coloured spectrum -> prominence ~1-2 everywhere; a tone is a line that
    # sticks out of its neighbourhood -> prominence >> 5. A global max/median would wrongly
    # fire on ordinary EMG whose power is concentrated in 50-150 Hz.
    nfft = 256 if sfreq >= 1000 else 128                         # ~8 Hz bins at 2 kHz
    hop_fft = nfft // 4
    f, t, S = spectrogram(emg, fs=sfreq, nperseg=nfft, noverlap=nfft - hop_fft, axis=1)
    n_avg = max(int(round(avg_s * sfreq / hop_fft)), 1)
    S = uniform_filter1d(S, size=n_avg, axis=2, mode="nearest")   # time-averaging
    df = f[1] - f[0]
    nb_half = max(int(round(nb_hz / df)), 3)                        # neighbourhood half-width in bins
    foot = np.ones((1, 2 * nb_half + 1, 1), dtype=bool)
    foot[0, nb_half - 1:nb_half + 2, 0] = False                  # exclude centre +/- 1 bin
    band = (f >= fmin) & (f <= fmax)
    # neighbourhood restricted to the band (reflect at the band edges) so a bin near 20 Hz is
    # not compared against the near-zero bins below the high-pass
    Sb_ = S[:, band, :]
    local_b = median_filter(Sb_, footprint=foot, mode="reflect")
    prom = np.zeros_like(S)
    prom[:, band, :] = Sb_ / (local_b + 1e-30)
    # Persistence: a resonance holds the SAME bin for a while, the loudest noise bin hops
    # around. Keep a (bin, time) hit only if it is above threshold for >= persist_s in a row.
    n_persist = max(int(round(persist_s * sfreq / hop_fft)), 1)
    hit = (prom > peak_ratio) & band[None, :, None]
    hit = minimum_filter1d(hit.astype(np.uint8), size=n_persist, axis=2, mode="constant", cval=0) == 1
    hit = binary_dilation(hit, structure=np.ones((1, 1, n_persist), dtype=bool))  # restore full run
    bad_t = hit.any(axis=1)                                      # (ch, t)

    mask = np.zeros(emg.shape, dtype=bool)
    centres = np.round(t * sfreq).astype(int)
    half = int(win_s * sfreq) // 2
    nper = 2 * half
    n = emg.shape[1]
    for c in range(emg.shape[0]):
        for i in np.where(bad_t[c])[0]:
            mask[c, max(centres[i] - half, 0):min(centres[i] + half, n)] = True
    k = int(round(pad_s * sfreq))
    if k > 0:
        mask = binary_dilation(mask, structure=np.ones((1, 2 * k + 1), dtype=bool))

    if not return_freq:
        return mask
    f_peak = np.full(bad_t.shape, np.nan)
    idx = np.where(hit, prom, 0).argmax(axis=1)                  # the persistent bin itself
    f_peak[bad_t] = f[idx[bad_t]]
    return mask, t, f_peak


def filter_emg(emg: np.ndarray,
               sfreq: float,
               l_freq: float = 20.0,
               h_freq: float = 450.0,
               bp_order: int = 4,
               notch_freqs=(60, 120, 180),
               notch_q: float = 30.0,
               extra_notch_freqs=(),
               extra_notch_q: float = 10.0) -> np.ndarray:
    """
    Zero-phase band-pass + mains notches with scipy, on a (n_channels, n_samples) array.
    A 4th-order Butterworth (8th effective after filtfilt) rings for ~50 ms; an iirnotch with
    Q=30 (2 Hz wide at 60 Hz) for tens of ms. The v1 MNE FIR filters had kernels of 0.66 s
    (band-pass) and 3.3 s (notch), which is why any residual transient smeared into a long
    narrow-band burst.
    Args:
        emg: (n_channels, n_samples) array, any unit.
        sfreq: Sampling frequency in Hz.
        l_freq, h_freq: Band-pass edges in Hz (h_freq is clipped below Nyquist).
        bp_order: Butterworth order (per pass).
        notch_freqs: Mains fundamental and harmonics to notch; those above Nyquist are skipped.
        notch_q: Quality factor of each notch (bandwidth = f0 / Q).
        extra_notch_freqs: Additional fixed-frequency notches, e.g. a band resonance found with
            narrowband_diagnostic(). Use only when the artifact frequency is the SAME every time;
            if it drifts, use detect_narrowband() instead.
        extra_notch_q: Q of the extra notches (wider than the mains notches by default).
    Returns:
        Filtered array, same shape as the input.
    """
    nyq = sfreq / 2
    h_freq = min(h_freq, 0.95 * nyq)
    sos = butter(bp_order, [l_freq, h_freq], btype="bandpass", fs=sfreq, output="sos")
    x = sosfiltfilt(sos, emg, axis=1)
    for f0, q in [(f, notch_q) for f in notch_freqs] + [(f, extra_notch_q) for f in extra_notch_freqs]:
        if f0 >= nyq:
            continue
        b, a = iirnotch(f0, q, fs=sfreq)
        x = filtfilt(b, a, x, axis=1)
    return x


def minmax_normalise(x: np.ndarray) -> np.ndarray:
    """Min-max over all channels jointly (same as the v1 channel_wise=False behaviour)."""
    return (x - x.min()) / (x.max() - x.min() + 1e-12)


# ----------------------------------------------------------------------------
# plotting
# ----------------------------------------------------------------------------

def plot_windowed_data(data_folder: pathlib.Path,
                       save_folder: pathlib.Path,
                       fname_stem: str,
                       window_size: int,
                       step_size: int,
                       signal_type: str,
                       sfreq: float = 2000,
                       raw_data: bool = True,
                       channel_id: int = None,
                       max_windows: int = None,
                       reject_bad_channels: bool = False,
                       remove_transients: bool = True,
                       remove_narrowband: bool = True,
                       narrowband_peak_ratio: float = 6.0,
                       extra_notch_freqs=(),
                       diagnose: list = None,
                       shade_transients: bool = True,
                       dpi: int = 120):
    """
    Plot the windowed data of either the EMG or the Joint Angle Calculated by Inverse Kinematics.
    Args:
        data_folder: The path to the data folder.
        save_folder: The path to the save folder.
        fname_stem: The stem of the data file.
        window_size: The size of the window in samples.
        step_size: The step size in samples.
        signal_type: The type of signal to plot, either "EMG" or "Joint Angle".
        sfreq: The sampling frequency of the data (overridden by the value in the BDF header).
        raw_data: Whether to plot the raw data, if True, plot the raw data, if False, then bandpass filter the data and plot the filtered data.
        channel_id: The ID of the channel to plot, if None, plot all channels on the same plot
        max_windows: Stop after this many windows (None = all). Useful for a quick look.
        reject_bad_channels: Whether to reject the bad channels, if True, then reject the bad channels and interpolate the bad channels.
        remove_transients: (filtered mode only) Detect pops/motion steps on the raw EMG and interpolate
            across them BEFORE filtering, so they do not ring through the band-pass / notch.
        remove_narrowband: (filtered mode only) Detect stretches dominated by a single-frequency
            tone (band resonance / vibration) on the band-passed EMG and interpolate across them.
            These sit inside the EMG passband, so no filter can remove them.
        narrowband_peak_ratio: Threshold for detect_narrowband (local spectral prominence).
        extra_notch_freqs: Fixed notches to add on top of 60/120/180 Hz, e.g. (47,) once
            narrowband_diagnostic shows the artifact is always at the same frequency.
        diagnose: Optional list of (channel, t_bad, t_good) tuples; prints the artifact peak
            frequency for each, computed on the raw EMG in CONTINUOUS time (before BAD_IK removal).
        shade_transients: Shade the flagged artifact samples (step + narrow-band) on the plot.
        dpi: Resolution of the saved PNG files.
    Returns:
        List of the saved figure paths.
    """
    assert signal_type in ("EMG", "Joint Angle"), f"unknown signal_type {signal_type!r}"

    def sliding_window(data, window_size, step_size):
        r"""
        Generate the segmented EMG and Joint Angle Data -> (n_windows, n_channels, window_size)
        """
        n_channels, n_samples = data.shape
        # +1 so a window that ends exactly at the last sample is kept
        start_indices = np.arange(0, n_samples - window_size + 1, step_size)
        print(f"The given data can be segmented into {len(start_indices)} segments with a window size of {window_size} and a step size of {step_size}.")
        segmented_data = np.array([data[:, start:start + window_size] for start in start_indices])
        return segmented_data, start_indices

    # ---- 1. read the BDF with MNE, then leave MNE behind --------------------
    data_fname = f"{fname_stem}_emg.bdf"
    event_fname = f"{fname_stem}_events.tsv"
    channel_fname = f"{fname_stem}_channels.tsv"
    data = mne.io.read_raw_bdf(data_folder / data_fname, preload=True, verbose='error')
    sfreq = data.info['sfreq']
    print(f"Sampling frequency: {sfreq} Hz")
    chans = pd.read_csv(data_folder / channel_fname, sep='\t')
    emg_names = chans.loc[chans['type'] == 'EMG', 'name'].tolist()
    # joint angle is categorized as "MISC" in the channel file
    joint_names = chans.loc[chans['type'] == 'MISC', 'name'].tolist()

    emg_raw = data.get_data(picks=emg_names)          # (16, n) volts
    joints_raw = data.get_data(picks=joint_names)     # (20, n) rad
    n_times = emg_raw.shape[1]
    del data

    # ---- 2. BAD_IK events -> boolean keep mask on the continuous time base ---
    events = pd.read_csv(data_folder / event_fname, sep="\t")
    bad = events[events["trial_type"] == "BAD_IK"]
    print(bad[["onset", "duration"]])
    keep = np.ones(n_times, dtype=bool)
    for onset, dur in zip(bad["onset"].to_numpy(), bad["duration"].to_numpy()):
        a = int(round(onset * sfreq))
        b = int(round((onset + dur) * sfreq))
        keep[max(a, 0):min(b, n_times)] = False

    # ---- 3. processing (all numpy / scipy) ----------------------------------
    transient_mask = None       # combined artifact mask on the continuous (pre-omit) time base
    if raw_data:
        emg_unit = "µV"
        emg_proc = emg_raw * 1e6
    else:
        emg_proc = emg_raw.copy()
        transient_mask = np.zeros(emg_proc.shape, dtype=bool)

        if diagnose:
            print("--- narrow-band diagnostic (raw EMG, continuous time) ---")
            for ch, t_bad, t_good in diagnose:
                narrowband_diagnostic(emg_proc, sfreq, ch, t_bad, t_good)

        # 3a. find & blank broadband transients on the RAW data.
        # A band-pass cannot remove a pop / contact-loss step: the step has energy at
        # every frequency, and what survives 20-450 Hz looks like a ~20 Hz spindle;
        # what survives a narrow notch looks like a 60 Hz tail. So the step has to be
        # removed while it is still a step, i.e. before any filtering.
        if remove_transients:
            step_mask = detect_transients(emg_proc, sfreq)
            frac = step_mask.mean(axis=1) * 100
            print("Step/pop samples flagged per channel (%): "
                  + ", ".join(f"{n}:{p:.2f}" for n, p in zip(emg_names, frac)))
            if (frac > 5).any():
                print("WARNING: >5% flagged on some channels; raise step_ratio or inspect the "
                      "shaded regions before trusting the interpolation.")
            emg_proc = blank_transients(emg_proc, step_mask)
            transient_mask |= step_mask

        # 3b. zero-phase band-pass + mains notches (short impulse responses)
        emg_proc = filter_emg(emg_proc, sfreq,
                              l_freq=20, h_freq=450, bp_order=4,
                              notch_freqs=(60, 120, 180), notch_q=30,
                              extra_notch_freqs=extra_notch_freqs)

        # 3c. find & blank single-tone stretches on the BAND-PASSED data.
        # A tone inside 20-450 Hz (band resonance, vibration) survives every filter
        # above, so it is flagged by its spectral shape (one line vs. broadband EMG).
        if remove_narrowband:
            nb_mask, nb_t, nb_f = detect_narrowband(emg_proc, sfreq,
                                                    peak_ratio=narrowband_peak_ratio,
                                                    return_freq=True)
            frac = nb_mask.mean(axis=1) * 100
            print("Narrow-band samples flagged per channel (%): "
                  + ", ".join(f"{n}:{p:.2f}" for n, p in zip(emg_names, frac)))
            flagged_f = nb_f[~np.isnan(nb_f)]
            if flagged_f.size:
                print(f"  dominant frequency of flagged windows: median {np.median(flagged_f):.1f} Hz, "
                      f"IQR {np.percentile(flagged_f, 25):.1f}-{np.percentile(flagged_f, 75):.1f} Hz "
                      f"-> if the IQR is narrow, add that frequency to extra_notch_freqs")
            emg_proc = blank_transients(emg_proc, nb_mask)
            transient_mask |= nb_mask

        if not transient_mask.any():
            transient_mask = None

        # 3d. min-max over all channels jointly
        emg_proc = minmax_normalise(emg_proc)
        emg_unit = "a.u. (min-max)"

    # ---- 4. drop BAD_IK samples AFTER filtering ------------------------------
    # The filters ran on the continuous data, so the seams do not ring. The
    # remaining signal is NOT continuous at the seams; fine for eyeballing.
    emg_clean = emg_proc[:, keep]
    joints_clean = joints_raw[:, keep]
    mask_clean = transient_mask[:, keep] if transient_mask is not None else None

    emg_segments, starts = sliding_window(emg_clean, window_size, step_size)  # (N, 16, W)
    joints_segments, _ = sliding_window(joints_clean, window_size, step_size)  # (N, 20, W)
    mask_segments = None
    if mask_clean is not None:
        mask_segments = np.array([mask_clean[:, s:s + window_size] for s in starts])
    data_segmented = {"EMG": emg_segments, "Joint Angle": joints_segments}
    ch_names = {"EMG": emg_names, "Joint Angle": joint_names}[signal_type]
    units = {"EMG": emg_unit, "Joint Angle": "rad"}[signal_type]
    print(f"The shape of the EMG segments is {emg_segments.shape}, and the shape of the joint segments is {joints_segments.shape}.")

    segments = data_segmented[signal_type]
    n_windows = segments.shape[0]
    n_channels = segments.shape[1]
    if channel_id is not None and channel_id > n_channels - 1:
        print(f"channel_id {channel_id} out of range, plotting the last channel ({n_channels - 1}) instead")
        channel_id = n_channels - 1
    if max_windows is not None:
        n_windows = min(n_windows, max_windows)

    # One vertical spacing for the whole recording so windows are comparable.
    # Each trace is centred on its own median inside the window, and the spacing
    # comes from a robust peak amplitude (99th percentile of |x - median|) so that
    # bursty EMG, whose peaks are 10-20x its SD, still fits in its lane.
    if channel_id is None:
        centred = segments - np.nanmedian(segments, axis=-1, keepdims=True)
        peak = np.nanpercentile(np.abs(centred), 99)        # one number for all channels/windows
        spacing = 2.0 * peak if peak > 0 else 1.0
    else:
        spacing = 0.0
    offsets = np.arange(n_channels)[::-1] * spacing       # channel 0 on top

    save_folder = pathlib.Path(save_folder)
    save_folder.mkdir(parents=True, exist_ok=True)
    tag = "raw" if raw_data else "filtered"
    sig_tag = signal_type.replace(" ", "")
    saved = []
    bad_channel_log = {}

    if reject_bad_channels:
        for n in range(n_windows):
            rms_channels = detect_high_rms_channels(segments[n, :, :], 3)
            powerline_channels = detect_powerline_channels(segments[n, :, :], 200, sfreq)
            # combine the bad channels
            bad_channels = np.concatenate((rms_channels, powerline_channels))
            bad_channels = np.unique(bad_channels)
            print(f"The {n}th window has {len(bad_channels)} bad channels.")
            bad_channel_log[f"win{n}"] = bad_channels
            segments[n, :, :] = arm_band_interpolate_channels(segments[n, :, :], bad_channels, n_channels)
    else:
        for n in range(n_windows):
            bad_channel_log[f"win{n}"] = "N/A"

    show_mask = (shade_transients and signal_type == "EMG" and mask_segments is not None)

    def shade(ax, t, m, y0, y1):
        """Shade contiguous True runs of mask m (1-D) between y0 and y1 (axes fractions)."""
        if not m.any():
            return
        edges = np.diff(np.concatenate(([0], m.astype(int), [0])))
        for a, b in zip(np.where(edges == 1)[0], np.where(edges == -1)[0]):
            ax.axvspan(t[a], t[min(b, len(t) - 1)], ymin=y0, ymax=y1,
                       color="tab:red", alpha=0.18, lw=0)

    for i in range(n_windows):
        # continuous time axis: window i spans [starts[i], starts[i] + window_size) samples
        t = (starts[i] + np.arange(window_size)) / sfreq
        t0, t1 = t[0], t[-1] + 1 / sfreq
        title = (f"Window {i} ({t0:.2f}-{t1:.2f} s) of the {signal_type} data ({tag}) "
                 f"for data file\n{data_fname}\n{bad_channel_log[f'win{i}']}")
        fig, ax = plt.subplots(1, 1, figsize=(12, 10 if channel_id is None else 4))

        if channel_id is not None:
            # segments is (n_windows, n_channels, window_size): channel axis is 1
            y = segments[i, channel_id, :]
            ax.plot(t, y, lw=0.6, color="k")
            ax.set_ylabel(f"{ch_names[channel_id]} ({units})")
            if show_mask:
                shade(ax, t, mask_segments[i, channel_id, :], 0.0, 1.0)
            fname = f"{fname_stem}_{sig_tag}_{tag}_ch{channel_id}_win{i:04d}.png"
        else:
            # all channels stacked with a constant vertical offset
            ylo, yhi = offsets[-1] - spacing, offsets[0] + spacing
            for c in range(n_channels):
                y = segments[i, c, :]
                y = y - np.nanmedian(y)                     # centre on this channel's baseline
                ax.plot(t, y + offsets[c], lw=0.5, color="k")
                if show_mask:
                    # shade this channel's lane only (axes-fraction coordinates)
                    lane_lo = (offsets[c] - spacing / 2 - ylo) / (yhi - ylo)
                    lane_hi = (offsets[c] + spacing / 2 - ylo) / (yhi - ylo)
                    shade(ax, t, mask_segments[i, c, :], lane_lo, lane_hi)
            ax.set_yticks(offsets)
            ax.set_yticklabels(ch_names)
            ax.set_ylabel(f"channel (lane height {spacing:.3g} {units}, median removed)")
            ax.set_ylim(ylo, yhi)
            fname = f"{fname_stem}_{sig_tag}_{tag}_all_win{i:04d}.png"

        ax.set_xlim(t0, t1)
        ax.set_xlabel("Time (s), counted over the data with BAD_IK samples removed")
        ax.set_title(title)
        ax.grid(True, axis="x", alpha=0.3)
        fig.tight_layout()
        out = save_folder / fname
        fig.savefig(out, dpi=dpi)
        plt.close(fig)          # important in a loop: frees memory
        saved.append(out)

    print(f"Saved {len(saved)} figure(s) to {save_folder}")
    return saved



if __name__ == "__main__":
    window_size = 10000
    step_size = 10000
    signal_type = "EMG"
    sfreq = 2000
    raw_data = False
    channel_id = None
    subject_id = [i for i in range(2, 3)]
    subject_str = [f"sub-{i:02d}" for i in subject_id]
    sessions = ['ses-01', 'ses-02', 'ses-03', 'ses-04']
    # BIDS root is the dataset folder (nm000281), not the download parent.
    # Use raw strings so Windows backslashes are not treated as escape sequences.
    data_root = pathlib.Path(r'F:\EEG_Challenge\Dataset\Salter2024Emg2pose\download\nm000281')
    save_root = pathlib.Path(r'F:\EEG_Challenge\Inspection_FilterV3')

    for s in subject_str:
        for sess in sessions:
            data_folder = data_root / s / sess / 'emg'
            save_folder = save_root / s / sess
            save_folder.mkdir(parents=True, exist_ok=True)

            # Left/right is not odd/even (e.g. run-03 is right), and session run
            # counts differ, so plot whatever BDF files are actually present.
            bdf_files = sorted(data_folder.glob('*_emg.bdf'))
            if not bdf_files:
                print(f"No BDF files found in {data_folder}")
                continue
            for bdf in bdf_files:
                fname_stem = bdf.name.removesuffix('_emg.bdf')
                plot_windowed_data(
                    data_folder, save_folder, fname_stem,
                    window_size, step_size, signal_type, sfreq, raw_data, channel_id,
                    remove_transients=True,
                    remove_narrowband=True,
                    narrowband_peak_ratio=6.0,
                    extra_notch_freqs=(),           # e.g. (47,) once the diagnostic shows a fixed tone
                    # times are CONTINUOUS seconds (before BAD_IK removal); example for
                    # sub-02 ses-01 run-01 left, window 0:
                    diagnose=[(2, 1.3, 0.3), (4, 3.5, 0.3), (3, 4.65, 0.3)]
                    if fname_stem.endswith("run-01_recording-left") else None,
                    shade_transients=True,
                )
