import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import mne
import pandas as pd
import pathlib
from scipy.ndimage import binary_dilation
from channel_rejection import detect_high_rms_channels, detect_powerline_channels, arm_band_interpolate_channels


def detect_transients(emg: np.ndarray,
                      sfreq: float,
                      jump_mad: float = 12.0,
                      swing_ratio: float = 8.0,
                      swing_hfreq: float = 10.0,
                      pad_s: float = 0.05) -> np.ndarray:
    """
    Flag broadband transients (electrode pops, contact loss, motion steps) on the RAW EMG.
    These are what turn into narrow-band "ringing" after band-pass / notch filtering, so they
    have to be found here, before any filter sees them.
    Args:
        emg: Raw EMG, shape (n_channels, n_samples), in volts.
        sfreq: Sampling frequency in Hz.
        jump_mad: Sample-to-sample jump threshold, in robust SDs (MAD * 1.4826) above the median jump.
        swing_ratio: Low-frequency baseline swing threshold, as a multiple of the median swing.
        swing_hfreq: Cut-off of the low-pass used to isolate baseline swings (below the EMG band).
        pad_s: Each flagged sample is dilated by +/- this many seconds so the whole event is covered.
    Returns:
        Boolean mask, shape (n_channels, n_samples), True where the sample is an artifact.
    """
    # a) abrupt steps: first difference far beyond normal EMG
    d = np.abs(np.diff(emg, axis=1, prepend=emg[:, :1]))
    med = np.median(d, axis=1, keepdims=True)
    mad = np.median(np.abs(d - med), axis=1, keepdims=True)
    jump = d > med + jump_mad * 1.4826 * mad

    # b) slow baseline swings (electrode motion): power well below the EMG band
    lp = mne.filter.filter_data(emg, sfreq, l_freq=None, h_freq=swing_hfreq, verbose="error")
    swing = np.abs(lp - np.median(lp, axis=1, keepdims=True))
    swing_bad = swing > swing_ratio * np.median(swing, axis=1, keepdims=True)

    mask = jump | swing_bad
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
        sfreq: The sampling frequency of the data.
        raw_data: Whether to plot the raw data, if True, plot the raw data, if False, then bandpass filter the data and plot the filtered data.
        channel_id: The ID of the channel to plot, if None, plot all channels on the same plot
        max_windows: Stop after this many windows (None = all). Useful for a quick look.
        reject_bad_channels: Whether to reject the bad channels, if True, then reject the bad channels and interpolate the bad channels.
        remove_transients: (filtered mode only) Detect pops/motion steps on the raw EMG and interpolate
            across them BEFORE filtering, so they do not ring through the band-pass / notch.
        shade_transients: Shade the flagged transient samples on the plot (filtered mode only).
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
    # tell MNE the real channel types EMG and Joint Angle (it guessed "eeg" for all).
    data.set_channel_types({**{n: "emg" for n in emg_names},
                            **{n: "misc" for n in joint_names}}, verbose="error")
    print(data.info)

    events = pd.read_csv(data_folder / event_fname, sep="\t")
    bad = events[events["trial_type"] == "BAD_IK"]
    print(bad[["onset", "duration"]])
    data.set_annotations(mne.Annotations(onset=bad["onset"].to_numpy(),
                                         duration=bad["duration"].to_numpy(),
                                         description=["BAD_IK"] * len(bad)))

    # transient mask on the continuous (pre-omit) time base; None when not computed
    transient_mask = None

    if raw_data:
        emg_unit = "µV"
        emg_scale = 1e6          # MNE returns volts
    else:
        emg_picks = mne.pick_types(data.info, emg=True)

        # ---- 1. find & blank broadband transients on the RAW data ----------
        # A band-pass cannot remove a pop / contact-loss step: the step has energy at
        # every frequency, and what survives 20-450 Hz looks like a ~20 Hz spindle;
        # what survives a narrow notch looks like a 60 Hz tail. So the step has to be
        # removed while it is still a step, i.e. before any filtering.
        if remove_transients:
            raw_emg = data.get_data(picks=emg_picks)
            transient_mask = detect_transients(raw_emg, sfreq)
            frac = transient_mask.mean(axis=1) * 100
            print("Transient samples flagged per channel (%): "
                  + ", ".join(f"{n}:{p:.2f}" for n, p in zip(emg_names, frac)))
            data._data[emg_picks] = blank_transients(raw_emg, transient_mask)

        # ---- 2. short-impulse-response filters (IIR Butterworth, zero-phase) ---
        # The v1 FIR band-pass (5 Hz transition at 20 Hz) had a ~0.66 s kernel and the
        # v1 FIR notch (notch_widths=6, trans_bandwidth=2) a ~3.3 s kernel, so any
        # residual transient rang for a long time. A 4th-order Butterworth rings for
        # ~50 ms; a 2nd-order IIR notch (width 2 Hz, Q~30) for tens of ms.
        data = data.filter(l_freq=20, h_freq=450, picks="emg", method="iir",
                           iir_params=dict(order=4, ftype="butter"),
                           phase="zero", verbose="error")
        data = data.notch_filter(freqs=[60, 120, 180], picks="emg", method="iir",
                                 iir_params=dict(order=2, ftype="butter"),
                                 notch_widths=2, verbose="error")

        # ---- 3. normalise ----------------------------------------------------
        data = data.apply_function(lambda x: (x - x.min()) / (x.max() - x.min() + 1e-12),
                                   picks="emg", verbose="error", channel_wise=False)
        emg_unit = "a.u. (min-max)"
        emg_scale = 1.0          # already dimensionless

    # any description starting with "BAD" is honoured by reject_by_annotation.
    # "omit" drops the BAD_IK samples, so the remaining signal is NOT continuous
    # at the seams; fine for eyeballing, use "NaN" to keep the gaps visible.
    # NOTE: filtering above runs on the continuous data, so the seams do not ring.
    emg_clean = data.get_data(picks=emg_names, reject_by_annotation="omit") * emg_scale
    joints_clean = data.get_data(picks=joint_names, reject_by_annotation="omit")

    # carry the transient mask through the same BAD_IK omission so it lines up
    mask_clean = None
    if transient_mask is not None:
        keep = np.ones(data.n_times, dtype=bool)
        for onset, dur in zip(bad["onset"].to_numpy(), bad["duration"].to_numpy()):
            a = int(round(onset * sfreq))
            b = int(round((onset + dur) * sfreq))
            keep[max(a, 0):min(b, data.n_times)] = False
        mask_clean = transient_mask[:, keep]

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
        """Shade contiguous True runs of mask m (1-D) between y0 and y1."""
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
    save_root = pathlib.Path(r'F:\EEG_Challenge\Inspection_NoFilter')

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
                    remove_transients=True, shade_transients=True,
                )
