import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import mne
import pandas as pd
import pathlib
from channel_rejection import detect_high_rms_channels, detect_powerline_channels, arm_band_interpolate_channels


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
    if raw_data:
        emg_unit = "µV"
        emg_scale = 1e6          # MNE returns volts
    else:
        data = data.filter(l_freq=20, h_freq=450, picks="emg", verbose="error")
        data = data.notch_filter(freqs=[60, 120, 180], picks="emg", verbose="error", notch_widths=6, trans_bandwidth=2)
        data = data.apply_function(lambda x: (x - x.min()) / (x.max() - x.min() + 1e-12),
                                   picks="emg", verbose="error", channel_wise=False)
        emg_unit = "a.u. (min-max)"
        emg_scale = 1.0          # already dimensionless

    # any description starting with "BAD" is honoured by reject_by_annotation.
    # "omit" drops the BAD_IK samples, so the remaining signal is NOT continuous
    # at the seams; fine for eyeballing, use "NaN" to keep the gaps visible.
    emg_clean = data.get_data(picks=emg_names, reject_by_annotation="omit") * emg_scale
    joints_clean = data.get_data(picks=joint_names, reject_by_annotation="omit")

    emg_segments, starts = sliding_window(emg_clean, window_size, step_size)  # (N, 16, W)
    joints_segments, _ = sliding_window(joints_clean, window_size, step_size)  # (N, 20, W)
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
            fname = f"{fname_stem}_{sig_tag}_{tag}_ch{channel_id}_win{i:04d}.png"
        else:
            # all channels stacked with a constant vertical offset
            for c in range(n_channels):
                y = segments[i, c, :]
                y = y - np.nanmedian(y)                     # centre on this channel's baseline
                ax.plot(t, y + offsets[c], lw=0.5, color="k")
            ax.set_yticks(offsets)
            ax.set_yticklabels(ch_names)
            ax.set_ylabel(f"channel (lane height {spacing:.3g} {units}, median removed)")
            ax.set_ylim(offsets[-1] - spacing, offsets[0] + spacing)
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
                )

