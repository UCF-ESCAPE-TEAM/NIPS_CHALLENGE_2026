import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import mne
import pandas as pd
import pathlib
from scipy.signal import welch

def plot_windowed_psd(data_folder: pathlib.Path,
                      save_folder: pathlib.Path,
                      fname_stem: str,
                      window_size: int,
                      step_size: int,
                      signal_type: str,
                      sfreq: float = 2000,
                      raw_data: bool = False,
                      channel_id: int = None,
                      max_windows: int = None,
                      fmax: float = None,
                      nperseg: int = 2048,
                      mark_mains: float = 60.0,
                      dpi: int = 120):
    """
    Plot the Welch power spectral density of each window of either the EMG or the Joint
    Angle data. Same loading / filtering / windowing logic as ``plot_windowed_data``.
    Args:
        data_folder: The path to the data folder.
        save_folder: The path to the save folder.
        fname_stem: The stem of the data file.
        window_size: The size of the window in samples.
        step_size: The step size in samples.
        signal_type: The type of signal to plot, either "EMG" or "Joint Angle".
        sfreq: The sampling frequency of the data (overwritten by the file header).
        raw_data: If True use the raw signal; if False band-pass + notch + min-max the EMG first.
        channel_id: Channel to plot; None plots all channels on the same axes (one colour each).
        max_windows: Stop after this many windows (None = all).
        fmax: Upper frequency limit of the plot (default: Nyquist for EMG, 20 Hz for joints).
        nperseg: Welch segment length in samples (2048 @ 2 kHz -> ~1 Hz resolution).
        mark_mains: Draw dashed lines at this frequency and its harmonics (None to disable).
        dpi: Resolution of the saved PNG files.
    Returns:
        List of the saved figure paths.
    """
    assert signal_type in ("EMG", "Joint Angle"), f"unknown signal_type {signal_type!r}"

    def sliding_window(data, window_size, step_size):
        n_channels, n_samples = data.shape
        start_indices = np.arange(0, n_samples - window_size + 1, step_size)
        print(f"The given data can be segmented into {len(start_indices)} segments with a window size of {window_size} and a step size of {step_size}.")
        segmented_data = np.array([data[:, start:start + window_size] for start in start_indices])
        return segmented_data, start_indices

    # ---------------------------------------------------------------- load
    data_fname = f"{fname_stem}_emg.bdf"
    event_fname = f"{fname_stem}_events.tsv"
    channel_fname = f"{fname_stem}_channels.tsv"
    data = mne.io.read_raw_bdf(data_folder / data_fname, preload=True, verbose='error')
    sfreq = data.info['sfreq']
    print(f"Sampling frequency: {sfreq} Hz")
    chans = pd.read_csv(data_folder / channel_fname, sep='\t')
    emg_names = chans.loc[chans['type'] == 'EMG', 'name'].tolist()
    joint_names = chans.loc[chans['type'] == 'MISC', 'name'].tolist()
    data.set_channel_types({**{n: "emg" for n in emg_names},
                            **{n: "misc" for n in joint_names}}, verbose="error")

    events = pd.read_csv(data_folder / event_fname, sep="\t")
    bad = events[events["trial_type"] == "BAD_IK"]
    data.set_annotations(mne.Annotations(onset=bad["onset"].to_numpy(),
                                         duration=bad["duration"].to_numpy(),
                                         description=["BAD_IK"] * len(bad)))
    if raw_data:
        emg_unit = "µV"
        emg_scale = 1e6
    else:
        data = data.filter(l_freq=20, h_freq=450, picks="emg", verbose="error")
        data = data.notch_filter(freqs=[60, 120, 180], picks="emg", verbose="error")
        data = data.apply_function(lambda x: (x - x.min()) / (x.max() - x.min() + 1e-12),
                                   picks="emg", verbose="error", channel_wise=False)
        emg_unit = "a.u."
        emg_scale = 1.0

    emg_clean = data.get_data(picks=emg_names, reject_by_annotation="omit") * emg_scale
    joints_clean = data.get_data(picks=joint_names, reject_by_annotation="omit")

    emg_segments, starts = sliding_window(emg_clean, window_size, step_size)
    joints_segments, _ = sliding_window(joints_clean, window_size, step_size)
    segments = {"EMG": emg_segments, "Joint Angle": joints_segments}[signal_type]
    ch_names = {"EMG": emg_names, "Joint Angle": joint_names}[signal_type]
    units = {"EMG": emg_unit, "Joint Angle": "rad"}[signal_type]

    n_windows, n_channels, _ = segments.shape
    if channel_id is not None and channel_id > n_channels - 1:
        print(f"channel_id {channel_id} out of range, plotting the last channel ({n_channels - 1}) instead")
        channel_id = n_channels - 1
    if max_windows is not None:
        n_windows = min(n_windows, max_windows)
    if fmax is None:
        fmax = sfreq / 2 if signal_type == "EMG" else 20.0
    nperseg = min(nperseg, window_size)

    # ------------------------------------------------------------- PSD
    # Welch on every window and channel at once: (N, C, F)
    freqs, psd = welch(segments - segments.mean(axis=-1, keepdims=True),
                       fs=sfreq, nperseg=nperseg, noverlap=nperseg // 2, axis=-1)
    keep = freqs <= fmax
    freqs, psd = freqs[keep], psd[..., keep]
    psd_db = 10 * np.log10(psd + 1e-20)                          # dB re 1 unit^2/Hz

    # common y-limits across windows so figures are comparable
    sel = psd_db[:n_windows] if channel_id is None else psd_db[:n_windows, channel_id]
    y_lo, y_hi = np.percentile(sel, 0.5) - 3, sel.max() + 3     # never clip a peak

    save_folder = pathlib.Path(save_folder)
    save_folder.mkdir(parents=True, exist_ok=True)
    tag = "raw" if raw_data else "filtered"
    sig_tag = signal_type.replace(" ", "")
    colors = plt.cm.viridis(np.linspace(0, 1, n_channels))
    saved = []

    for i in range(n_windows):
        t0 = starts[i] / sfreq
        t1 = (starts[i] + window_size) / sfreq
        title = (f"PSD of window {i} ({t0:.2f}-{t1:.2f} s) of the {signal_type} data ({tag}) "
                 f"for data file\n{data_fname}")
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))

        if channel_id is not None:
            ax.plot(freqs, psd_db[i, channel_id], lw=1.0, color="k", label=ch_names[channel_id])
            fname = f"{fname_stem}_{sig_tag}_{tag}_ch{channel_id}_psd_win{i:04d}.png"
        else:
            for c in range(n_channels):
                ax.plot(freqs, psd_db[i, c], lw=0.8, color=colors[c], label=ch_names[c])
            fname = f"{fname_stem}_{sig_tag}_{tag}_all_psd_win{i:04d}.png"
        ax.legend(ncol=4, fontsize=7, loc="upper right", framealpha=0.8)

        if mark_mains and signal_type == "EMG":
            for k in range(1, int(fmax // mark_mains) + 1):
                ax.axvline(k * mark_mains, color="r", ls="--", lw=0.6, alpha=0.5)

        ax.set_xlim(0, fmax)
        ax.set_ylim(y_lo, y_hi)
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel(f"PSD (dB re 1 {units}$^2$/Hz)")
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        out = save_folder / fname
        fig.savefig(out, dpi=dpi)
        plt.close(fig)
        saved.append(out)

    print(f"Saved {len(saved)} PSD figure(s) to {save_folder}")
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
    save_root = pathlib.Path(r'F:\EEG_Challenge\Inspection_psd')

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
                plot_windowed_psd(
                    data_folder, save_folder, fname_stem,
                    window_size, step_size, signal_type, sfreq, raw_data, channel_id,
                )