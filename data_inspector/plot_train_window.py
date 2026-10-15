"""Save one training window (16-ch EMG + 20 joint angles) of the emg/pose task
to PNG, optionally with a filter applied to the extracted EMG.

Usage:
    python plot_train_window.py                              # raw EMG only
    python plot_train_window.py --highpass 20                # raw vs. 20 Hz high-pass
    python plot_train_window.py --highpass 20 --notch 60     # + 60 Hz notch (and harmonics)
    python plot_train_window.py --bandpass 20 450 --normalize
    python plot_train_window.py --index 42 --split val --out figs

With any filter option the figure has three panels: raw EMG, filtered EMG,
joint angles. Without one it has two. Filters are zero-phase (filtfilt) SOS
Butterworth / IIR notch from scipy.signal, applied per channel to the 5-s
window exactly as a model front end would see it. Nothing is displayed: the
Agg backend is used and the figure is written to <out>/<split>_window_<index>.png.
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: never open a window
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

from neuralbench.data import get_default_dataloaders

SFREQ = 2000.0


# --------------------------------------------------------------------- filtering
def filter_emg(
    x: np.ndarray,
    sfreq: float = SFREQ,
    highpass: float | None = None,
    lowpass: float | None = None,
    notch: float | None = None,
    order: int = 4,
    notch_q: float = 30.0,
    normalize: bool = False,
) -> np.ndarray:
    """Filter a (C, T) EMG window. Returns a new array; the input is untouched.

    Steps, in order (each skipped when its argument is None/False):
      1. notch at ``notch`` Hz and its harmonics below Nyquist (IIR, Q=notch_q)
      2. Butterworth high-pass / low-pass / band-pass of ``order`` (SOS, zero-phase)
      3. per-channel z-score over the window (what InstanceNorm1d does in the model)
    """
    y = np.asarray(x, dtype=np.float64)
    nyq = sfreq / 2.0

    if notch is not None:
        f0 = notch
        while f0 < nyq:
            b, a = signal.iirnotch(f0, notch_q, fs=sfreq)
            y = signal.filtfilt(b, a, y, axis=-1)
            f0 += notch

    if highpass is not None and lowpass is not None:
        sos = signal.butter(order, [highpass, lowpass], btype="bandpass", fs=sfreq, output="sos")
    elif highpass is not None:
        sos = signal.butter(order, highpass, btype="highpass", fs=sfreq, output="sos")
    elif lowpass is not None:
        sos = signal.butter(order, lowpass, btype="lowpass", fs=sfreq, output="sos")
    else:
        sos = None
    if sos is not None:
        y = signal.sosfiltfilt(sos, y, axis=-1)

    if normalize:
        y = (y - y.mean(-1, keepdims=True)) / (y.std(-1, keepdims=True) + 1e-5)

    return y.astype(np.float32)


# ----------------------------------------------------------------------- plotting
def _stack(ax, t, x, ylabel, title=None):
    """Plot (C, T) channels stacked with a constant offset."""
    offset = 3 * x.std()
    for c in range(x.shape[0]):
        ax.plot(t, x[c] + c * offset, lw=0.4, color="k")
    ax.set_yticks(np.arange(x.shape[0]) * offset)
    ax.set_yticklabels([f"ch{c}" for c in range(x.shape[0])])
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="train", choices=["train", "val", "test"])
    parser.add_argument("--index", type=int, default=None,
                        help="dataset index; default = first window of the first batch")
    parser.add_argument("--out", default="figs", help="output directory")
    parser.add_argument("--dpi", type=int, default=150)
    # filter options
    parser.add_argument("--highpass", type=float, default=None, help="high-pass cutoff (Hz)")
    parser.add_argument("--lowpass", type=float, default=None, help="low-pass cutoff (Hz)")
    parser.add_argument("--bandpass", type=float, nargs=2, metavar=("LOW", "HIGH"),
                        help="band-pass cutoffs (Hz); shorthand for --highpass LOW --lowpass HIGH")
    parser.add_argument("--notch", type=float, default=None, help="notch frequency (Hz), harmonics too")
    parser.add_argument("--order", type=int, default=4, help="Butterworth order")
    parser.add_argument("--normalize", action="store_true",
                        help="per-channel z-score over the window after filtering")
    args = parser.parse_args()
    if args.bandpass:
        args.highpass, args.lowpass = args.bandpass
    do_filter = any(v is not None for v in (args.highpass, args.lowpass, args.notch)) or args.normalize

    # ---- extract one window exactly as the task config windows it
    loaders = get_default_dataloaders(
        "emg", "pose",
        batch_size=4,
        **{"neuro.infra.cluster": None,  # extract in-process, no SLURM
           "target.infra.cluster": None},
    )
    loader = loaders[args.split]
    if args.index is None:
        batch, i, index = next(iter(loader)), 0, 0
    else:
        batch, i, index = loader.dataset[args.index], 0, args.index

    x = batch.data["neuro"][i].cpu().numpy()    # (16, 10000) uV, raw
    y = batch.data["target"][i].cpu().numpy()   # (20, 10000) radians
    ev = batch.segments[i].events.iloc[0]
    user, stage = ev.get("user", "?"), ev.get("stage", "?")
    t = np.arange(x.shape[1]) / SFREQ

    # ---- figure
    n_rows = 3 if do_filter else 2
    fig, axes = plt.subplots(n_rows, 1, figsize=(12, 3.5 * n_rows), sharex=True)
    title = f"{args.split} window #{index} — user {user}, stage {stage}"
    _stack(axes[0], t, x, "raw EMG (µV, offset)", title)

    if do_filter:
        xf = filter_emg(x, SFREQ, args.highpass, args.lowpass, args.notch,
                        args.order, normalize=args.normalize)
        parts = []
        if args.highpass is not None: parts.append(f"HP {args.highpass:g} Hz")
        if args.lowpass is not None: parts.append(f"LP {args.lowpass:g} Hz")
        if args.notch is not None: parts.append(f"notch {args.notch:g} Hz")
        if args.normalize: parts.append("z-score")
        unit = "z" if args.normalize else "µV"
        _stack(axes[1], t, xf, f"filtered EMG ({unit}, offset)", ", ".join(parts))

    axes[-1].plot(t, np.degrees(y.T), lw=0.8)
    axes[-1].set_ylabel("joint angle (°)")
    axes[-1].set_xlabel("time (s)")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.split}_window_{index}.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=args.dpi)
    plt.close(fig)
    print(f"saved {out_path.resolve()}")


if __name__ == "__main__":
    main()