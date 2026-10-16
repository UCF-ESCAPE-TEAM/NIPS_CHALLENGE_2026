import mne
import numpy as np
import pathlib
import pandas as pd
import torch

def prepare_data(
        data_folder: pathlib.Path,
        subject: str='sub-02',
        session: str='ses-01',
        run: str='01',
        side: str='left'
):
    fname_stem = f"{subject}_{session}_task-emg2pose_run-{run}_recording-{side}"
    data_fname = f"{fname_stem}_emg.bdf"
    event_fname = f"{fname_stem}_events.tsv"
    channel_fname = f"{fname_stem}_channels.tsv"
    data = mne.io.read_raw_bdf(data_folder / subject / session / 'emg' / data_fname, preload=True, verbose='error')
    sfreq = data.info['sfreq']
    print(f"Sampling frequency: {sfreq} Hz")

    chans = pd.read_csv(data_folder / subject / session / 'emg' / channel_fname, sep='\t')
    emg_names = chans.loc[chans['type'] == 'EMG', 'name'].tolist()
    joint_names = chans.loc[chans['type'] == 'MISC', 'name'].tolist()

    # Optional but tidy: tell MNE the real channel types (it guessed "eeg" for all).
    data.set_channel_types({**{n: "emg" for n in emg_names},
                            **{n: "misc" for n in joint_names}}, verbose="error")

    print(data.info)
    emg_data = data.get_data(picks=emg_names) * 1e6  # convert to microvolts
    joints = data.get_data(picks=joint_names)
    print(f"The shape of the EMG data is {emg_data.shape}, and the shape of the joint data is {joints.shape}.")

    events = pd.read_csv(data_folder / subject / session / 'emg' / event_fname, sep="\t")
    bad = events[events["trial_type"] == "BAD_IK"]
    print(bad[["onset", "duration"]])
    data.set_annotations(mne.Annotations(onset=bad["onset"].to_numpy(),
                                        duration=bad["duration"].to_numpy(),
                                        description=["BAD_IK"] * len(bad)))
    # any description starting with "BAD" is honoured by reject_by_annotation
    emg_nan    = data.get_data(picks=emg_names,   reject_by_annotation="omit") * 1e6
    joints_nan = data.get_data(picks=joint_names, reject_by_annotation="omit")
    return emg_nan, joints_nan, sfreq, emg_names, joint_names


def sliding_window(data, window_size, step_size):
    r"""
    Generate the segmented EMG and Joint Angle Data
    """
    n_channels, n_samples = data.shape
    start_indices = np.arange(0, n_samples - window_size, step_size)
    print(f"The given data can be segmented into {len(start_indices)} segments with a window size of {window_size} and a step size of {step_size}.")
    segmented_data = np.array([data[:, start:start + window_size] for start in start_indices])
    return segmented_data

if __name__ == "__main__":
    data_folder = pathlib.Path(r'/Users/jiruifu/Documents/NIPS_Challenge_2026/sample_data_nm000281')
    emg_nan, joints_nan, sfreq, emg_names, joint_names = prepare_data(data_folder)
    print(f"EMG data shape: {emg_nan.shape}, Joint data shape: {joints_nan.shape}")
    ws = 10000
    ss = 10000
    emg_segments = sliding_window(emg_nan, ws, ss)
    joint_segments = sliding_window(joints_nan, ws, ss)
    print(f"Segmented EMG data shape: {emg_segments.shape}, Segmented Joint data shape: {joint_segments.shape}")
