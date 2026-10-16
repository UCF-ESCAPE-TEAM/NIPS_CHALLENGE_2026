
import mne
import numpy as np
import pathlib
import pandas as pd

# 1_load a sample data to test the model

data_folder = pathlib.Path(r'/Users/jiruifu/Documents/NIPS_Challenge_2026/sample_data_nm000281')
subject = 'sub-02'
session = 'ses-01'
run = '01'
side = 'left'
plot = False

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

if plot:
    data.plot(
        picks = "emg",
        duration = 10,
        start = 0,
        n_channels = 1,
        scalings = "auto",
        title = 'EMG',
        block = True
    )

data_f = (
    data.copy()
    .filter(l_freq=20, h_freq=450, picks="emg", verbose="error")
    .notch_filter(freqs=[60, 120, 180], picks="emg", verbose="error")
)

if plot:
    data.plot(
        picks = "emg",
        duration = 10,
        start = 0,
        n_channels = 1,
        scalings = "auto",
        title = 'EMG',
        block = True
    )

emg = data_f.get_data(picks=emg_names, reject_by_annotation="omit") * 1e6  # convert to microvolts
joint_angles = data_f.get_data(picks=joint_names, reject_by_annotation="omit")
print(f"The shape of the EMG data is {emg.shape}, and the shape of the joint data is {joint_angles.shape}.")