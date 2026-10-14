import pickle
import os

pkl_file_path = '/home/ji045311/NIPS_2026_Challenge/Results/neuralbench.main.Experiment.run,1/seed=33,task_name=pose,loss.name=L1Loss,brain_model_output_size=20,delete_checkpoints_on_exit=False,brain_model_config.name=Cnn1...545-e0e3cd47/submitit/1342145_0_result.pkl'
with open(pkl_file_path, 'rb') as f:
    data = pickle.load(f)
    print(data)