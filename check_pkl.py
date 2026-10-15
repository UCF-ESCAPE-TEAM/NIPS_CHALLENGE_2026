import pickle
import os

pkl_file_path = '/home/ji045311/NIPS_2026_Challenge/Results/neuralbench.main.Experiment.run,1/seed=33,task_name=pose,loss.name=L1Loss,brain_model_output_size=20,delete_checkpoints_on_exit=False,augmentation={num_bands=1,pr...530-89ad2378/best.ckpt'
with open(pkl_file_path, 'rb') as f:
    data = pickle.load(f)
    print(data)