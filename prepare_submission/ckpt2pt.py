import torch
import os
ckpt_file_path = '/home/ji045311/NIPS_2026_Challenge/Results/neuralbench.main.Experiment.run,1/seed=33,task_name=pose,loss.name=L1Loss,brain_model_output_size=20,brain_model_config.name=UlhmPose,delete_checkpoints_on_exit=F...509-ed517fc3/best.ckpt'
save_path = '/home/ji045311/NIPS_2026_Challenge/submission/ulhm_1dcnn'
if not os.path.exists(save_path):
    os.makedirs(save_path)
# Load the checkpoint
checkpoint = torch.load(ckpt_file_path, map_location=torch.device('cpu'))

print("best epoch:", checkpoint["epoch"])
state = {k.removeprefix("model."): v for k, v in checkpoint["state_dict"].items()}
torch.save(state, os.path.join(save_path, "weights.pt"))
print(f"Model weights saved to {os.path.join(save_path, 'weights.pt')}")