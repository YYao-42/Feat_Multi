import subprocess
import os
from multiprocessing import Pool

video_folder = rf"C:\Users\Gebruiker\Documents\Experiments\downsamp_video"
# list all mp4 files that do not end with _output
video_files = [f for f in os.listdir(video_folder) if f.endswith('.mp4') and not f.endswith('_output.mp4')]
features = ['ObjTempCtr', 'ObjRMSCtr', 'RMSCtr', 'ObjFlow']

script_name = 'feats_extrac.py'

# list all experiments
experiments = []
for video_file in video_files:
    for feature in features:
        experiments.append({'videoname': video_file, 'featurename': feature})


# Function to run a single experiment
def run_experiment(arg_set):
    cmd = ['python', script_name]
    for key, value in arg_set.items():
        if isinstance(value, bool):
            if value:
                cmd.append(f'--{key}')
        elif isinstance(value, list):
            cmd.append(f'--{key}')
            cmd.extend(map(str, value))
        else:
            cmd.append(f'--{key}={value}')
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd)

if __name__ == "__main__":
    # You can change this to the number of parallel processes you want
    with Pool(processes=1) as pool:
        pool.map(run_experiment, experiments)
