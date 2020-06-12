import subprocess
import argparse
import multiprocessing as mp
import json
import os
import math
import random

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, default=None, help="Directory of data files.")
args = parser.parse_args()
DATA_DIR = args.data_dir

os.system("cp -r " + DATA_DIR + " .")
folder_name = DATA_DIR[DATA_DIR.rfind("/")+1:]
n_folders = len(os.listdir(folder_name))


group_size = math.ceil(n_folders/8)
processes_and_folders = []
folders = os.listdir(folder_name)
random.shuffle(folders)

for i in range(0, n_folders, group_size):
	end = min(n_folders, i + group_size)
	help_folder_name = "help_folder{}".format(int(i/group_size))
	os.mkdir(help_folder_name)
	cp_folders = [os.path.join(folder_name, word_folder) for word_folder in folders[i:end]]
	os.system("cp -r "  + " ".join(cp_folders) + " " + help_folder_name)
	args = ["python3", "training_bert.py", "--data_dir", help_folder_name, "--save_path", os.path.join(help_folder_name, 'results.json')]
	p = subprocess.Popen(args)
	processes_and_folders.append([p, help_folder_name])


save_dict = {}
for p , help_folder_name in processes_and_folders:
	p.wait() 
	with open(os.path.join(help_folder_name, 'results.json')) as f:
 		data = json.load(f)
 		save_dict.update(data)
	os.system("rm -r " + help_folder_name)

os.system("rm -r " + folder_name)

with open('results.json', 'w') as f:
	json.dump(save_dict, f)






