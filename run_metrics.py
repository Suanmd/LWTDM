import sys
import subprocess

arg1 = sys.argv[1]  # sr4_01_ddim
arg2 = sys.argv[2]  # 3

command1 = f"python experiments/check_fid.py {arg1}"
command2 = f"python -m pytorch_fid experiments/{arg1}/results1 experiments/{arg1}/results2 --device cuda:{arg2}"
command3 = f"python eval.py -p experiments/{arg1}/results"

print(f"Running command 1: {command1}")
subprocess.run(command1, shell=True)

print(f"Running command 2: {command2}")
subprocess.run(command2, shell=True)

print(f"Running command 3: {command3}")
subprocess.run(command3, shell=True)
