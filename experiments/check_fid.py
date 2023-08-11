import os
import sys
import shutil

def copy_files(source_dir, destination_dir, file_extension):
    if not os.path.exists(destination_dir):
        os.makedirs(destination_dir)

    files = os.listdir(source_dir)
    for file in files:
        if file.endswith(file_extension):
            file_path = os.path.join(source_dir, file)
            shutil.copy(file_path, destination_dir)

# 获取命令行参数
folder_A = os.path.join('experiments', sys.argv[1], 'results')  # 文件夹A的路径
folder_B = os.path.join('experiments', sys.argv[1], 'results1')  # 新文件夹B的路径
folder_C = os.path.join('experiments', sys.argv[1], 'results2')  # 新文件夹C的路径

# 移动以"_hr.png"结尾的文件到文件夹B
copy_files(folder_A, folder_B, "_hr.png")

# 移动以"_sr.png"结尾的文件到文件夹C
copy_files(folder_A, folder_C, "_sr.png")

