import os

def delete_numeric_directories(path):
    for dir_name in os.listdir(path):
        dir_path = os.path.join(path, dir_name)
        if os.path.isdir(dir_path) and dir_name.isdigit():
            print(f"Deleting directory: {dir_path}")
            os.rmdir(dir_path)

# Specify the path to the directory you want to process
path = "/mnt/qdata/share/raecker1/ukbdata_70k/sa_heart/processed/seg"

delete_numeric_directories(path)