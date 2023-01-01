import os
import shutil
files = os.listdir('/home/studxusiy1/mr_recon/03_cine2dt/preprocessing/h5_compressed_new_name')

for file in files:
    shutil.copyfile(os.path.joinpath('/home/studxusiy1/mr_recon/03_cine2dt/preprocessing/h5_compressed_new_name', file), os.path.joinpath('/mnt/qdata/rawdata/CINE/2D_h5_compressed', file[:-3]))