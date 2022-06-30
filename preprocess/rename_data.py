import shutil
import os
from tqdm import tqdm

if __name__ == '__main__':
    sinpath = '/mnt/qdata/share/rakuest1/data/UKB/raw/abdominal_MRI/raw/'
    pats = os.listdir(sinpath)

    for pat in tqdm(pats):
        try:
            shutil.move(os.path.join(sinpath, pat, 'T1_fat.nii.gz'), os.path.join(sinpath, pat, 'fat.nii.gz'))
            shutil.move(os.path.join(sinpath, pat, 'T1_in.nii.gz'), os.path.join(sinpath, pat, 'inp.nii.gz'))
            shutil.move(os.path.join(sinpath, pat, 'T1_opp.nii.gz'), os.path.join(sinpath, pat, 'opp.nii.gz'))
            shutil.move(os.path.join(sinpath, pat, 'T1_water.nii.gz'), os.path.join(sinpath, pat, 'wat.nii.gz'))
        except:
            continue