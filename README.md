
# Imaging-derived biological age across multiple organs links to mortality and aging-related health outcomes

## 1. Installation

Install required packages defined in requirements.txt in virtual environment

## 2. Preprocessing

Prepare image data:
- run [ukbbrain.py](https://github.com/lab-midas/biological_age/blob/master/preprocess/ukbbrain.py), [ukbabdomen.py](https://github.com/lab-midas/biological_age/blob/master/preprocess/ukbabdomen.py), [ukbfundus.py](https://github.com/lab-midas/biological_age/blob/master/preprocess/ukbfundus.py), and [heart.py](https://github.com/lab-midas/biological_age/blob/master/preprocess/heart.py) for preprocessing of the nifti imaging data of the UK Biobank. Includes various organ-specific preprocessing steps and h5 file creation.
- run [nakobrain.py](https://github.com/lab-midas/biological_age/blob/master/preprocess/nakobrain.py), [nakoabdomen.py](https://github.com/lab-midas/biological_age/blob/master/preprocess/nakoabdomen.py), and [heart.py](https://github.com/lab-midas/biological_age/blob/master/preprocess/heart.py) for preprocessing of the nifti imaging data of the NAKO. Includes various organ-specific preprocessing steps and h5 file creation.

Prepare training and testing keys:
- [filter_keys.py](https://github.com/lab-midas/biological_age/blob/master/preprocess/filter_keys.py): run filter_keys to filter healthy and diseased subjects (organ-specific diseases).
- [create_keys.py](https://github.com/lab-midas/biological_age/blob/master/preprocess/create_keys.py): run create_keys to create all key files (train, test, gradcam).

## 3. Training

Run train3d.py in [trainer](https://github.com/lab-midas/biological_age/tree/master/brainage/trainer) folder.
