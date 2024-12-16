import h5py
import numpy as np
from scipy.ndimage import zoom
import matplotlib.pyplot as plt

def downsample_image(image, target_dim, target_size):
    """
    Downsample a 3D image to a new size along a given dimension, preserving aspect ratio.

    Parameters:
    - image: 3D numpy array (e.g., shape (depth, height, width))
    - target_dim: int (0 for depth, 1 for height, 2 for width)
    - target_size: int (desired size for the specified dimension)

    Returns:
    - downsampled_image: 3D numpy array with the new size, maintaining aspect ratio
    """
    # Get the original dimensions of the image
    original_shape = np.array(image.shape)
    print(f'original_shape: {original_shape}')
    
    # Calculate the scaling factor for the target dimension
    scale_factor = target_size / original_shape[target_dim]
    print(f'scale_factor: {scale_factor}')
    
    # Calculate the new shape for all dimensions
    new_shape = (original_shape * scale_factor).astype(int)
    print(f'new_shape: {new_shape}')
    
    # Resize the image using the scaling factor for each dimension
    scaling_factors = new_shape / original_shape
    print(f'scaling_factors: {scaling_factors}')
    downsampled_image = zoom(image, scaling_factors, order=1)  # order=1 is bilinear interpolation

    return downsampled_image

if __name__ == '__main__':
    # Open the HDF5 file
    #file_path_new = '/mnt/qdata/share/rakuest1/data/UKB/interim/ukb_pnc_preprocessed.h5'
    file_path_new = '/mnt/qdata/share/raecker1/nako_test/water_120235.h5'

    with h5py.File(file_path_new, 'r') as hdf_new:
        # Get the keys of the new HDF5 file
        keys_new = list(hdf_new.keys())
        print(keys_new)
        data = hdf_new['image']
        data_np = np.array(data)
        plt.imshow(data_np[40, :, :])
        plt.savefig('dummy.png')
        data_new = downsample_image(data_np, 0, 60)
        print(data_new.shape)
        plt.imshow(data_new[30, :, :])
        plt.savefig('dummy_new.png')
        print('done')