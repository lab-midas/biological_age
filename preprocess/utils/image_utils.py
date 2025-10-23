# Copyright 2017, Wenjia Bai. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the 'License');
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an 'AS IS' BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import cv2
import numpy as np
import nibabel as nib
import tensorflow as tf
from scipy import ndimage
import scipy.ndimage.measurements as measure


def crop(x, s, c, shift_center=True):
    """
    For cardiac data: crop the input data to the desired size at the specified center.
    Args:
        x (ndarray): Input data.
        s (list of int): Desired size.
        c (list of int): Center.
        shift_center (bool): Shift the center of cropping so that the whole patch is inside the image (True),
            or symmetric-pad in case that patches extend beyond image borders (False).
    Returns:
        x_crop (ndarray): Cropped data.
    """

    if type(s) is not np.ndarray:
        s = np.asarray(s, dtype='f')

    if type(c) is not np.ndarray:
        c = np.asarray(c, dtype='f')

    if type(x) is not np.ndarray:
        x = np.asarray(x, dtype='f')

    m = np.asarray(np.shape(x), dtype='f')
    if len(m) < len(s):
        m = [m, np.ones(1, len(s) - len(m))]

    if np.sum(m == s) == len(m):
        return x

    def get_limits(sin, cin):
        if np.remainder(sin, 2) == 0:
            lower = cin + 1 + np.ceil(-sin / 2) - 1
            upper = cin + np.ceil(sin / 2)
        else:
            lower = cin + np.ceil(-sin / 2) - 1
            upper = cin + np.ceil(sin / 2) - 1
        return lower, upper

    idx = list()
    idx_pad = ()
    for n in range(np.size(s)):
        lower, upper = get_limits(s[n], c[n])

        # shift center
        if shift_center:
            if lower < 0:
                c[n] += np.abs(lower)

            if upper > m[n]:
                c[n] -= np.abs(upper - m[n])

            lower, upper = get_limits(s[n], c[n])

        # if even shifting the center is not helping, pad data
        if lower < 0:
            low_pad = int(np.abs(lower))
            lower = 0
            upper = s[n]
        else:
            low_pad = 0

        if upper > m[n]:
            upper_pad = int(np.abs(upper - m[n]))
            upper = m[n]
            lower = m[n] - s[n]
        else:
            upper_pad = 0

        idx.append(list(np.arange(lower, upper, dtype=int)))
        idx_pad += (low_pad, upper_pad),

    index_arrays = np.ix_(*idx)
    x = np.pad(x, idx_pad, mode='symmetric')
    return x[index_arrays]


def crop_image(image, cx, cy, size):
    """ Crop a 3D image using a bounding box centred at (cx, cy) with specified size """
    X, Y = image.shape[:2]
    r = int(size / 2)
    x1, x2 = cx - r, cx + r
    y1, y2 = cy - r, cy + r
    x1_, x2_ = max(x1, 0), min(x2, X)
    y1_, y2_ = max(y1, 0), min(y2, Y)
    # Crop the image
    crop = image[x1_: x2_, y1_: y2_]
    # Pad the image if the specified size is larger than the input image size
    if crop.ndim == 3:
        crop = np.pad(crop,
                      ((x1_ - x1, x2 - x2_), (y1_ - y1, y2 - y2_), (0, 0)),
                      'constant')
    elif crop.ndim == 4:
        crop = np.pad(crop,
                      ((x1_ - x1, x2 - x2_), (y1_ - y1, y2 - y2_), (0, 0), (0, 0)),
                      'constant')
    else:
        print('Error: unsupported dimension, crop.ndim = {0}.'.format(crop.ndim))
        exit(0)
    return crop


def normalise_intensity(image, thres_roi=10.0):
    """ Normalise the image intensity by the mean and standard deviation """
    val_l = np.percentile(image, thres_roi)
    roi = (image >= val_l)
    mu, sigma = np.mean(image[roi]), np.std(image[roi])
    eps = 1e-6
    image2 = (image - mu) / (sigma + eps)
    return image2


def rescale_intensity(image, thres=(1.0, 99.0)):
    """ Rescale the image intensity to the range of [0, 1] """
    val_l, val_h = np.percentile(image, thres)
    image2 = image
    image2[image < val_l] = val_l
    image2[image > val_h] = val_h
    image2 = (image2.astype(np.float32) - val_l) / (val_h - val_l)
    return image2


def get_largest_cc(binary):
    """ Get the largest connected component in the foreground. """
    cc, n_cc = measure.label(binary)
    max_n = -1
    max_area = 0
    for n in range(1, n_cc + 1):
        area = np.sum(cc == n)
        if area > max_area:
            max_area = area
            max_n = n
    largest_cc = (cc == max_n)
    return largest_cc


def remove_small_cc(binary, thres=10):
    """ Remove small connected component in the foreground. """
    cc, n_cc = measure.label(binary)
    binary2 = np.copy(binary)
    for n in range(1, n_cc + 1):
        area = np.sum(cc == n)
        if area < thres:
            binary2[cc == n] = 0
    return binary2


def split_sequence(image_name, output_name):
    """ Split an image sequence into a number of time frames. """
    nim = nib.load(image_name)
    T = nim.header['dim'][4]
    affine = nim.affine
    image = nim.get_data()

    for t in range(T):
        image_fr = image[:, :, :, t]
        nim2 = nib.Nifti1Image(image_fr, affine)
        nib.save(nim2, '{0}{1:02d}.nii.gz'.format(output_name, t))


def make_sequence(image_names, dt, output_name):
    """ Combine a number of time frames into one image sequence. """
    nim = nib.load(image_names[0])
    affine = nim.affine
    X, Y, Z = nim.header['dim'][1:4]
    T = len(image_names)
    image = np.zeros((X, Y, Z, T))

    for t in range(T):
        image[:, :, :, t] = nib.load(image_names[t]).get_data()

    nim2 = nib.Nifti1Image(image, affine)
    nim2.header['pixdim'][4] = dt
    nib.save(nim2, output_name)


def split_volume(image_name, output_name):
    """ Split an image volume into a number of slices. """
    nim = nib.load(image_name)
    Z = nim.header['dim'][3]
    affine = nim.affine
    image = nim.get_data()

    for z in range(Z):
        image_slice = image[:, :, z]
        image_slice = np.expand_dims(image_slice, axis=2)
        affine2 = np.copy(affine)
        affine2[:3, 3] += z * affine2[:3, 2]
        nim2 = nib.Nifti1Image(image_slice, affine2)
        nib.save(nim2, '{0}{1:02d}.nii.gz'.format(output_name, z))


def image_apply_mask(input_name, output_name, mask_image, pad_value=-1):
    # Assign the background voxels (mask == 0) with pad_value
    nim = nib.load(input_name)
    image = nim.get_data()
    image[mask_image == 0] = pad_value
    nim2 = nib.Nifti1Image(image, nim.affine)
    nib.save(nim2, output_name)


def padding(input_A_name, input_B_name, output_name, value_in_B, value_output):
    nim = nib.load(input_A_name)
    image_A = nim.get_data()
    image_B = nib.load(input_B_name).get_data()
    image_A[image_B == value_in_B] = value_output
    nim2 = nib.Nifti1Image(image_A, nim.affine)
    nib.save(nim2, output_name)


def auto_crop_image(input_name, output_name, reserve):
    nim = nib.load(input_name)
    image = nim.get_data()
    X, Y, Z = image.shape[:3]

    # Detect the bounding box of the foreground
    idx = np.nonzero(image > 0)
    x1, x2 = idx[0].min() - reserve, idx[0].max() + reserve + 1
    y1, y2 = idx[1].min() - reserve, idx[1].max() + reserve + 1
    z1, z2 = idx[2].min() - reserve, idx[2].max() + reserve + 1
    x1, x2 = max(x1, 0), min(x2, X)
    y1, y2 = max(y1, 0), min(y2, Y)
    z1, z2 = max(z1, 0), min(z2, Z)
    print('Bounding box')
    print('  bottom-left corner = ({},{},{})'.format(x1, y1, z1))
    print('  top-right corner = ({},{},{})'.format(x2, y2, z2))

    # Crop the image
    image = image[x1:x2, y1:y2, z1:z2]

    # Update the affine matrix
    affine = nim.affine
    affine[:3, 3] = np.dot(affine, np.array([x1, y1, z1, 1]))[:3]
    nim2 = nib.Nifti1Image(image, affine)
    nib.save(nim2, output_name)