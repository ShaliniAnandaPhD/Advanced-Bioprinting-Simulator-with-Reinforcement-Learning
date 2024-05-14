import numpy as np
import nibabel as nib
from scipy.ndimage import zoom

def load_nifti_data(file_path):
    """
    Load a NIfTI file and return the image data as a numpy array.
    """
    img = nib.load(file_path)
    data = img.get_fdata()
    return data

def save_nifti_data(data, file_path):
    """
    Save a numpy array as a NIfTI file.
    """
    affine = np.eye(4)
    img = nib.Nifti1Image(data, affine)
    nib.save(img, file_path)

def preprocess_organ_data(data, target_resolution):
    """
    Preprocess the organ data by resampling it to the target resolution.
    """
    zoom_factors = tuple(target_dimension / original_dimension for target_dimension, original_dimension in zip(target_resolution, data.shape))
    preprocessed_data = zoom(data, zoom_factors, order=1)
    return preprocessed_data

def extract_organ_region(data, threshold=0.5):
    """
    Extract the organ region from the image data based on a threshold value.
    """
    organ_region = np.where(data > threshold, 1, 0)
    return organ_region

def generate_material_distribution(organ_data, num_materials):
    """
    Generate a material distribution based on the organ data.
    """
    material_distribution = np.zeros(organ_data.shape + (num_materials,))
    
    # Assign materials based on the intensity values of the organ data
    intensity_ranges = np.linspace(organ_data.min(), organ_data.max(), num_materials + 1)
    for i in range(num_materials):
        material_indices = np.logical_and(organ_data >= intensity_ranges[i], organ_data < intensity_ranges[i + 1])
        material_distribution[material_indices, i] = 1
    
    return material_distribution

def convert_to_voxel_data(data, resolution):
    """
    Convert the data to voxel representation based on the specified resolution.
    """
    voxel_data = np.zeros(resolution)
    zoom_factors = tuple(target_dimension / original_dimension for target_dimension, original_dimension in zip(resolution, data.shape))
    voxel_data = zoom(data, zoom_factors, order=0)
    return voxel_data

def convert_to_mesh_data(voxel_data, threshold=0.5):
    """
    Convert the voxel data to mesh representation based on a threshold value.
    """
    # Use marching cubes algorithm to extract the mesh from the voxel data
    from skimage.measure import marching_cubes
    vertices, faces, _, _ = marching_cubes(voxel_data, level=threshold)
    return vertices, faces