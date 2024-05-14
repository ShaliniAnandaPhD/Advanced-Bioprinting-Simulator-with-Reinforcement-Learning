import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def create_directory(directory):
    """
    Create a directory if it doesn't exist.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)

def save_numpy_array(array, file_path):
    """
    Save a NumPy array to a file.
    """
    np.save(file_path, array)

def load_numpy_array(file_path):
    """
    Load a NumPy array from a file.
    """
    array = np.load(file_path)
    return array

def plot_bioprinted_structure(bioprint_env, title=None, save_path=None):
    """
    Plot the bioprinted structure in 3D.
    """
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Get the indices of non-zero voxels
    x, y, z = np.where(bioprint_env > 0)
    
    # Plot the scatter points
    ax.scatter(x, y, z, c=bioprint_env[x, y, z], cmap='viridis', alpha=0.8)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    if title:
        ax.set_title(title)
    
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()

def plot_training_progress(log_file, save_path=None):
    """
    Plot the training progress from a log file.
    """
    data = np.genfromtxt(log_file, delimiter=',', skip_header=1, names=['episode', 'reward', 'loss'])
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    ax1.plot(data['episode'], data['reward'])
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Reward')
    ax1.set_title('Training Reward')
    
    ax2.plot(data['episode'], data['loss'])
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Loss')
    ax2.set_title('Training Loss')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()

def print_progress_bar(iteration, total, prefix='', suffix='', length=50, fill='█'):
    """
    Print a progress bar.
    """
    percent = int(100 * (iteration / total))
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end='', flush=True)
    if iteration == total:
        print()