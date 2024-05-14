import numpy as np
from sklearn.metrics import jaccard_score, hausdorff_distance

def jaccard_similarity(bioprint_env, target_structure):
    """
    Calculate the Jaccard similarity between the bioprinted structure and the target structure.
    """
    bioprint_binary = bioprint_env > 0.5
    target_binary = target_structure > 0.5
    jaccard = jaccard_score(target_binary.flatten(), bioprint_binary.flatten())
    return jaccard

def hausdorff_distance(bioprint_env, target_structure):
    """
    Calculate the Hausdorff distance between the bioprinted structure and the target structure.
    """
    bioprint_coords = np.argwhere(bioprint_env > 0.5)
    target_coords = np.argwhere(target_structure > 0.5)
    hausdorff = hausdorff_distance(bioprint_coords, target_coords)
    return hausdorff

def volume_overlap(bioprint_env, target_structure):
    """
    Calculate the volume overlap between the bioprinted structure and the target structure.
    """
    bioprint_volume = np.sum(bioprint_env > 0.5)
    target_volume = np.sum(target_structure > 0.5)
    overlap_volume = np.sum((bioprint_env > 0.5) & (target_structure > 0.5))
    overlap_ratio = overlap_volume / (bioprint_volume + target_volume - overlap_volume)
    return overlap_ratio

def surface_smoothness(bioprint_env):
    """
    Calculate the surface smoothness of the bioprinted structure.
    """
    # Apply a Gaussian filter to smooth the bioprinted structure
    smoothed_env = gaussian_filter(bioprint_env, sigma=1)
    
    # Calculate the surface gradient magnitude
    gradient_magnitude = np.linalg.norm(np.gradient(smoothed_env), axis=0)
    
    # Calculate the average surface smoothness
    smoothness = 1 - np.mean(gradient_magnitude)
    return smoothness

def evaluate_bioprinted_structure(bioprint_env, target_structure):
    """
    Evaluate the bioprinted structure using multiple metrics.
    """
    jaccard = jaccard_similarity(bioprint_env, target_structure)
    hausdorff = hausdorff_distance(bioprint_env, target_structure)
    overlap = volume_overlap(bioprint_env, target_structure)
    smoothness = surface_smoothness(bioprint_env)
    
    evaluation_results = {
        'Jaccard Similarity': jaccard,
        'Hausdorff Distance': hausdorff,
        'Volume Overlap': overlap,
        'Surface Smoothness': smoothness
    }
    
    return evaluation_results

def print_evaluation_results(evaluation_results):
    """
    Print the evaluation results in a readable format.
    """
    print("Evaluation Results:")
    for metric, value in evaluation_results.items():
        print(f"{metric}: {value:.4f}")

if __name__ == '__main__':
    # Load the bioprinted structure and target structure
    bioprint_env = np.load('bioprinted_structure.npy')
    target_structure = np.load('target_structure.npy')
    
    # Evaluate the bioprinted structure
    evaluation_results = evaluate_bioprinted_structure(bioprint_env, target_structure)
    
    # Print the evaluation results
    print_evaluation_results(evaluation_results)