# Bioprinting Environment Configuration
ENV_CONFIG = {
    'resolution': (64, 64, 64),  # Resolution of the bioprinting environment
    'materials': ['Hydrogel', 'Collagen', 'Gelatin'],  # Materials used in the bioprinting process
    'extrusion_rate': 0.05,  # Extrusion rate of the bioprinter
    'diffusion_rate': 0.01,  # Diffusion rate of the materials
    'settling_rate': 0.02,  # Settling rate of the materials due to gravity
    'max_steps': 1000  # Maximum number of steps in the bioprinting process
}

# Reinforcement Learning Agent Configuration
AGENT_CONFIG = {
    'learning_rate': 3e-4,  # Learning rate for the PPO algorithm
    'gamma': 0.99,  # Discount factor
    'n_steps': 2048,  # Number of steps to collect for each environment per update
    'ent_coef': 0.01,  # Entropy coefficient for exploration
    'clip_range': 0.2,  # Clipping parameter for PPO
    'n_epochs': 10,  # Number of epochs when optimizing the surrogate loss
    'gae_lambda': 0.95,  # Factor for trade-off of bias vs variance for Generalized Advantage Estimator
    'max_grad_norm': 0.5,  # Maximum norm for the gradient clipping
    'vf_coef': 0.5,  # Value function coefficient for the loss calculation
    'batch_size': 64,  # Minibatch size
    'verbose': 1  # Verbosity level: 0 for no output, 1 for info messages, 2 for debug messages
}

# Training Pipeline Configuration
TRAINING_CONFIG = {
    'num_epochs': 100,  # Number of training epochs
    'num_steps_per_epoch': 1000,  # Number of steps to collect for each epoch
    'num_eval_episodes': 5,  # Number of episodes to evaluate the agent
    'save_path': 'trained_models',  # Path to save the trained models
    'log_dir': 'training_logs'  # Directory to save the training logs
}

# Evaluation Metrics Configuration
EVALUATION_CONFIG = {
    'num_episodes': 10,  # Number of episodes to evaluate the agent
    'jaccard_threshold': 0.5,  # Threshold for binarizing the structures in Jaccard similarity
    'hausdorff_threshold': 0.5,  # Threshold for extracting coordinates in Hausdorff distance
    'smoothing_sigma': 1.0  # Sigma value for Gaussian smoothing in surface smoothness calculation
}

# Data Processing Configuration
DATA_CONFIG = {
    'organ_data_path': 'data/organ_models',  # Path to the organ model data
    'preprocessed_data_path': 'data/preprocessed',  # Path to store preprocessed data
    'voxel_data_path': 'data/voxel_data',  # Path to store voxel data
    'mesh_data_path': 'data/mesh_data',  # Path to store mesh data
    'target_resolution': (64, 64, 64),  # Target resolution for preprocessing
    'nifti_extension': '.nii.gz'  # File extension for NIfTI files
}

# Visualization Configuration
VISUALIZATION_CONFIG = {
    'color_map': ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                  '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'],  # Color map for materials
    'opacity': 0.8,  # Opacity of the visualized structures
    'marker_size': 3,  # Size of the markers in the visualizations
    'animation_fps': 10,  # Frames per second for animations
    'plot_width': 800,  # Width of the plotly figures
    'plot_height': 600  # Height of the plotly figures
}