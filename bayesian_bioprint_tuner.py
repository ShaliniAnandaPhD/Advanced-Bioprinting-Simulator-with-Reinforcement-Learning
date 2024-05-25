# BioprintOptimizer: Bayesian optimization of bioprinting process parameters with limited experimental data
# Based on the paper "Bayesian optimization of bioprinting process parameters with limited experimental data" (2023)

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from scipy.optimize import minimize
from scipy.stats import norm

### Bayesian Optimization Framework ###

# Define the objective function
def objective_function(params):
    # Evaluate the bioprinting process at the given parameters
    # Return the quality metric (e.g., cell viability, print accuracy)
    # This function will be replaced with the actual bioprinting process evaluation
    quality_metric = np.random.random()  # Placeholder for demonstration
    return quality_metric

# Define the Gaussian process regressor
def gaussian_process_regressor(X_train, y_train, noise=1e-5):
    kernel = RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2)) + WhiteKernel(noise_level=noise)
    gpr = GaussianProcessRegressor(kernel=kernel, alpha=noise, n_restarts_optimizer=10)
    gpr.fit(X_train, y_train)
    return gpr

# Define the acquisition function (Expected Improvement)
def acquisition_function(X, gpr, best_y, xi=0.01):
    mu, sigma = gpr.predict(X, return_std=True)
    Z = (mu - best_y - xi) / sigma
    ei = (mu - best_y - xi) * norm.cdf(Z) + sigma * norm.pdf(Z)
    return ei

# Define the optimization function
def bayesian_optimization(bounds, n_iterations, n_init_samples):
    # Generate initial random samples
    X_train = np.random.uniform(bounds[:, 0], bounds[:, 1], size=(n_init_samples, bounds.shape[0]))
    y_train = np.array([objective_function(x) for x in X_train])
    
    # Initialize the best parameters and quality metric
    best_params = None
    best_quality = -np.inf
    
    for i in range(n_iterations):
        # Fit the Gaussian process regressor
        gpr = gaussian_process_regressor(X_train, y_train)
        
        # Find the next parameters to evaluate
        next_params = find_next_params(gpr, bounds, n_restarts=25)
        
        # Evaluate the objective function at the next parameters
        next_quality = objective_function(next_params)
        
        # Update the best parameters and quality metric
        if next_quality > best_quality:
            best_params = next_params
            best_quality = next_quality
        
        # Add the new sample to the training data
        X_train = np.vstack((X_train, next_params))
        y_train = np.append(y_train, next_quality)
        
        # Print the iteration summary
        print(f"Iteration {i+1}: Best Quality = {best_quality:.3f}")
    
    return best_params, best_quality

# Define the function to find the next parameters to evaluate
def find_next_params(gpr, bounds, n_restarts=25):
    # Optimize the acquisition function to find the next parameters
    best_params = None
    best_acquisition = -np.inf
    
    for _ in range(n_restarts):
        # Generate random starting points for optimization
        start_params = np.random.uniform(bounds[:, 0], bounds[:, 1], size=(1, bounds.shape[0]))
        
        # Minimize the negative acquisition function
        res = minimize(lambda x: -acquisition_function(x.reshape(1, -1), gpr, np.max(gpr.y_train_)),
                       start_params, bounds=bounds, method='L-BFGS-B')
        
        # Update the best parameters and acquisition function value
        if -res.fun > best_acquisition:
            best_params = res.x
            best_acquisition = -res.fun
    
    return best_params

### Bioprinting Process Optimization ###

# Define the parameter bounds for bioprinting process
bounds = np.array([
    [180, 220],  # Temperature (°C)
    [10, 50],    # Pressure (kPa)
    [0.1, 1.0],  # Print speed (mm/s)
    [0.1, 0.5]   # Layer height (mm)
])

# Define the number of iterations and initial samples
n_iterations = 20
n_init_samples = 5

# Run the Bayesian optimization
best_params, best_quality = bayesian_optimization(bounds, n_iterations, n_init_samples)

# Print the best parameters and quality metric
print("\nBest Parameters:")
print(f"Temperature: {best_params[0]:.2f} °C")
print(f"Pressure: {best_params[1]:.2f} kPa")
print(f"Print Speed: {best_params[2]:.2f} mm/s")
print(f"Layer Height: {best_params[3]:.2f} mm")
print(f"\nBest Quality Metric: {best_quality:.3f}")
