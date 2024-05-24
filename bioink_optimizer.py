# BioinkOptimizer: Data-driven optimization of bioink rheology for enhanced printability and cell viability
# Based on the paper "Data-driven optimization of bioink rheology for enhanced printability and cell viability" (2022)

import numpy as np
import pandas as pd
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

### Data Preparation ###

# Load rheology data from CSV file
# Data format: Each row represents a bioink formulation
#   Columns: Bioink components (concentrations), Rheological properties, Printability score, Cell viability score
# Data size: 500 formulations
data = pd.read_csv('bioink_rheology_data.csv')

# Extract input features (bioink components and concentrations)
X = data.iloc[:, :-2]

# Extract target variables (printability and cell viability scores)
y_printability = data.iloc[:, -2]
y_viability = data.iloc[:, -1]

# Split data into training and testing sets
X_train, X_test, y_printability_train, y_printability_test, y_viability_train, y_viability_test = train_test_split(
    X, y_printability, y_viability, test_size=0.2, random_state=42)

### Model Training ###

# Create SVR models for printability and cell viability
printability_model = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=0.1)
viability_model = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=0.1)

# Train the models
printability_model.fit(X_train, y_printability_train)
viability_model.fit(X_train, y_viability_train)

# Evaluate model performance on testing set
printability_score = printability_model.score(X_test, y_printability_test)
viability_score = viability_model.score(X_test, y_viability_test)
print(f"Printability R^2 score: {printability_score:.3f}")
print(f"Cell viability R^2 score: {viability_score:.3f}")

### Optimization ###
# Define the objective function for optimization
def objective_function(bioink_formulation):
    printability = printability_model.predict(bioink_formulation)
    viability = viability_model.predict(bioink_formulation)
    return printability, viability

# Define the optimization function
def optimize_bioink(objective_function, n_iterations=1000, n_samples=100):
    best_formulation = None
    best_printability = -np.inf
    best_viability = -np.inf

    for _ in range(n_iterations):
        # Generate random bioink formulations
        formulations = np.random.rand(n_samples, X.shape[1])
        
        # Evaluate the objective function for each formulation
        printability_scores, viability_scores = objective_function(formulations)
        
        # Find the formulation with the highest printability and viability scores
        max_printability_idx = np.argmax(printability_scores)
        max_viability_idx = np.argmax(viability_scores)
        
        if printability_scores[max_printability_idx] > best_printability:
            best_printability = printability_scores[max_printability_idx]
            best_formulation = formulations[max_printability_idx]
        
        if viability_scores[max_viability_idx] > best_viability:
            best_viability = viability_scores[max_viability_idx]
            best_formulation = formulations[max_viability_idx]
    
    return best_formulation, best_printability, best_viability

# Run the optimization
best_formulation, best_printability, best_viability = optimize_bioink(objective_function)

# Print the optimal bioink formulation and scores
print("Optimal Bioink Formulation:")
for component, concentration in zip(X.columns, best_formulation):
    print(f"{component}: {concentration:.3f}")
print(f"Printability Score: {best_printability:.3f}")
print(f"Cell Viability Score: {best_viability:.3f}")
