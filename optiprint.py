# OptiBioPrint: Multi-objective optimization of multi-material bioprinting using machine learning
# Based on the paper "Multi-objective optimization of multi-material bioprinting using machine learning" (2022)

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

### Data Preparation ###

# Simulate bioprinting experiment data
def simulate_data(n_samples, n_materials, n_properties):
    # Simulate composition ratios (0-100%)
    compositions = np.random.rand(n_samples, n_materials) * 100
    
    # Normalize composition ratios to sum to 100%
    compositions = compositions / compositions.sum(axis=1, keepdims=True) * 100
    
    # Simulate tissue properties
    properties = np.random.rand(n_samples, n_properties)
    
    # Combine into a DataFrame
    data = pd.DataFrame(np.concatenate([compositions, properties], axis=1),
                        columns=[f'Material_{i}' for i in range(n_materials)] + 
                                [f'Property_{i}' for i in range(n_properties)])
    return data

# Generate simulated data
n_samples = 10000
n_materials = 5
n_properties = 3
data = simulate_data(n_samples, n_materials, n_properties)

# Split into input features X (material ratios) and output targets y (tissue properties)
X = data.iloc[:, :n_materials]
y = data.iloc[:, n_materials:]

# Normalize input to 0-1 range
X = (X - X.min()) / (X.max() - X.min())

# Split into training and test sets 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

### Predictive Model ### 

# Define neural network architecture
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(y_train.shape[1])
])

# Compile model with mean squared error loss and Adam optimizer
model.compile(optimizer='adam', loss='mse')

# Train model for 100 epochs with 20% validation split, in batches of 32
history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2)

# Evaluate model performance on held-out test set
test_loss = model.evaluate(X_test, y_test)
print(f'Test loss: {test_loss:.4f}')

### Genetic Algorithm for Optimization ###

# Define fitness function using trained neural network
def fitness_func(X):
    return model.predict(X)
    
# Define function to identify Pareto front
def identify_pareto(population, fitnesses):
    # Find non-dominated solutions
    pareto_front = np.ones(len(population), dtype=bool)
    for i, fitness_i in enumerate(fitnesses):
        for fitness_j in fitnesses[i+1:]:
            if np.all(fitness_j >= fitness_i) and np.any(fitness_j > fitness_i):
                pareto_front[i] = 0
                break
    return population[pareto_front]

# Define function to select parents from Pareto front
def select_parents(pareto_front, n_parents=10):
    # Select parents randomly from Pareto front
    return pareto_front[np.random.choice(len(pareto_front), n_parents, replace=False)]

# Define function for crossover operation
def crossover(parents, n_offspring=20):
    offspring = np.empty((n_offspring, parents.shape[1]))
    for i in range(n_offspring):
        parent1, parent2 = parents[np.random.choice(len(parents), 2, replace=False)]
        cross_point = np.random.randint(1, parents.shape[1]-1)
        offspring[i, :cross_point] = parent1[:cross_point]
        offspring[i, cross_point:] = parent2[cross_point:]
    return offspring

# Define function for mutation operation  
def mutate(offspring, mutation_rate=0.01, mutation_scale=0.1):
    mutations = np.random.normal(loc=0, scale=mutation_scale, 
                                 size=offspring.shape)
    mutations[np.random.rand(*offspring.shape) > mutation_rate] = 0
    offspring += mutations
    offspring = np.clip(offspring, 0, 1) 
    return offspring
    
# Define function to replace weakest solutions with offspring
def replace_weakest(population, offspring):
    fitnesses = fitness_func(population)
    weakest_idx = np.argsort(np.sum(fitnesses, axis=1))[:len(offspring)]
    population[weakest_idx] = offspring
    return population

# Set GA hyperparameters
n_pop = 100
n_generations = 50
n_parents = 10
n_offspring = 20
mutation_rate = 0.01
mutation_scale = 0.1

# Generate initial population of random solutions 
population = np.random.rand(n_pop, X_train.shape[1])

# Run GA optimization
for i in range(n_generations):
    
    # Evaluate fitnesses using neural network prediction
    fitnesses = fitness_func(population) 
    
    # Find Pareto front of non-dominated solutions
    pareto_front = identify_pareto(population, fitnesses)
    
    # Select parents from Pareto front
    parents = select_parents(pareto_front, n_parents)
    
    # Generate offspring via crossover + mutation
    offspring = crossover(parents, n_offspring)
    offspring = mutate(offspring, mutation_rate, mutation_scale)
    
    # Replace weakest solutions with offspring
    population = replace_weakest(population, offspring)
    
    # Print progress
    print(f"Generation {i+1}/{n_generations} - Pareto front size: {len(pareto_front)}")

# Evaluate final Pareto front    
final_pareto = identify_pareto(population, fitness_func(population))

# Print best solutions
print("\nOptimal material compositions found:")
print(pd.DataFrame(final_pareto, columns=[f'Material_{i}' for i in range(n_materials)]))
