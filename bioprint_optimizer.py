# BioPrintOptimizer: Genetic algorithm-based optimization of bioprinting process parameters
# Based on the paper "Genetic algorithm-based optimization of bioprinting process parameters" (2022)

import numpy as np
import random

### Problem Definition ###

# Define the bioprinting process parameters and their ranges
parameters = {
    'temperature': (20, 40),  # Temperature range (°C)
    'pressure': (0.1, 1.0),   # Pressure range (bar)
    'speed': (1, 10),         # Print speed range (mm/s)
    'layer_height': (0.1, 0.5),  # Layer height range (mm)
    'infill_density': (0.2, 0.8)  # Infill density range (ratio)
}

# Define the objective functions for the specific tissue type
def objective1(individual):
    # Maximize cell viability
    # Example: Simulate cell viability based on bioprinting parameters
    temperature, pressure, speed, layer_height, infill_density = individual
    viability = temperature * 0.8 - pressure * 0.5 + speed * 0.2 - layer_height * 0.3 + infill_density * 0.6
    return viability

def objective2(individual):
    # Minimize printing time
    # Example: Estimate printing time based on bioprinting parameters
    temperature, pressure, speed, layer_height, infill_density = individual
    printing_time = temperature * 0.1 + pressure * 0.2 - speed * 0.4 + layer_height * 0.3 - infill_density * 0.2
    return -printing_time  # Minimize printing time

### Genetic Algorithm ###

# Define the genetic algorithm parameters
population_size = 100
num_generations = 50
crossover_rate = 0.8
mutation_rate = 0.1

# Define the individual representation
def create_individual():
    individual = []
    for param, range_ in parameters.items():
        value = random.uniform(range_[0], range_[1])
        individual.append(value)
    return individual

# Define the population initialization
def initialize_population():
    population = []
    for _ in range(population_size):
        individual = create_individual()
        population.append(individual)
    return population

# Define the fitness evaluation
def evaluate_fitness(individual):
    fitness1 = objective1(individual)
    fitness2 = objective2(individual)
    return fitness1, fitness2

# Define the selection method (tournament selection)
def selection(population, tournament_size=5):
    selected = []
    for _ in range(len(population)):
        tournament = random.sample(population, tournament_size)
        best = max(tournament, key=lambda x: evaluate_fitness(x)[0] + evaluate_fitness(x)[1])
        selected.append(best)
    return selected

# Define the crossover method (arithmetic crossover)
def crossover(parent1, parent2):
    child = []
    for i in range(len(parent1)):
        if random.random() < crossover_rate:
            child.append(parent1[i] * random.random() + parent2[i] * (1 - random.random()))
        else:
            child.append(parent1[i] if random.random() < 0.5 else parent2[i])
    return child

# Define the mutation method (gaussian mutation)
def mutation(individual):
    mutated = []
    for i in range(len(individual)):
        if random.random() < mutation_rate:
            param, range_ = list(parameters.items())[i]
            std = (range_[1] - range_[0]) * 0.1
            mutated.append(np.clip(individual[i] + random.gauss(0, std), range_[0], range_[1]))
        else:
            mutated.append(individual[i])
    return mutated

# Define the genetic algorithm main loop
def genetic_algorithm():
    population = initialize_population()
    
    for generation in range(num_generations):
        # Evaluate fitness of each individual
        fitnesses = [evaluate_fitness(individual) for individual in population]
        
        # Print the best individual in the current generation
        best_index = np.argmax([f[0] + f[1] for f in fitnesses])
        best_individual = population[best_index]
        best_fitness = fitnesses[best_index]
        print(f"Generation {generation+1}: Best Individual = {best_individual}, Fitness = {best_fitness}")
        
        # Perform selection
        parents = selection(population)
        
        # Create the next generation
        offspring = []
        for i in range(0, len(parents), 2):
            parent1, parent2 = parents[i], parents[i+1]
            child1 = crossover(parent1, parent2)
            child2 = crossover(parent1, parent2)
            offspring.append(mutation(child1))
            offspring.append(mutation(child2))
        
        population = offspring
    
    # Return the best individual found
    best_individual = max(population, key=lambda x: evaluate_fitness(x)[0] + evaluate_fitness(x)[1])
    return best_individual

### Optimization Results ###

# Run the genetic algorithm
best_solution = genetic_algorithm()

# Print the optimized bioprinting parameters
print("\nOptimized Bioprinting Parameters:")
for param, value in zip(parameters.keys(), best_solution):
    print(f"{param}: {value}")

# Evaluate the optimized solution
fitness = evaluate_fitness(best_solution)
print(f"\nOptimized Fitness: Cell Viability = {fitness[0]}, Printing Time = {-fitness[1]}")
