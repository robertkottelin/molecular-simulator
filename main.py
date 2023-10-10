import numpy as np

def calculate_distance(atom1, atom2):
    return np.linalg.norm(atom1 - atom2)

def lennard_jones_potential(r, epsilon, sigma):
    return 4 * epsilon * ((sigma/r)**12 - (sigma/r)**6)

def total_energy(atoms_positions, epsilon, sigma):
    energy = 0
    num_atoms = len(atoms_positions)
    for i in range(num_atoms):
        for j in range(i+1, num_atoms):
            r = calculate_distance(atoms_positions[i], atoms_positions[j])
            energy += lennard_jones_potential(r, epsilon, sigma)
    return energy

def initialize_population(pop_size, num_atoms):
    # Randomly initialize atom positions in 3D space
    return np.random.rand(pop_size, num_atoms, 3) 

def evaluate_fitness(population, epsilon, sigma):
    # Calculate the energy for each configuration
    return [total_energy(config, epsilon, sigma) for config in population]

def select_parents(fitness, num_parents):
    # Select configurations with low energies
    parents_idx = np.argsort(fitness)[:num_parents]
    return parents_idx

def crossover(parent1, parent2):
    # Simple crossover: take half atoms from each parent
    num_atoms = len(parent1)
    crossover_point = num_atoms // 2
    child = np.vstack((parent1[:crossover_point], parent2[crossover_point:]))
    return child

def mutate(child, mutation_rate):
    # Slightly change atom positions
    mutation_mask = np.random.rand(*child.shape) < mutation_rate
    mutation_change = np.random.uniform(-0.05, 0.05, size=child.shape) 
    child[mutation_mask] += mutation_change[mutation_mask]
    return child

def genetic_algorithm(pop_size=50, num_generations=100, epsilon=0.1, sigma=1.0):
    # Parameters
    mutation_rate = 0.2  # Chance per atom coordinate to be mutated
    num_parents = pop_size // 2
    
    # Initialize
    population = initialize_population(pop_size, num_atoms=10)
    fitness = evaluate_fitness(population, epsilon, sigma)

    for gen in range(num_generations):
        # Select Parents
        parents_idx = select_parents(fitness, num_parents)
        
        new_population = []
        for i in range(0, num_parents, 2):
            # Crossover
            parent1, parent2 = population[parents_idx[i]], population[parents_idx[i+1]]
            child = crossover(parent1, parent2)
            
            # Mutation
            child = mutate(child, mutation_rate)
            
            new_population.append(child)
        
        # Update Population
        population[:num_parents] = new_population
        fitness = evaluate_fitness(population, epsilon, sigma)
        
        # Logging
        print(f"Generation {gen}, Min Energy: {min(fitness)}")
    
    best_config_idx = np.argmin(fitness)
    return population[best_config_idx]

best_configuration = genetic_algorithm()
