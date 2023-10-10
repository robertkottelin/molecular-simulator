


import numpy as np
from scipy.optimize import minimize

EPSILON = 0.1  # Energy minimum [kJ/mol]
SIGMA = 3.4  # Distance at which the potential energy is zero [Angstrom]
COULOMB_CONSTANT = 332.06371  # [kJ mol^-1 e^-2 / Angstrom], approximately

class Atom:
    def __init__(self, symbol, x, y, z, charge=0):
        self.symbol = symbol
        self.coordinate = np.array([x, y, z])
        self.charge = charge

class Molecule:
    def __init__(self):
        self.atoms = []

    def add_atom(self, atom):
        self.atoms.append(atom)

def parse_pdb(file_path):
    molecule = Molecule()
    with open(file_path, 'r') as file:
        for line in file:
            if line.startswith("ATOM") or line.startswith("HETATM"):
                x = float(line[30:38].strip())
                y = float(line[38:46].strip())
                z = float(line[46:54].strip())
                symbol = line[76:78].strip()
                molecule.add_atom(Atom(symbol, x, y, z))
    return molecule

def realistic_energy_function(coords, molecule):
    """
    Calculate the potential energy of a molecule based on Lennard-Jones and Coulomb potential.
    :param coords: 1D array of atomic coordinates [x1, y1, z1, x2, y2, z2, ...]
    :param molecule: Molecule object
    :return: Total potential energy [kJ/mol]
    """
    energy = 0.0
    num_atoms = len(molecule.atoms)
    reshaped_coords = np.reshape(coords, (num_atoms, 3))

    for i in range(num_atoms):
        molecule.atoms[i].coordinate = reshaped_coords[i]

    for i in range(num_atoms):
        for j in range(i + 1, num_atoms):
            distance = np.linalg.norm(molecule.atoms[i].coordinate - molecule.atoms[j].coordinate)

            # Lennard-Jones potential
            energy += lennard_jones_potential(distance, EPSILON, SIGMA)

            # Coulomb's potential
            energy += coulomb_potential(molecule.atoms[i].charge, molecule.atoms[j].charge, distance)

    return energy

def coulomb_potential(q1, q2, r):
    """
    Calculate Coulomb's potential between two charged atoms.
    :param q1, q2: partial charges of the atoms [e]
    :param r: distance between atoms [Angstrom]
    :return: Coulomb's potential [kJ/mol]
    """
    return (COULOMB_CONSTANT * q1 * q2) / r

def lennard_jones_potential(r, epsilon, sigma):
    """
    Calculate Lennard-Jones potential between two atoms at distance r.
    :param r: distance between two atoms [Angstrom]
    :param epsilon: well depth [kJ/mol]
    :param sigma: finite distance at which the inter-particle potential is zero [Angstrom]
    :return: Lennard-Jones potential [kJ/mol]
    """
    return 4 * epsilon * ((sigma / r)**12 - (sigma / r)**6)

def crossover(mol1_coords, mol2_coords):
    return (mol1_coords + mol2_coords) / 2

def mutate(mol_coords, mutation_rate=0.1, mutation_scale=0.1):
    mutation = (np.random.rand(*mol_coords.shape) - 0.5) * mutation_scale
    mol_coords += mutation * (np.random.rand() < mutation_rate)
    return mol_coords

# Example Genetic Algorithm
def genetic_algorithm(molecule, generations=10000, population_size=1000, mutation_rate=0.1):
    population = [np.array([atom.coordinate for atom in molecule.atoms]).flatten() for _ in range(population_size)]
    energies = [realistic_energy_function(ind, molecule) for ind in population]
    
    for _ in range(generations):
        # Selection: top half of the molecules are selected
        selected_indices = np.argsort(energies)[:population_size//2]
        selected_population = [population[i] for i in selected_indices]
        
        # Crossover: generating offspring by mixing coordinates
        offspring = []
        for i in range(0, len(selected_population), 2):
            if i+1 < len(selected_population):
                offspring.append(crossover(selected_population[i], selected_population[i+1]))
            else:
                offspring.append(crossover(selected_population[i], selected_population[0]))
        
        # Mutation: introducing minor random alterations
        offspring = [mutate(ind, mutation_rate) for ind in offspring]
        
        # Replacement: replacing the old population with offspring
        population = selected_population + offspring
        
        # Recalculate energies
        energies = [realistic_energy_function(ind, molecule) for ind in population]
        
        # Logging the current best energy value
        print("Current Best Energy:", min(energies))
    
    # Returning the best conformation
    best_index = np.argmin(energies)
    best_coords_reshaped = np.reshape(population[best_index], (len(molecule.atoms), 3))
    
    for i, atom in enumerate(molecule.atoms):
        atom.coordinate = best_coords_reshaped[i]
    
    return molecule

# Example usage
if __name__ == "__main__":
    # Replace with your PDB file path
    molecule = parse_pdb("water.pdb")
    
    # Optimize using the genetic algorithm
    optimized_molecule = genetic_algorithm(molecule)
    
    # At this point, `optimized_molecule` contains the optimized atom coordinates
