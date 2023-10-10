import numpy as np

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

class System:
    def __init__(self):
        self.molecules = []

    def add_molecule(self, molecule):
        self.molecules.append(molecule)

    def get_all_atoms(self):
        return [atom for molecule in self.molecules for atom in molecule.atoms]

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

def coulomb_potential(q1, q2, r):
    return (COULOMB_CONSTANT * q1 * q2) / r

def lennard_jones_potential(r, epsilon, sigma):
    return 4 * epsilon * ((sigma / r)**12 - (sigma / r)**6)

def energy_function_system(coords, system):
    energy = 0.0
    all_atoms = system.get_all_atoms()
    num_atoms = len(all_atoms)
    reshaped_coords = np.reshape(coords, (num_atoms, 3))

    for i, atom in enumerate(all_atoms):
        atom.coordinate = reshaped_coords[i]

    for i in range(num_atoms):
        for j in range(i + 1, num_atoms):
            distance = np.linalg.norm(all_atoms[i].coordinate - all_atoms[j].coordinate)
            energy += lennard_jones_potential(distance, EPSILON, SIGMA)
            energy += coulomb_potential(all_atoms[i].charge, all_atoms[j].charge, distance)

    return energy

def crossover(mol1_coords, mol2_coords):
    return (mol1_coords + mol2_coords) / 2

def mutate(mol_coords, mutation_rate=0.1, mutation_scale=0.1):
    mutation = (np.random.rand(*mol_coords.shape) - 0.5) * mutation_scale
    mol_coords += mutation * (np.random.rand() < mutation_rate)
    return mol_coords

def genetic_algorithm(system, generations=1000, population_size=100, mutation_rate=0.1):
    initial_coords = np.array([atom.coordinate for atom in system.get_all_atoms()]).flatten()
    population = [initial_coords + (np.random.rand(*initial_coords.shape) - 0.5) for _ in range(population_size)]
    energies = [energy_function_system(ind, system) for ind in population]
    
    for _ in range(generations):
        selected_indices = np.argsort(energies)[:population_size//2]
        selected_population = [population[i] for i in selected_indices]
        
        offspring = []
        for i in range(0, len(selected_population), 2):
            if i + 1 < len(selected_population):
                offspring.append(crossover(selected_population[i], selected_population[i+1]))
            else:
                offspring.append(crossover(selected_population[i], selected_population[0]))

        offspring = [mutate(ind, mutation_rate) for ind in offspring]
        population = selected_population + offspring
        energies = [energy_function_system(ind, system) for ind in population]
        
        print("Current Best Energy:", min(energies))
    
    best_index = np.argmin(energies)
    best_coords_reshaped = np.reshape(population[best_index], (len(system.get_all_atoms()), 3))
    
    for i, atom in enumerate(system.get_all_atoms()):
        atom.coordinate = best_coords_reshaped[i]
    
    return system

def export_pdb(file_path, system, remark="OPTIMIZED COORDINATES"):
    """
    Export the atomic coordinates in PDB format.
    
    :param file_path: The path to the file where the coordinates will be saved.
    :param system: The system containing the atomic coordinates.
    :param remark: An optional remark line.
    """
    with open(file_path, "w") as file:
        file.write("REMARK     {}\n".format(remark))
        
        atom_index = 1  # PDB files typically start numbering atoms at 1
        for molecule in system.molecules:
            for atom in molecule.atoms:
                # PDB Format: http://www.wwpdb.org/documentation/file-format-content/format33/sect9.html
                file.write(
                    "{:<6}{:>5} {:<4}{:<1}{:<3} {:<1}{:>4}{:<1}   {:>8.3f}{:>8.3f}{:>8.3f}{:>6.2f}{:>6.2f}          {:>2}{:>2}\n".format(
                        "ATOM", atom_index, atom.symbol, "", "RES", "A", 1, "",
                        atom.coordinate[0], atom.coordinate[1], atom.coordinate[2],
                        1.00,  # Occupancy - placeholder value
                        0.00,  # Temperature factor - placeholder value
                        atom.symbol, ""  # Element symbol and charge
                    )
                )
                atom_index += 1


if __name__ == "__main__":
    water = parse_pdb("water.pdb")  # Replace with your PDB file path
    octane = parse_pdb("octane.pdb")  # Replace with your PDB file path
    
    system = System()
    system.add_molecule(water)
    system.add_molecule(octane)

    optimized_system = genetic_algorithm(system)
    
    # Exporting the optimized coordinates
    export_pdb("optimized_coordinates.pdb", optimized_system)
