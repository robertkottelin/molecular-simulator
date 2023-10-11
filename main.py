import numpy as np

# Parameters
EPSILON = {'O': 0.1, 'H': 0.05}  # Energy minimum [kJ/mol] at the equilibrium distance
SIGMA = {'O': 3.4, 'H': 2.9}  # Distance at which the potential energy is zero [Angstrom] (i.e. the van der Waals radius)
COULOMB_CONSTANT = 332.06371  # [kJ mol^-1 e^-2 / Angstrom] 

class Bond:
    def __init__(self, atom1, atom2, length, k):
        self.atom1 = atom1
        self.atom2 = atom2
        self.length = length  # Equilibrium bond length
        self.k = k  # Bond force constant
    
    def calculate_energy(self):
        # Harmonic potential for bonds
        r = np.linalg.norm(self.atom1.coordinate - self.atom2.coordinate)
        return self.k * (r - self.length)**2


class Angle:
    def __init__(self, atom1, atom2, atom3, theta, k):
        self.atom1 = atom1
        self.atom2 = atom2
        self.atom3 = atom3
        self.theta = theta  # Equilibrium angle (in radians)
        self.k = k  # Angle force constant
    
    def calculate_energy(self):
        # Harmonic potential for angles
        a = self.atom2.coordinate - self.atom1.coordinate
        b = self.atom2.coordinate - self.atom3.coordinate
        theta = np.arccos(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
        return self.k * (theta - self.theta)**2

class Dihedral:
    def __init__(self, atom1, atom2, atom3, atom4, phi, k, n):
        self.atom1 = atom1
        self.atom2 = atom2
        self.atom3 = atom3
        self.atom4 = atom4
        self.phi = phi  # Equilibrium dihedral angle (in radians)
        self.k = k  # Dihedral force constant
        self.n = n  # Periodicity
    
    def calculate_energy(self):
        # Periodic potential for dihedrals
        a = self.atom2.coordinate - self.atom1.coordinate
        b = self.atom3.coordinate - self.atom2.coordinate
        c = self.atom4.coordinate - self.atom3.coordinate
        normal1 = np.cross(a, b)
        normal2 = np.cross(b, c)
        x = np.dot(normal1, normal2)
        y = np.dot(np.cross(normal1, b/np.linalg.norm(b)), normal2)
        phi = -np.arctan2(y, x)
        return self.k * (1 + np.cos(self.n * phi - self.phi))

class Atom:
    def __init__(self, name, symbol, x, y, z, charge=0):
        self.name = name
        self.symbol = symbol
        self.coordinate = np.array([x, y, z])
        self.charge = charge

class Molecule:
    def __init__(self, name=""):
        self.atoms = []
        self.bonds = []
        self.angles = []
        self.dihedrals = []
        self.name = name
    
    def add_atom(self, atom):
        self.atoms.append(atom)
    
    def add_bond(self, bond):
        self.bonds.append(bond)
    
    def add_angle(self, angle):
        self.angles.append(angle)
    
    def add_dihedral(self, dihedral):
        self.dihedrals.append(dihedral)


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
                name = line[12:16].strip()
                x = float(line[30:38].strip())
                y = float(line[38:46].strip())
                z = float(line[46:54].strip())
                symbol = line[76:78].strip()
                
                # Extract symbol from name if missing
                if not symbol:
                    symbol = ''.join(filter(str.isalpha, name))
                    if not symbol:
                        raise ValueError(f"Unable to determine atom symbol in line: {line.strip()}")
                
                molecule.add_atom(Atom(name, symbol, x, y, z))
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

    # Bonded Interactions: bonds, angles, dihedrals
    for molecule in system.molecules:
        for bond in molecule.bonds:
            energy += bond.calculate_energy()
        for angle in molecule.angles:
            energy += angle.calculate_energy()
        for dihedral in molecule.dihedrals:
            energy += dihedral.calculate_energy()

    # Non-Bonded Interactions: van der Waals and electrostatic
    for i in range(num_atoms):
        for j in range(i + 1, num_atoms):
            distance = np.linalg.norm(all_atoms[i].coordinate - all_atoms[j].coordinate)
            try:
                epsilon_i = EPSILON[all_atoms[i].symbol]
                epsilon_j = EPSILON[all_atoms[j].symbol]
                sigma_i = SIGMA[all_atoms[i].symbol]
                sigma_j = SIGMA[all_atoms[j].symbol]
            except KeyError as e:
                raise KeyError(f"Missing parameters for atom symbol: {str(e)}")
            
            average_epsilon = np.sqrt(epsilon_i * epsilon_j)
            average_sigma = np.sqrt(sigma_i * sigma_j)
            energy += lennard_jones_potential(distance, average_epsilon, average_sigma)
            energy += coulomb_potential(all_atoms[i].charge, all_atoms[j].charge, distance)

    return energy
def crossover(mol1_coords, mol2_coords):
    """
    Perform one-point crossover between two parent molecules.
    """
    cut_point = np.random.randint(low=1, high=mol1_coords.size - 1)
    child1 = np.concatenate((mol1_coords[:cut_point], mol2_coords[cut_point:]))
    child2 = np.concatenate((mol2_coords[:cut_point], mol1_coords[cut_point:]))
    return child1, child2

def mutate(mol_coords, mutation_rate=0.1, mutation_scale=0.1):
    """
    Apply Gaussian mutation to the molecular coordinates.
    """
    mutation = np.random.randn(*mol_coords.shape) * mutation_scale
    mask = np.random.rand(*mol_coords.shape) < mutation_rate
    mol_coords += mask * mutation
    return mol_coords

def genetic_algorithm(system, generations=1000, population_size=100, mutation_rate=0.2):
    initial_coords = np.array([atom.coordinate for atom in system.get_all_atoms()]).flatten()
    population = [initial_coords + (np.random.rand(*initial_coords.shape) - 0.2) for _ in range(population_size)]
    energies = [energy_function_system(ind, system) for ind in population]
    
    for gen in range(generations):
        selected_indices = np.argsort(energies)[:population_size//2]
        selected_population = [population[i] for i in selected_indices]
        
        offspring = []
        for i in range(0, len(selected_population), 2):
            if i + 1 < len(selected_population):
                child1, child2 = crossover(selected_population[i], selected_population[i+1])
                offspring.append(child1)
                offspring.append(child2)
            else:
                child1, child2 = crossover(selected_population[i], selected_population[0])
                offspring.append(child1)
                offspring.append(child2)

        offspring = [mutate(ind, mutation_rate) for ind in offspring]
        population = selected_population + offspring
        energies = [energy_function_system(ind, system) for ind in population]
        
        if gen % 100 == 0:  # Print every 100 generations for brevity
            print(f"Generation: {gen + 1}, Current Best Energy: {min(energies)}")
    
    best_index = np.argmin(energies)
    best_coords_reshaped = np.reshape(population[best_index], (len(system.get_all_atoms()), 3))
    
    for i, atom in enumerate(system.get_all_atoms()):
        atom.coordinate = best_coords_reshaped[i]
    
    return system

def export_pdb(file_path, system, remark="OPTIMIZED COORDINATES"):
    with open(file_path, "w") as file:
        file.write("REMARK     {}\n".format(remark))
        
        atom_index = 1
        for molecule_index, molecule in enumerate(system.molecules, start=1):
            # Use the name attribute of the molecule in the COMPND line
            file.write("COMPND      {}\n".format(molecule.name)) 
            
            for atom in molecule.atoms:
                # Adjusted format string to skip the "RES A" columns and directly output the molecule index
                file.write(
                    "{:<6}{:>5}  {:<4}       {:>4}   {:>8.3f}{:>8.3f}{:>8.3f}{:>6.2f}{:>6.2f}          {:>2}{:>2}\n".format(
                        "HETATM", atom_index, atom.name, molecule_index,
                        atom.coordinate[0], atom.coordinate[1], atom.coordinate[2],
                        1.00,  # Occupancy - placeholder value
                        0.00,  # Temperature factor - placeholder value
                        atom.symbol, ""  # Element symbol and charge
                    )
                )
                atom_index += 1
            
            file.write("TER       {:>5}              {:>4}\n".format(atom_index, molecule_index))
        file.write("END\n")

if __name__ == "__main__":
    water = parse_pdb("water.pdb")  # Replace with your PDB file path
    water.name = "Water"
    ethanol = parse_pdb("ethanol.pdb")  # Replace with your PDB file path
    ethanol.name = "Ethanol"
    
    system = System()
    system.add_molecule(water)
    # system.add_molecule(ethanol)

    optimized_system = genetic_algorithm(system)
    
    # Exporting the optimized coordinates
    export_pdb("optimized_coordinates.pdb", optimized_system)