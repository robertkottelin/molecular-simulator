import numpy as np

# Parameters
EPSILON = {'O': 0.1, 'H': 0.05, 'C': 0.07}  # Add epsilon for C
SIGMA = {'O': 3.4, 'H': 2.9, 'C': 3.5}  # Add sigma for C
COULOMB_CONSTANT = 332.06371  # [kJ mol^-1 e^-2 / Angstrom] 

# Example parameters (you should look up actual parameters in a force field or literature)
BOND_LENGTHS = {('O', 'H'): 0.96, ('C', 'H'): 1.09, ('C', 'O'): 1.43}  # Add bond lengths for C-H and C-O
BOND_STRENGTHS = {('O', 'H'): 450, ('C', 'H'): 410, ('C', 'O'): 360}  # Add bond strengths for C-H and C-O

ANGLE_EQUILIBRIUM = {('H', 'O', 'H'): 104.5, ('H', 'C', 'H'): 109.5}  # Add angle for H-C-H
ANGLE_STRENGTHS = {('H', 'O', 'H'): 55, ('H', 'C', 'H'): 60}  # Add strength for H-C-H

BOND_PARAMS = {('O', 'H'): (450, 0.96), ('C', 'H'): (410, 1.09), ('C', 'O'): (360, 1.43)}  # Add bond parameters for C-H and C-O
ANGLE_PARAMS = {('H', 'O', 'H'): (55, 104.5), ('H', 'C', 'H'): (60, 109.5)}  # Add angle parameters for H-C-H
DIHEDRAL_PARAMS = {('H', 'O', 'H', 'O'): (1, 0, 1), ('H', 'C', 'O', 'H'): (1, 0, 2)}  # Add dihedral parameters for H-C-O-H

class Atom:
    def __init__(self, name, symbol, x, y, z, charge=0):
        self.name = name
        self.symbol = symbol
        self.coordinate = np.array([x, y, z])
        self.charge = charge

class Molecule:
    def __init__(self, name=""):
        self.atoms = []
        self.name = name

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

def bond_stretching(atom1, atom2, params):
    r = np.linalg.norm(atom1.coordinate - atom2.coordinate)
    k, r_0 = params
    return k * (r - r_0)**2

def angle_bending(atom1, atom2, atom3, params):
    vector1 = atom1.coordinate - atom2.coordinate
    vector2 = atom3.coordinate - atom2.coordinate
    cosine_angle = np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))  # in radians
    k, theta_0 = params
    theta_0 = np.deg2rad(theta_0)  # converting to radians
    return k * (angle - theta_0)**2

def torsional(atom1, atom2, atom3, atom4, params):
    # Compute vectors along the bonds
    b1 = atom2.coordinate - atom1.coordinate
    b2 = atom3.coordinate - atom2.coordinate
    b3 = atom4.coordinate - atom3.coordinate
    
    # Compute normal vectors
    n1 = np.cross(b1, b2)
    n2 = np.cross(b2, b3)
    
    # Compute torsion angle
    cosine_phi = np.dot(n1, n2) / (np.linalg.norm(n1) * np.linalg.norm(n2))
    phi = np.arccos(np.clip(cosine_phi, -1.0, 1.0))  # in radians
    
    V_n, phi_0, n = params
    phi_0 = np.deg2rad(phi_0)  # converting to radians
    return V_n * (1 + np.cos(n*phi - phi_0))

def energy_function_system(coords, system, bond_params, angle_params, dihedral_params):
    energy = 0.0
    all_atoms = system.get_all_atoms()
    num_atoms = len(all_atoms)
    reshaped_coords = np.reshape(coords, (num_atoms, 3))

    for i, atom in enumerate(all_atoms):
        atom.coordinate = reshaped_coords[i]

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
        # Bonded interactions
    for i in range(num_atoms):
        for j in range(i + 1, num_atoms):
            # Bond stretching
            if (all_atoms[i].symbol, all_atoms[j].symbol) in bond_params:
                params = bond_params[(all_atoms[i].symbol, all_atoms[j].symbol)]
                energy += bond_stretching(all_atoms[i], all_atoms[j], params)
            
            # Angle bending
            for k in range(j + 1, num_atoms):
                if (all_atoms[i].symbol, all_atoms[j].symbol, all_atoms[k].symbol) in angle_params:
                    params = angle_params[(all_atoms[i].symbol, all_atoms[j].symbol, all_atoms[k].symbol)]
                    energy += angle_bending(all_atoms[i], all_atoms[j], all_atoms[k], params)
                
                # Torsional (dihedral) angles
                for l in range(k + 1, num_atoms):
                    if (all_atoms[i].symbol, all_atoms[j].symbol, all_atoms[k].symbol, all_atoms[l].symbol) in dihedral_params:
                        params = dihedral_params[(all_atoms[i].symbol, all_atoms[j].symbol, all_atoms[k].symbol, all_atoms[l].symbol)]
                        energy += torsional(all_atoms[i], all_atoms[j], all_atoms[k], all_atoms[l], params)

    return energy

def crossover(mol1_coords, mol2_coords):
    return (mol1_coords + mol2_coords) / 2

def mutate(mol_coords, mutation_rate=0.1, mutation_scale=0.1):
    mutation = (np.random.rand(*mol_coords.shape) - 0.5) * mutation_scale
    mol_coords += mutation * (np.random.rand() < mutation_rate)
    return mol_coords

def genetic_algorithm(system, generations=10000, population_size=100, mutation_rate=0.01):
    initial_coords = np.array([atom.coordinate for atom in system.get_all_atoms()]).flatten()
    population = [initial_coords + (np.random.rand(*initial_coords.shape) - 0.1) for _ in range(population_size)]
    energies = [energy_function_system(ind, system, BOND_PARAMS, ANGLE_PARAMS, DIHEDRAL_PARAMS) for ind in population]
    
    for gen in range(generations):
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
        energies = [energy_function_system(ind, system, BOND_PARAMS, ANGLE_PARAMS, DIHEDRAL_PARAMS) for ind in population]
        
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
    system.add_molecule(ethanol)

    optimized_system = genetic_algorithm(system)
    
    # Exporting the optimized coordinates
    export_pdb("optimized_coordinates.pdb", optimized_system)
