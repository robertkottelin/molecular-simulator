def strong_force(p1, p2):
    """
    Calculate the strong force between two protons/neutrons.
    :param p1, p2: the two protons/neutrons
    :return: the strong force
    """
    # Your implementation here
    pass

def weak_force(p1, p2):
    """
    Calculate the weak force between two particles.
    :param p1, p2: the two particles
    :return: the weak force
    """
    # Your implementation here
    pass

# in the energy_function:
for i in range(num_atoms):
    for j in range(i + 1, num_atoms):
        #...
        energy += strong_force(molecule.atoms[i], molecule.atoms[j])
        energy += weak_force(molecule.atoms[i], molecule.atoms[j])
