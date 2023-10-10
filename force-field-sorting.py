def calculate_van_der_Waals_energy(distance, epsilon, sigma):
    """
    Simplistic Lennard-Jones Potential for van der Waals energy.
    E = 4 * epsilon * [(sigma/distance)^12 - (sigma/distance)^6]
    """
    return 4 * epsilon * ((sigma/distance)**12 - (sigma/distance)**6)

def calculate_electrostatic_energy(distance, charge1, charge2):
    """
    Coulomb's Law for electrostatic energy.
    E = (charge1 * charge2) / (4 * pi * epsilon * distance)
    """
    epsilon_0 = 8.85e-12  # Vacuum permittivity in C^2/(J*m)
    return (charge1 * charge2) / (4 * 3.141592 * epsilon_0 * distance)
