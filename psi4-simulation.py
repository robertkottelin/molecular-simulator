import psi4
import numpy as np
from typing import Union, Tuple

# Constants
INITIAL_ITERATIONS = 10
INCREMENT_ITERATIONS = 10
MAX_LOOPS = 20

# Set Global Options for Psi4
psi4.core.print_global_options()

psi4.set_options({
    "GEOM_MAXITER": 100,
    "GUESS": "sad",
    "REFERENCE": "rhf",
    "SCF_TYPE": "direct",
    "INTS_TOLERANCE": 1.0e-8
    })


psi4.set_memory("4GB")
psi4.set_output_file("p4_output.txt", append=False, loglevel=20, print_header=True)
psi4.core.set_num_threads(4)

# Define Molecule
molecule = psi4.geometry(
    """
    0 1
    C               -1.837430602809657    -0.177564442075653     1.467983295051322
    O               -1.790272937984613     0.572210224938438     2.326698462632702
    O               -1.931574832968839    -0.942126798368836     0.625917214068646
    --
    0 1
    N                0.912451885545963     0.792042375029606     0.630699427712218
    C                1.346960214547239     0.827200345844812    -0.758860653490601
    C                1.290324405556613    -0.562034541454072    -1.361197290315369
    O                1.735995737176714    -0.469381315224569    -2.687160148267861
    H                0.832863879183169     1.726369729683071     0.994064232425070
    H                1.607893489136067     0.338081439891139     1.200590132661419
    H                2.358914735021171     1.226959536627974    -0.896205507694667
    H                0.669961130307541     1.473700556503163    -1.320113013035866
    H                0.266288722985364    -0.939951159225802    -1.302567228119095
    H                1.928121739052621    -1.241756033506110    -0.780738817694685
    H                1.651218319857755    -1.311106113036174    -3.107019301039250
    units angstrom
    no_reorient
    symmetry c1  
    """
)


def psi4_optimize(molecule: psi4.core.Molecule,
                  initial_iterations: int = INITIAL_ITERATIONS,
                  increment_iteration: int = INCREMENT_ITERATIONS,
                  max_loop_count: int = MAX_LOOPS,
                  method: str = "hf/cc-pvdz"
                  ) -> Tuple[float, psi4.core.Wavefunction, dict]:
    """
    Function to run Psi4 optimization. If it doesn't converge, it restarts up to max_loop_count times.
    Each restart increases the number of iterations by increment_iteration.
    """

    psi4.core.be_quiet()
    
    loop_count = 0
    molecule_traj = []
    psi4.set_options({"MAXITER": initial_iterations, "GEOM_MAXITER": initial_iterations})
    
    while True:
        try:
            print(f"\nLoop Count: {loop_count}\n")
            opt_energy, wfn, traj = psi4.optimize(method, return_wfn=True, return_history=True)
            break
            
        except psi4.driver.ConvergenceError as cerr:
            print(f"WARNING - Geometry unconverged will try restarting. Error: {str(cerr)}")
            
            bohr_coor = molecule.geometry()
            bohr_coor.scale(psi4.constants.bohr2angstroms)
            molecule_traj.append(bohr_coor.to_array())
            
            loop_count += 1
            if loop_count >= max_loop_count:
                print("WARNING - Unconverged in maximum number of loops")
                raise cerr

            psi4.set_options({
                "GUESS": "read",
                "MAXITER": initial_iterations + loop_count * increment_iteration,
                "GEOM_MAXITER": initial_iterations + loop_count * increment_iteration
            })

    traj = {"coordinates": tuple(np.array(molecule_traj))}
    return opt_energy, wfn, traj

def save_geometry_to_pdb(molecule: psi4.core.Molecule, filename: str):
    """
    Save the molecule's geometry to a PDB file.
    """
    xyz_data = psi4.core.Molecule.save_string_xyz_file(molecule)
    
    with open(filename, "w") as pdb_file:
        for i, line in enumerate(xyz_data.split("\n")[2:]):
            if line.strip():
                atom, x, y, z = line.split()
                pdb_file.write(f"HETATM{int(i)+1:5d}  {atom:3s} LIG A   1    {float(x):8.3f}{float(y):8.3f}{float(z):8.3f}  1.00  0.00           {atom:2s}\n")

energy, wfn, traj = psi4_optimize(molecule)
print(f"Optimized energy: {energy:.6f}")

print("Calculating frequency...")
frequencyenergy, wfn = psi4.frequency("hf/cc-pvdz", molecule=molecule, return_wfn=True)
print(f"Optimized final frequencyenergy: {frequencyenergy:.6f}")

save_geometry_to_pdb(molecule, "optimized_geometry.pdb")
print("Optimized geometry written to optimized_geometry.pdb")

psi4.core.clean()
