import psi4
import numpy as np
from typing import Union, Tuple

# Print the global options avaliable in Psi4
print(psi4.core.print_global_options())

psi4.set_options(
    {
        "GEOM_MAXITER": 100,
        "GUESS": "sad",
        "REFERENCE": "rhf",
        "SCF_TYPE": "direct",
        "INTS_TOLERANCE": 1.0e-8
    }
)

psi4.set_memory("4GB")
psi4.set_output_file("p4_output.txt", 
                     append=False,
                     loglevel=20, 
                     print_header=True
                    )
psi4.core.set_num_threads(4)

# molecule = psi4.geometry(
#     """
#     0 1
#     O        0.00000        0.64670       -0.01863
#     H        0.76026        0.61622       -0.62453
#     H       -0.76026        0.61622       -0.62453
#     --
#     0 1
#     O        0.00000       -0.04191        2.64300
#     H        0.00000        0.08820        1.66859
#     H        0.00000        0.87457        2.95608
#     units angstrom
#     symmetry c1
#     """ 
# )

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

def psi4_optimize(initial_iterations: int = 2,
                  increment_iteration: int = 2,
                  max_loop_count: int = 20,
                  molecule: Union[psi4.core.Molecule, None] = None,
                  method: str = "hf/cc-pvdz"
                 ) -> Tuple[float, psi4.core.Wavefunction, dict]:
    """
    General function to run Psi4 optimization and restart a max of max_loop_count times incrementing the number
    of iterations by increment_iteration from initial_iterations each loop
    :param initial_iterations: integer the number of initial iterations
    :param increment_iteration: integer the number of iteration increment by each time it fails to converge
    :param max_loop_count: integer the maximum number of times to increment for convergence incomplete
    :param molecule: psi4.core.Molecule the molecule object or None if given will save last geometry of each unconverged loop
    :method: string define the quantum chem optimization method eg hf/cc-pvdz
    :return: Tuple[float, psi4.core.Wavefunction, list] the energy, the wavefunction object and the history of conformations
    """
    
    unconv = True

    psi4.set_options(
        {
            "MAXITER": initial_iterations,
            "GEOM_MAXITER": initial_iterations
        }
    )
    
    loop_count = 0
    
    if molecule is not None:
        molecule_traj = []

    while unconv is True:
        
        try:
            
            print("\ncount {}\n".format(loop_count))
            opt_energy, wfn, traj = psi4.optimize(method, return_wfn=True, return_history=True)
            print("The optimized energy for this configuration is {:.6f} Hartree".format(opt_energy))
            unconv = False
            
        except psi4.driver.ConvergenceError as cerr:
            
            print("WARNING - Geometry unconverged will try restarting")
            if molecule is not None:
                bohr_coor = molecule.geometry()
                bohr_coor.scale(psi4.constants.bohr2angstroms)
                molecule_traj.append(bohr_coor.to_array())
            unconv_wfn = cerr.wfn
            unconv_wfn.to_file(unconv_wfn.get_scratch_filename(180))
            psi4.set_options(
                {
                    "GUESS": "read",
                    "MAXITER": initial_iterations + increment_iteration,
                    "GEOM_MAXITER": initial_iterations + increment_iteration
                }
            )
            
            loop_count = loop_count + 1

            if max_loop_count <= loop_count:
                print("WARNING - Unconverged in maximum number of loops")
                raise cerr
                
    if molecule is not None:
        traj = {"coordinates": tuple(np.array(molecule_traj))}

    return opt_energy, wfn, traj

# energy = psi4.energy("hf/cc-pvdz")
# print("The energy for this configuration is {:.6f} Hartree".format(energy))
# >>> -76.025611

# opt_energy = psi4.optimize("hf/cc-pvdz")
# print("The optimized energy for this configuration is {:.6f} Hartree".format(opt_energy))
# >>> -76.027054

interaction_energy = psi4.energy("hf/cc-pvdz", bsse_type="cp")
print("Interaction Basis Set Superposition Error (BSSE) energy {:.6f} Hartree".format(interaction_energy))
energy, wfn = psi4.frequency("hf/cc-pvdz", molecule=molecule, return_wfn=True)
# >>> -0.004137

e_sapt0 = psi4.energy("sapt0/aug-cc-pVDZ")
print("Interaction Symmetry Adapted Perturbation Theory (SAPT) energy {:.6f} Hartree".format(e_sapt0))
# >>> -0.0067524


# energy, wfn, traj = psi4_optimize(initial_iterations=10,
#                                   increment_iteration=10,
#                                   max_loop_count=20,
#                                   method="hf/cc-pvdz",
#                                  )

# print("Optimized energy {:.6f}".format(energy))

# oeprops = psi4.core.OEProp(wfn)
# oeprops.add("DIPOLE")
# oeprops.add("QUADRUPOLE")
# oeprops.add("MULLIKEN_CHARGES")
# oeprops.add("MULTIPOLE(4)")
# oeprops.add("ESP_AT_NUCLEI")
# oeprops.add("MO_EXTENTS")
# oeprops.add("LOWDIN_CHARGES")
# oeprops.add("WIBERG_LOWDIN_INDICES")
# oeprops.add("MAYER_INDICES")
# oeprops.add("NO_OCCUPATIONS")
# oeprops.compute()

# properties = wfn.variables()
# print(properties["DIPOLE"])

psi4.core.clean()