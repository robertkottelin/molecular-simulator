import psi4

molecule = psi4.geometry(
    """
    0 1
    C               -1.837430602809657    -0.177564442075653     1.467983295051322
    O               -1.790272937984613     0.572210224938438     2.326698462632702
    O               -1.931574832968839    -0.942126798368836     0.625917214068646
    --
    0 1
    N                0.912451885545963     0.792042375029606     5.630699427712218
    C                1.346960214547239     0.827200345844812     4.241138346221618
    C                1.290324405556613    -0.562034541454072     3.638801709396850
    O                1.735995737176714    -0.469381315224569     2.312838851444358
    H                0.832863879183169     1.726369729683071     5.994064232425070
    H                1.607893489136067     0.338081439891139     6.200590132661419
    H                2.358914735021171     1.226959536627974     4.103794492305333
    H                0.669961130307541     1.473700556503163     3.679886986964133
    H                0.266288722985364    -0.939951159225802     3.697432771880904
    H                1.928121739052621    -1.241756033506110     4.219261182305315
    H                1.651218319857755    -1.311106113036174     1.892980698960749
    units angstrom
    no_reorient
    symmetry c1
    """ 
)

# Set up the output file
psi4.set_output_file("p4_output.txt", False)

# Introducing the solvent using PCM
psi4.set_options({'pcm': True, 'pcm_scf_type': 'total', 'solvent': 'Water'})

previous_sapt_energy = None
threshold = 1.0e-7  # Convergence threshold for SAPT energy
max_iterations = 1  # Maximum number of iterations


for iteration in range(max_iterations):
    
    # 1. Optimize the Geometry
    psi4.core.be_quiet()
    energy = psi4.energy("hf/cc-pvdz", molecule=molecule)
    print("The energy for this configuration is {:.6f} Hartree".format(energy))
    
    energy, wfn = psi4.frequency("hf/cc-pvdz", molecule=molecule, return_wfn=True)
    
    opt_energy = psi4.optimize("hf/cc-pvdz", molecule=molecule) 
    print("The optimized energy for this configuration is {:.6f} Hartree".format(opt_energy))

    psi4.core.reopen_outfile()

else: 
    print("Warning: Maximum number of iterations reached. The calculation did not converge.")

# Extracting XYZ data from Psi4's molecule object
xyz_data = psi4.core.Molecule.save_string_xyz_file(molecule)

# Writing to PDB file
with open("optimized_geometry.pdb", "w") as pdb_file:
    for i, line in enumerate(xyz_data.split("\n")[2:]):
        if line.strip():
            atom, x, y, z = line.split()
            pdb_file.write(f"HETATM{int(i)+1:5d}  {atom:3s} LIG A   1    {float(x):8.3f}{float(y):8.3f}{float(z):8.3f}  1.00  0.00           {atom:2s}\n")

print("Optimized geometry written to optimized_geometry.pdb") 
