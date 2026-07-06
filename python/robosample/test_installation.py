from pathlib import Path


def run_test():
    import robosample

    current_dir = Path(__file__).parent.resolve()
    repo_root = current_dir.parents[1]
    examples_dir = repo_root / "examples"

    # Create simulation
    name = "ala-dipeptide.test.rigid"
    seed = 42
    prmtop = str(examples_dir / "ala-dipeptide.prmtop")
    xyz = str(examples_dir / "ala-dipeptide.rst7")
    write_freq = 1
    equil_steps = 10
    prod_steps = 10
    context = robosample.Context(
        name=name,
        seed=seed,
        prmtop=prmtop,
        inpcrd=xyz,
        write_freq=write_freq,
        testing=False,
    )

    # Add cartesian world (will integrate with OpenMM)
    context.add_cartesian_world().add_sampler(
        timeStep=0.001,
        mdSteps=10,
        boostMDSteps=10,
        acceptRejectMode=robosample.rb.AcceptRejectMode.MetropolisHastings,
    )

    # Add torsional world.
    # NOTE: build_flexibilities() returns a nested + pre-converted structure that
    # add_robotic_world() does not accept (it expects a FLAT list of BondFlexibility
    # carrying RAW prmtop atom indices, which it converts to BAT internally). Build
    # that flat list directly — this is the working pattern used by production drivers.
    dihs = ["phi", "psi"]
    bonds = context.standard_dihedral_bonds.loc[
        context.standard_dihedral_bonds["dihedral_type"].isin(dihs)
    ]
    sele = []
    for _, bond in bonds.iterrows():
        flex = robosample.rb.BondFlexibility()
        flex.globalIndex1 = int(bond["atom1_parmed_index"])
        flex.globalIndex2 = int(bond["atom2_parmed_index"])
        flex.mobility = robosample.rb.BondMobility.Torsion
        sele.append(flex)
    context.add_robotic_world(sele).add_sampler(
        timeStep=0.01,
        mdSteps=10,
        boostMDSteps=10,
        acceptRejectMode=robosample.rb.AcceptRejectMode.MetropolisHastings,
        use_nuts=False,
    )

    # Add one replica at 300 K
    context.initialize([300.0])

    # Run the simulation for no equilibration steps and 100 production steps totaling 100 ps of simulation time
    context.run_rex(equil_steps, prod_steps, write_freq, True)


if __name__ == "__main__":
    try:
        run_test()
    except Exception as e:
        print(f"Error: {e}")
    else:
        print("\n\nRobosample installed successfully.")
