"""Build a C7ax / alpha_L starting structure (phi~+60 deg, psi~-70 deg) for the trapped-mode
race. The forward jump into C7ax is a ~8 kcal/mol rare event; to observe barrier crossing
fairly in minutes we start BOTH methods INSIDE C7ax and race the escape/recrossing of phi=0.
Restrain phi->+60, psi->-70, minimize, briefly relax at 300 K under restraint, save rst7."""
import numpy as np, mdtraj as md
import openmm as mm, openmm.app as app, openmm.unit as u

R = "/home/alexb/Robosample_disasm"; PRM = R+"/examples/ala-dipeptide.prmtop"; RST = R+"/examples/ala-dipeptide.rst7"
OUT = R+"/examples/ala-dipeptide-c7ax.rst7"
TOP = md.load_prmtop(PRM)
phi_idx = md.compute_phi(md.load(RST, top=PRM))[0][0]   # 4 atom indices
psi_idx = md.compute_psi(md.load(RST, top=PRM))[0][0]
print(f"phi atoms={phi_idx}  psi atoms={psi_idx}")

system = app.AmberPrmtopFile(PRM).createSystem(nonbondedMethod=app.NoCutoff, implicitSolvent=app.OBC2, constraints=app.HBonds)
# harmonic dihedral restraints (0.5*k*(theta-theta0)^2) driving into C7ax
rest = mm.CustomTorsionForce("0.5*k*min(dtheta, 2*pi-dtheta)^2; dtheta=abs(theta-theta0); pi=3.14159265")
rest.addPerTorsionParameter("k"); rest.addPerTorsionParameter("theta0")
K = 1000.0  # kJ/mol/rad^2, stiff
rest.addTorsion(*[int(i) for i in phi_idx], [K, np.radians(60.0)])
rest.addTorsion(*[int(i) for i in psi_idx], [K, np.radians(-70.0)])
system.addForce(rest)

integ = mm.LangevinMiddleIntegrator(300*u.kelvin, 1.0/u.picosecond, 0.002*u.picoseconds)
ctx = mm.Context(system, integ, mm.Platform.getPlatformByName("CUDA"))
ctx.setPositions(app.AmberInpcrdFile(RST).positions)
mm.LocalEnergyMinimizer.minimize(ctx)
integ.step(25000)  # 50 ps relax into the restrained basin
pos = ctx.getState(getPositions=True).getPositions(asNumpy=True)

# verify
xyz = pos.value_in_unit(u.nanometer)[None]
phi = np.degrees(md.compute_phi(md.Trajectory(xyz, TOP))[1][0, 0])
psi = np.degrees(md.compute_psi(md.Trajectory(xyz, TOP))[1][0, 0])
print(f"after restrained relax: phi={phi:.1f}  psi={psi:.1f}  (target +60 / -70)")

# save as rst7 via parmed
import parmed as pmd
st = pmd.load_file(PRM, RST); st.coordinates = pos.value_in_unit(u.angstrom)
st.save(OUT, format="rst7", overwrite=True)
print(f"wrote {OUT}")
