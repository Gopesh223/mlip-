import numpy as np
from ase.filters import FrechetCellFilter
from ase.optimize import BFGS

EV_A3_TO_GPA = 160.21766208

def relax_full(atoms, fmax=0.005, steps=300):
    BFGS(FrechetCellFilter(atoms), logfile=None).run(fmax=fmax, steps=steps)
    return atoms

def elastic_tensor(atoms, delta=0.005, fmax=0.005):
    """6x6 elastic tensor (GPa) from central-difference stress-strain.
    Atoms must already have a calculator and a relaxed cell.
    Voigt order: xx, yy, zz, yz, xz, xy (same as ASE)."""
    cell0 = atoms.get_cell().array.copy()
    pos0 = atoms.get_scaled_positions()
    C = np.zeros((6, 6))
    for j in range(6):
        sig = []
        for s in (+1, -1):
            eps = np.zeros((3, 3))
            if j < 3:
                eps[j, j] = s * delta
            else:  # engineering shear gamma = 2*eps_ij
                a, b = {3: (1, 2), 4: (0, 2), 5: (0, 1)}[j]
                eps[a, b] = eps[b, a] = s * delta / 2
            at = atoms.copy()
            at.calc = atoms.calc
            at.set_cell(cell0 @ (np.eye(3) + eps), scale_atoms=False)
            at.set_scaled_positions(pos0)
            BFGS(at, logfile=None).run(fmax=fmax, steps=200)  # relax ions only
            sig.append(at.get_stress(voigt=True))
        C[:, j] = (sig[0] - sig[1]) / (2 * delta)
    C = 0.5 * (C + C.T) * EV_A3_TO_GPA
    return C

def vrh_moduli(C):
    K_v = (C[0,0]+C[1,1]+C[2,2] + 2*(C[0,1]+C[0,2]+C[1,2])) / 9
    G_v = (C[0,0]+C[1,1]+C[2,2] - (C[0,1]+C[0,2]+C[1,2])
           + 3*(C[3,3]+C[4,4]+C[5,5])) / 15
    try:
        S = np.linalg.inv(C)
        K_r = 1 / (S[0,0]+S[1,1]+S[2,2] + 2*(S[0,1]+S[0,2]+S[1,2]))
        G_r = 15 / (4*(S[0,0]+S[1,1]+S[2,2]) - 4*(S[0,1]+S[0,2]+S[1,2])
                    + 3*(S[3,3]+S[4,4]+S[5,5]))
    except np.linalg.LinAlgError:
        K_r, G_r = np.nan, np.nan
    return (K_v + K_r) / 2, (G_v + G_r) / 2
