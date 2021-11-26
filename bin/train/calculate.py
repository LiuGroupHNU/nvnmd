
import numpy as np
from ase.io import read
from calculator import DP


def calculate_process(atom_fn, graph_fn):
    atoms = read(atom_fn)
    cal = DP(graph_fn)

    print("INFO: atom\n", atoms)

    atoms.cell = np.round(atoms.cell * 2**14) / 2**14
    cal.calculate(atoms)
    print("INFO: energy\n", cal.results['energy'])
    print("INFO: forces\n", cal.results['forces'])
    print("INFO: stress\n", cal.results['stress'])

    prec = 2 ** 14
    print("INFO: energy\n", np.floor(cal.results['energy'] * prec))
    print("INFO: forces\n", np.floor(cal.results['forces'] * prec))
    print("INFO: stress\n", np.floor(cal.results['stress'] * prec))

    np.save("./res", [cal.results['energy'], cal.results['forces'], cal.results['stress']], allow_pickle=True)

def calculate (args):
    calculate_process(args.atoms, args.graph)