import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs

import config


def atom_valences(atom_types):
    periodic_table = Chem.GetPeriodicTable()
    return [
        max(list(periodic_table.GetValenceList(atom_type))) for atom_type in atom_types
    ]


def get_fingerprint(smiles, fingerprint_length, fingerprint_radius):
    if smiles is None:
        return np.zeros((config.fingerprint_length, ))
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        return np.zeros((config.fingerprint_length, ))
    fingerprint = AllChem.GetMorganFingerprintAsBitVect(
        molecule, config.fingerprint_radius, config.fingerprint_length
    )
    arr = np.zeros((1,))
    DataStructs.ConvertToNumpyArray(fingerprint, arr)
    return arr