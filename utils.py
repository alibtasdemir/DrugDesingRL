import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs

# import config


def atom_valences(atom_types):
    """
    Creates a list of valences for corresponding atom types.

    It will return the maximum number of bonds each element will make. For example for 'C' -> 4, 'O' -> 2.

    :param atom_types: List. List of string atom types. e.g. ['C', 'O']
    :return: List. List of integer atom valences.
    """
    pt = Chem.GetPeriodicTable()
    return [
        max(list(pt.GetValenceList(atom_type))) for atom_type in atom_types
    ]


def get_fingerprint(smiles, fp_length, fp_radius):
    if smiles is None:
        return np.zeros((fp_length,))
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        return np.zeros((fp_length,))
    fingerprint = AllChem.GetMorganFingerprintAsBitVect(
        molecule,
        fp_radius,
        fp_length
    )
    arr = np.zeros((1,))
    DataStructs.ConvertToNumpyArray(fingerprint, arr)
    return arr
