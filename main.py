from rdkit import Chem, DataStructs
from rdkit.Chem import Draw, Descriptors, AllChem

from itertools import combinations
from PIL import Image

import numpy as np
import pandas as pd

import networkx as nx
import seaborn as sns
import matplotlib.pyplot as plt

import os


def show_smiles(smiles):
    molecules = [Chem.MolFromSmiles(smile) for smile in smiles]
    img = Draw.MolsToGridImage(
        molecules[:10], molsPerRow=5, subImgSize=(400, 400),
        legends=[f'LogP: {round(x, 2)}' for x in df['logP']], returnPNG=False
    ).save('molecules.png')


def histogram_of_distribution(data_frame):
    """
    visualize the distribution Number of Atoms and Distribution of LogP
    :param data_frame: DF with num_atoms & logP
    """
    sns.histplot(data=data_frame, x='num_atoms', bins=20, kde=True)
    plt.title('Distribution of Number of Atoms')
    plt.xlabel('Number of Atoms')
    plt.ylabel('Frequency')
    plt.show()

    sns.histplot(data=data_frame, x='logP', bins=20, kde=True)
    plt.title('Distribution of LogP')
    plt.xlabel('LogP')
    plt.ylabel('Frequency')
    plt.show()


def num_atoms_and_logP(data_frame):
    """plot the relationship between logP and number of atoms
    :param data_frame: DF with num_atoms & logP
    :return:
    """
    sns.scatterplot(data=data_frame, x='num_atoms', y='logP')
    sns.despine()
    sns.set_style("whitegrid")
    plt.title('LogP vs. Number of Atoms')
    plt.xlabel('Number of Atoms')
    plt.ylabel('LogP')
    plt.show()


def atom_frequency(smiles_list):
    atom_counts = []
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        atoms = mol.GetAtoms()
        atom_counts.extend([atom.GetSymbol() for atom in atoms])

    plot_data = pd.Series(atom_counts).value_counts().sort_index()
    sns.barplot(x=plot_data.index, y=plot_data.values)
    plt.show()


if __name__ == '__main__':
    DATAPATH = "data"
    df = pd.read_csv(os.path.join(DATAPATH, "SMILES_Big_Data_Set.csv"))
    print(df.head())
    #show_smiles(df["SMILES"])
    #histogram_of_distribution(df)
    #num_atoms_and_logP(df)
    atom_frequency(df["SMILES"])