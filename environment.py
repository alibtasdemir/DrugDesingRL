# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python2, python3
"""
Defines the Markov decision process of generating a molecule.

The problem of molecule generation as a Markov decision process, the
state space, action space, and reward function are defined.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import copy
import itertools

from rdkit import Chem
from rdkit.Chem import Draw
from six.moves import range
from six.moves import zip

import utils


class Result(collections.namedtuple("Result", ["state", "reward", "terminated"])):
    """
    A structured object to store the response of the environment to the agent.
    state: Chem.RWMol. The molecule reached at the current step
    reward: Float. The QED reward get after taking the action
    terminated: bool. Flag value for terminal state
    """


def get_valid_actions(
    state,
    atom_types,
    allow_removal,
    allow_no_modification,
    allowed_ring_sizes,
    allow_bonds_between_rings,
):
    """
    Search the valid options for a given molecule
    :param state: str. SMILES string of the molecule
    :param atom_types: Set of string atom types
    :param allow_removal: bool. Flag value for allowing removal operations
    :param allow_no_modification: bool. Flag value to set "no-op" action
    :param allowed_ring_sizes: Set of integer values for allowed ring sizes
    :param allow_bonds_between_rings: bool. Flag value for allowing bonds between ring structures
    :return: Set of SMILES contains the valid next states
    """
    if not state:
        # Available actions are adding a node of each type.
        return copy.deepcopy(atom_types)
    mol = Chem.MolFromSmiles(state)
    if mol is None:
        raise ValueError('Received invalid state: %s' % state)
    atom_valences = {
        atom_type: utils.atom_valences([atom_type])[0]
        for atom_type in atom_types
    }
    atoms_with_free_valence = {}
    for i in range(1, max(atom_valences.values())):
        # Only atoms that allow us to replace at least one H with a new bond are
        # enumerated here.
        atoms_with_free_valence[i] = [
            atom.GetIdx() for atom in mol.GetAtoms() if atom.GetNumImplicitHs() >= i
        ]
    valid_actions = set()
    valid_actions.update(
        _atom_addition(
            mol,
            atom_types=atom_types,
            atom_valences=atom_valences,
            atoms_with_free_valence=atoms_with_free_valence))
    valid_actions.update(
        _bond_addition(
            mol,
            atoms_with_free_valence=atoms_with_free_valence,
            allowed_ring_sizes=allowed_ring_sizes,
            allow_bonds_between_rings=allow_bonds_between_rings))
    if allow_removal:
        valid_actions.update(_bond_removal(mol))
    if allow_no_modification:
        valid_actions.add(Chem.MolToSmiles(mol))
    return valid_actions


def _atom_addition(state, atom_types, atom_valences, atoms_with_free_valence):
    """
    Computes valid actions that involve adding atoms to the molecule.
    :param state: RDKit. Molecule
    :param atom_types: Set of string atom types.
    :param atom_valences: Dict mapping string atom types to integer valences.
    :param atoms_with_free_valence: Dict mapping integer minimum available valence
    :return: Set of string SMILES; the available actions.
    """
    bond_order = {
        1: Chem.BondType.SINGLE,
        2: Chem.BondType.DOUBLE,
        3: Chem.BondType.TRIPLE,
    }
    atom_addition = set()
    for i in bond_order:
        for atom in atoms_with_free_valence[i]:
            for element in atom_types:
                if atom_valences[element] >= i:
                    new_state = Chem.RWMol(state)
                    idx = new_state.AddAtom(Chem.Atom(element))
                    new_state.AddBond(atom, idx, bond_order[i])
                    sanitization_result = Chem.SanitizeMol(new_state, catchErrors=True)
                    # When sanitization fails
                    if sanitization_result:
                        continue
                    atom_addition.add(Chem.MolToSmiles(new_state))
    return atom_addition


def _bond_addition(
    state, atoms_with_free_valence, allowed_ring_sizes, allow_bonds_between_rings
):
    """
    Computes valid actions that involve adding bonds to the molecule.
    :param state: RDKit. Molecule
    :param atoms_with_free_valence: Dict mapping integer minimum available valence
    :param allowed_ring_sizes: Set of integer allowed ring sizes
    :param allow_bonds_between_rings: bool. Flag value for allowing bonds between ring structures
    :return: Set of string SMILES; the available actions.
    """
    bond_orders = [
        None,
        Chem.BondType.SINGLE,
        Chem.BondType.DOUBLE,
        Chem.BondType.TRIPLE,
    ]
    bond_addition = set()
    for valence, atoms in atoms_with_free_valence.items():
        for atom1, atom2 in itertools.combinations(atoms, 2):
            # Get the bond from a copy of the molecule so that SetBondType() doesn't
            # modify the original state.
            bond = Chem.Mol(state).GetBondBetweenAtoms(atom1, atom2)
            new_state = Chem.RWMol(state)
            # Kekulize the new state to avoid sanitization errors; note that bonds
            # that are aromatic in the original state are not modified (this is
            # enforced by getting the bond from the original state with
            # GetBondBetweenAtoms()).
            Chem.Kekulize(new_state, clearAromaticFlags=True)
            if bond is not None:
                if bond.GetBondType() not in bond_orders:
                    continue  # Skip aromatic bonds.
                idx = bond.GetIdx()
                # Compute the new bond order as an offset from the current bond order.
                bond_order = bond_orders.index(bond.GetBondType())
                bond_order += valence
                if bond_order < len(bond_orders):
                    idx = bond.GetIdx()
                    bond.SetBondType(bond_orders[bond_order])
                    new_state.ReplaceBond(idx, bond)
                else:
                    continue
            # If do not allow new bonds between atoms already in rings.
            elif not allow_bonds_between_rings and (
                state.GetAtomWithIdx(atom1).IsInRing()
                and state.GetAtomWithIdx(atom2).IsInRing()
            ):
                continue
            # If the distance between the current two atoms is not in the
            # allowed ring sizes
            elif (
                allowed_ring_sizes is not None
                and len(Chem.rdmolops.GetShortestPath(state, atom1, atom2))
                not in allowed_ring_sizes
            ):
                continue
            else:
                new_state.AddBond(atom1, atom2, bond_orders[valence])
            sanitization_result = Chem.SanitizeMol(new_state, catchErrors=True)
            # When sanitization fails
            if sanitization_result:
                continue
            bond_addition.add(Chem.MolToSmiles(new_state))
    return bond_addition


def _bond_removal(state):
    """
    Computes valid actions that involve removing bonds from the molecule.
    :param state: RDKit Molecule
    :return: Set of string SMILES; the available actions.
    """
    bond_orders = [
        None,
        Chem.BondType.SINGLE,
        Chem.BondType.DOUBLE,
        Chem.BondType.TRIPLE,
    ]
    bond_removal = set()
    for valence in [1, 2, 3]:
        for bond in state.GetBonds():
            # Get the bond from a copy of the molecule so that SetBondType() doesn't
            # modify the original state.
            bond = Chem.Mol(state).GetBondBetweenAtoms(
                bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            )
            if bond.GetBondType() not in bond_orders:
                continue  # Skip aromatic bonds.
            new_state = Chem.RWMol(state)
            # Kekulize the new state to avoid sanitization errors; note that bonds
            # that are aromatic in the original state are not modified (this is
            # enforced by getting the bond from the original state with
            # GetBondBetweenAtoms()).
            Chem.Kekulize(new_state, clearAromaticFlags=True)
            # Compute the new bond order as an offset from the current bond order.
            bond_order = bond_orders.index(bond.GetBondType())
            bond_order -= valence
            if bond_order > 0:  # Downgrade this bond.
                idx = bond.GetIdx()
                bond.SetBondType(bond_orders[bond_order])
                new_state.ReplaceBond(idx, bond)
                sanitization_result = Chem.SanitizeMol(new_state, catchErrors=True)
                # When sanitization fails
                if sanitization_result:
                    continue
                bond_removal.add(Chem.MolToSmiles(new_state))
            elif bond_order == 0:  # Remove this bond entirely.
                atom1 = bond.GetBeginAtom().GetIdx()
                atom2 = bond.GetEndAtom().GetIdx()
                new_state.RemoveBond(atom1, atom2)
                sanitization_result = Chem.SanitizeMol(new_state, catchErrors=True)
                # When sanitization fails
                if sanitization_result:
                    continue
                smiles = Chem.MolToSmiles(new_state)
                parts = sorted(smiles.split("."), key=len)
                # We define the valid bond removing action set as the actions
                # that remove an existing bond, generating only one independent
                # molecule, or a molecule and an atom.
                if len(parts) == 1 or len(parts[0]) == 1:
                    bond_removal.add(parts[-1])
    return bond_removal


class Molecule(object):
    """
    Defines the Markov decision process of generating a molecule
    """

    def __init__(
        self,
        atom_types,
        init_mol=None,
        allow_removal=True,
        allow_no_modification=True,
        allow_bonds_between_rings=True,
        allowed_ring_sizes=None,
        max_steps=10,
        target_fn=None,
        record_path=False,
    ):
        """
        Initializes the parameters for the MDP.
        :param atom_types: The set of elements the molecule may contain
        :param init_mol: str. Initial molecule to start with
        :param allow_removal: bool. Flag value for allowing removal operations
        :param allow_no_modification: bool. Flag value to set "no-op" action
        :param allow_bonds_between_rings: bool. Flag value for allowing bonds between ring structures
        :param allowed_ring_sizes: Set of integer values for allowed ring sizes
        :param max_steps: int. Max steps to run the environment
        :param target_fn: A function or None.
        :param record_path: bool. Whether to record the steps internally.
        """
        if isinstance(init_mol, Chem.Mol):
            init_mol = Chem.MolToSmiles(init_mol)
        self.init_mol = init_mol
        self.atom_types = atom_types
        self.allow_removal = allow_removal
        self.allow_no_modification = allow_no_modification
        self.allow_bonds_between_rings = allow_bonds_between_rings
        self.allowed_ring_sizes = allowed_ring_sizes
        self.max_steps = max_steps
        self._state = None
        self._valid_actions = []
        # The status should be 'terminated' if initialize() is not called.
        self._counter = self.max_steps
        self._target_fn = target_fn
        self.record_path = record_path
        self._path = []
        self._max_bonds = 4
        atom_types = list(self.atom_types)
        self._max_new_bonds = dict(
            list(zip(atom_types, utils.atom_valences(atom_types)))
        )

    @property
    def state(self):
        return self._state

    @property
    def num_steps_taken(self):
        return self._counter

    def get_path(self):
        return self._path

    def initialize(self):
        """
        Resets the MDP to the initial state. Restarts the environment.
        """
        self._state = self.init_mol
        if self.record_path:
            self._path = [self._state]
        self._valid_actions = self.get_valid_actions(force_rebuild=True)
        self._counter = 0

    def get_valid_actions(self, state=None, force_rebuild=False):
        """
        Gets the valid actions for the state.
        :param state: str. The current molecule in the state
        :param force_rebuild: bool. Flag value to force rebuild of the valid action set
        :return: A set contains all the valid actions for the state
        """
        if state is None:
            if self._valid_actions and not force_rebuild:
                return copy.deepcopy(self._valid_actions)
            state = self._state
        if isinstance(state, Chem.Mol):
            state = Chem.MolToSmiles(state)
        self._valid_actions = get_valid_actions(
            state,
            atom_types=self.atom_types,
            allow_removal=self.allow_removal,
            allow_no_modification=self.allow_no_modification,
            allowed_ring_sizes=self.allowed_ring_sizes,
            allow_bonds_between_rings=self.allow_bonds_between_rings,
        )
        return copy.deepcopy(self._valid_actions)

    def _reward(self):
        """
        Gets the reward function. Implemented as dummy.
        :return: The reward for the current state
        """
        return 0.0

    def _goal_reached(self):
        """
        Sets the termination criterion for molecule
        :return: bool. Flag value if reached the goal.
        """
        if self._target_fn is None:
            return False
        return self._target_fn(self._state)

    def step(self, action):
        """
        akes a step forward according to the action.
        :param action: Chem.RWMol. The target molecule for the action
        :return: Result. Namedtuple defined with
            * state: The molecule reached after taking the action.
            * reward: The reward get after taking the action.
            * terminated: Whether this episode is terminated.
        """
        if self._counter >= self.max_steps or self._goal_reached():
            raise ValueError("This episode is terminated.")
        if action not in self._valid_actions:
            raise ValueError("Invalid action.")
        self._state = action
        if self.record_path:
            self._path.append(self._state)
        self._valid_actions = self.get_valid_actions(force_rebuild=True)
        self._counter += 1
        result = Result(
            state=self._state,
            reward=self._reward(),
            terminated=(self._counter >= self.max_steps) or self._goal_reached(),
        )
        return result

    def visualize_state(self, state=None, **kwargs):
        """
        Draws the current molecule.
        :param state: String. If string is given it will be drawn. Else the current state will be used.
        :param kwargs: The keyword arguments to pass Draw.MolToImage
        :return: A PIL image containing a drawing of the molecule
        """
        if state is None:
            state = self._state
        if isinstance(state, str):
            state = Chem.MolFromSmiles(state)
        return Draw.MolToImage(state, **kwargs)