import random
import re
import os

import numpy as np
from collections import deque
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem
from rdkit.Chem.Draw import MolToImage

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from environment import DrugDesignEnv
from model import DQN
from agent import DRLAgent


def preprocess_smiles(smiles):
    preprocessed_smiles = re.sub(r'\[.*?\]', '', smiles)
    preprocessed_smiles = re.sub(r'[@]\S*', '', preprocessed_smiles)
    return preprocessed_smiles


def calculate_molecular_features(smiles):
    molecule = Chem.MolFromSmiles(smiles)
    features = {}

    if molecule is not None:
        features['Molecular Weight'] = Descriptors.MolWt(molecule)
        features['LogP'] = Descriptors.MolLogP(molecule)
        features['H-Bond Donor Count'] = Descriptors.NumHDonors(molecule)
        features['H-Bond Aceeptor Count'] = Descriptors.NumHAcceptors(molecule)

    return features


def get_fingerprint(smiles):
    mol = Chem.MolFromSmiles(smiles)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
    fp_array = np.array(list(fp.ToBitString())).astype(int)
    # selected_X_train.append(fp_array)
    return fp_array


def prepare_data():
    df = pd.read_csv(os.path.join("data", "SMILES_Big_Data_Set.csv"))
    smiles = df['SMILES'].tolist()

    X_train, y_train = train_test_split(smiles, test_size=0.2, random_state=9)
    print(len(X_train))
    print("####")
    print(len(y_train))
    num_train_samples = min(len(X_train), len(y_train))
    selected_indices = random.sample(range(num_train_samples), num_train_samples)
    X_train = [X_train[i] for i in selected_indices]
    y_train = [y_train[i] for i in selected_indices]

    selected_X_train = []

    for smiles in X_train:
        try:
            fp_array = get_fingerprint(smiles)
            selected_X_train.append(fp_array)
        except:
            print(f"Error converting SMILES to fingerprint: {smiles}")

    selected_X_train = np.array(selected_X_train)

    selected_y_train = []
    for smiles in y_train:
        try:
            fp_array = get_fingerprint(smiles)
            selected_y_train.append(fp_array)
        except:
            print(f"Error converting SMILES to fingerprint: {smiles}")

    selected_y_train = np.array(selected_y_train)

    return selected_X_train, selected_y_train, smiles


def simulate(agent, env, batch_size, max_episodes, X_train, y_train, num_actions):
    episode = 0
    rewards = []
    generated_smiles = []

    while episode < max_episodes:
        state = env.reset()
        done = False
        episode_reward = 0

        while not done:
            action = agent.act(state)
            next_state, reward, done = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            episode_reward += reward

            if len(agent.memory) > batch_size:
                agent.replay(batch_size)

        indices = np.random.choice(len(X_train), batch_size, replace=False)
        selected_X_train = torch.FloatTensor([X_train[i] for i in indices])
        selected_y_train = torch.FloatTensor(y_train[indices])

        dataset = TensorDataset(selected_X_train, selected_y_train)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        for data, label in loader:
            agent.optimizer.zero_grad()
            outputs = agent.model(data)
            loss = nn.MSELoss()(outputs, label)
            loss.backward()
            agent.optimizer.step()

        episode += 1
        agent.update_target_model()
        rewards.append(episode_reward)
        generated_smiles.append(env.get_smiles())

    return rewards, generated_smiles


if __name__ == '__main__':
    X_train, y_train, smiles = prepare_data()

    num_features = 2048
    num_actions = 5
    state_size = 10

    env = DrugDesignEnv(num_features, num_actions, None, None)
    agent = DRLAgent(state_size, num_actions, np.zeros((1, num_features)))
    rewards, generated_smiles = simulate(agent, env, batch_size=32, max_episodes=30, X_train=X_train, y_train=y_train,
                                         num_actions=num_actions)
    print(rewards)
    print(generated_smiles)
