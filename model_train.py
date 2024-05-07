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


class DQN(nn.Module):
    def __init__(self, input_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_size, 32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, 16)
        self.out = nn.Linear(16, action_size)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.out(x)
        return x


class DRLAgent:
    def __init__(self, state_size, action_size, selected_X_train):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95  # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.selected_X_train = selected_X_train
        #self.num_actions = num_actions

        self.model = DQN(selected_X_train.shape[1], action_size)

        # target model for stability
        self.target_model = DQN(selected_X_train.shape[1], action_size)
        self.update_target_model()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def remember(self, state, action, reward, next_state, done):
        """
        Store the experience in the replay memory
        :param state: Current state (ndarray)
        :param action: Action taken (int)
        :param reward: Reward received (float)
        :param next_state: Next state (ndarray)
        :param done: Episode is done? (bool)
        :return:
        """
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        """
        Choose an action
        :param state: Current state (ndarray)
        :return: action: Chosen action (int)
        """
        if np.random.rand() <= self.epsilon:
            return np.random.randint(self.action_size)
        state = torch.from_numpy(state).float().unsqueeze(0)
        with torch.no_grad():
            action_values = self.model(state)
        return torch.argmax(action_values).item()

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return

        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            state = torch.from_numpy(state).float().unsqueeze(0)
            next_state = torch.from_numpy(next_state).float().unsqueeze(0)
            action, reward = torch.tensor([action]), torch.tensor([reward])

            if not done:
                target = reward + self.gamma * torch.max(self.target_model(next_state)).item()
            else:
                target = reward

            current_q = self.model(state).gather(1, action.unsqueeze(1))
            loss = nn.MSELoss()(current_q.squeeze(1), target)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, path):
        self.model.load_state_dict(torch.load(path))

    def save(self, path):
        torch.save(self.model.state_dict(), path)


class DrugDesignEnv:
    def __init__(self, num_features, num_actions, X, y):
        self.num_features = num_features
        self.num_actions = num_actions
        self.current_state = np.zeros((num_features,))
        self.reward = 0
        self.target = np.ones((self.num_features,))
        self.max_steps = 10
        self.step_count = 0
        self.generated_smiles = []
        self.X = X
        self.y = y

    def _get_reward(self):
        similarity = np.dot(self.current_state, self.target)
        return similarity

    def _is_done(self):
        self.step_count += 1
        return self.step_count >= self.max_steps

    def get_smiles(self):
        binary_string = ''.join([str(int(x)) for x in self.current_state])
        return binary_string

    def step(self, action):
        action = max(0, min(action, self.num_actions - 1))
        next_state = np.zeros((self.num_features,))
        next_state[action] = 1.0
        self.current_state = next_state
        reward = self._get_reward()
        done = self._is_done()
        self.generated_smiles.append(action)
        return self.current_state, reward, done

    def reset(self):
        self.current_state = np.zeros((self.num_features,))
        self.generated_smiles = []
        return self.current_state


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
