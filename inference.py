import argparse
import os

import torch
from rdkit import Chem
from rdkit.Chem import QED, Descriptors, Draw

from agent import QEDRewardMolecule, Agent
import utils
import numpy as np
import random
import time


class Inference(object):
    def __init__(self, config):
        if config.seed:
            np.random.seed(config.seed)
            random.seed(config.seed)
            torch.manual_seed(config.seed)
            torch.cuda.manual_seed(config.seed)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.start_time = None
        self.end_time = None

        self.max_steps_per_episode = config.max_steps_per_episode

        self.fingerprint_length = config.fingerprint_length
        self.fingerprint_radius = config.fingerprint_radius

        self.num_samples = config.num_samples
        self.save_molecules = config.save_molecules

        self.samples = []
        self.sample_paths = []

        self.eps_threshold = 1.0

        self.environment = QEDRewardMolecule(
            discount_factor=config.discount_factor,
            atom_types=set(config.atom_types),
            init_mol=config.start_molecule,
            allow_removal=config.allow_removal,
            allow_no_modification=config.allow_no_modification,
            allow_bonds_between_rings=config.allow_bonds_between_rings,
            allowed_ring_sizes=set(config.allowed_ring_sizes),
            max_steps=config.max_steps_per_episode,
            record_path=True
        )

        self.agent = Agent(
            self.fingerprint_length + 1,
            1,
            self.device,
            config,
            pretrained=True
        )

    def validate(self):
        rewards = []
        reward = None
        self.start_time = time.time()
        for _ in range(self.num_samples):
            self.environment.initialize()
            done = False
            while not done:
                steps_left = self.max_steps_per_episode - self.environment.num_steps_taken
                valid_actions = list(self.environment.get_valid_actions())

                observations = np.vstack(
                    [
                        np.append(
                            utils.get_fingerprint(act, self.fingerprint_length, self.fingerprint_radius),
                            steps_left,
                        )
                        for act in valid_actions
                    ]
                )
                observations_tensor = torch.Tensor(observations)

                act_index = self.agent.get_action(observations_tensor, self.eps_threshold)
                action = valid_actions[act_index]

                result = self.environment.step(action)
                next_state, reward, done = result

            rewards.append(reward)
            self.samples.append(self.environment.state)
            self.sample_paths.append(self.environment.get_path())

        self.environment.initialize()
        self.end_time = time.time()
        print(f"Generated {self.num_samples} samples in {self.end_time - self.start_time} seconds\n"
              f"Average QED: {round(np.array(rewards).mean(), 4)}")

        if self.save_molecules:
            print("Saving images to inference folder...")
            self.save_images()

    def save_images(self):
        os.makedirs("inference", exist_ok=True)
        qeds = []
        logps = []
        molecules = []
        for sidx, (sample, spath) in enumerate(zip(self.samples, self.sample_paths)):
            curr_mol = Chem.MolFromSmiles(sample)
            molecules.append(curr_mol)
            reward = QED.qed(curr_mol)
            qeds.append(reward)
            logp = Descriptors.MolLogP(curr_mol)
            logps.append(logp)
            self.environment.visualize_state(
                sample,
                legend=f"QED: {round(reward, 3)}\nLogP: {round(logp, 3)}"
            ).save(os.path.join("inference", f"sample{sidx}.png"))
            frames = []
            for i, mol in enumerate(spath):
                if i == 0:
                    continue
                reward = QED.qed(Chem.MolFromSmiles(mol))
                img = self.environment.visualize_state(mol, legend=f"Step: {i}\nQED: {round(reward, 3)}")
                frames.append(img)

            frame_one = frames[0]
            frame_one.save(
                os.path.join("inference", f"sample{sidx}_path.gif"),
                format="GIF",
                append_images=frames,
                save_all=True,
                duration=1000,
                loop=0
            )

        Draw.MolsToGridImage(
            molecules[:10],
            molsPerRow=5,
            subImgSize=(400, 400),
            legends=[f'LogP: {round(logp, 2)}\nQED: {round(qed, 2)}' for logp, qed in zip(logps[:10], qeds[:10])],
            returnPNG=False
        ).save(os.path.join("inference", f"sample_grid.png"))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Set seed for reproducibility
    parser.add_argument('--seed', type=int, default=9, help="Seed for training")

    # Model settings
    parser.add_argument('--model_path', type=str, default="dqn_weights.pth")
    parser.add_argument('--fingerprint_length', type=int, default=1024)
    parser.add_argument('--fingerprint_radius', type=int, default=3)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--replay_buffer_size', type=int, default=10000)

    # Environment settings
    parser.add_argument('--discount_factor', type=float, default=0.9)
    parser.add_argument('--atom_types', type=str, nargs="+", default=["C", "O", "N"])
    parser.add_argument('--start_molecule', type=str, default=None)
    parser.add_argument('--allow_removal', type=bool, default=True)
    parser.add_argument('--allow_no_modification', type=bool, default=True)
    parser.add_argument('--allow_bonds_between_rings', type=bool, default=False)
    parser.add_argument('--allowed_ring_sizes', type=int, nargs="+", default=[3, 4, 5, 6])
    parser.add_argument('--max_steps_per_episode', type=int, default=40)

    # Inference parameters
    parser.add_argument('--num_samples', type=int, default=10)
    parser.add_argument('--save_molecules', type=bool, default=True)


    parser.add_argument('--max_episodes', type=int, default=100)
    parser.add_argument('--epsilon_start', type=float, default=1.0)
    parser.add_argument('--epsilon_end', type=float, default=0.01)
    parser.add_argument('--epsilon_decay', type=int, default=2000)
    parser.add_argument('--polyak', type=float, default=0.995)
    parser.add_argument('--gamma', type=float, default=0.95)

    parser.add_argument('--update_interval', type=int, default=20)
    parser.add_argument('--log_interval', type=int, default=2)

    config = parser.parse_args()
    inference = Inference(config)
    inference.validate()
