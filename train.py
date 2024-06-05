import argparse
import os

import torch
import wandb
from agent import QEDRewardMolecule, Agent
import utils
import numpy as np
import random
import time


class Train(object):
    def __init__(self, config):
        if config.seed:
            np.random.seed(config.seed)
            random.seed(config.seed)
            torch.manual_seed(config.seed)
            torch.cuda.manual_seed(config.seed)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "mps")
        self.save_model = config.save_model
        self.save_path = config.save_path
        self.start_time = None
        self.end_time = None
        self.log_interval = config.log_interval

        self.batch_size = config.batch_size
        self.max_steps_per_episode = config.max_steps_per_episode
        self.update_interval = config.update_interval
        self.max_episodes = config.max_episodes
        self.gamma = config.gamma
        self.polyak = config.polyak

        self.fingerprint_length = config.fingerprint_length
        self.fingerprint_radius = config.fingerprint_radius

        self.current_iteration = 0
        self.current_episode = 0
        self.num_updates_per_it = 1
        self.eps_threshold = 1.0
        self.batch_losses = []
        self.final_reward = None

        self.max_rewards = [-np.inf]
        self.max_rewards_smiles = [None]
        self.max_rewards_min_idx = self.max_rewards.index(min(self.max_rewards))

        self.environment = QEDRewardMolecule(
            discount_factor=config.discount_factor,
            atom_types=set(config.atom_types),
            init_mol=config.start_molecule,
            allow_removal=config.allow_removal,
            allow_no_modification=config.allow_no_modification,
            allow_bonds_between_rings=config.allow_bonds_between_rings,
            allowed_ring_sizes=set(config.allowed_ring_sizes),
            max_steps=config.max_steps_per_episode,
        )

        self.agent = Agent(
            self.fingerprint_length + 1,
            1,
            self.device,
            config
        )

        self.exp_name = config.exp_name

    def train_episode(self):
        wandb.log({'iteration': self.current_iteration})
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

        action_fingerprint = np.append(
            utils.get_fingerprint(action, self.fingerprint_length, self.fingerprint_radius),
            steps_left
        )

        # Update left steps
        steps_left = self.max_steps_per_episode - self.environment.num_steps_taken

        next_state = utils.get_fingerprint(next_state, self.fingerprint_length, self.fingerprint_radius)
        action_fingerprints = np.vstack(
            [
                np.append(
                    utils.get_fingerprint(act, self.fingerprint_length, self.fingerprint_radius),
                    steps_left,
                )
                for act in self.environment.get_valid_actions()
            ]
        )

        self.agent.replay_buffer.add(
            obs_t=action_fingerprint,
            action=0,
            reward=reward,
            obs_tp1=action_fingerprints,
            done=float(result.terminated)
        )

        if done:
            self.final_reward = reward
            self.episode_end()

        if self.current_iteration % self.update_interval == 0 and self.agent.replay_buffer.__len__() >= self.batch_size:
            self.update_model()

        self.current_iteration += 1

    def update_model(self):
        for _ in range(self.num_updates_per_it):
            loss = self.agent.update_params(self.batch_size, self.gamma, self.polyak)
            loss_value = loss.item()
            self.batch_losses.append(loss_value)

    def episode_end(self):
        if self.current_episode != 0 and len(self.batch_losses) != 0:
            wandb.log({'episode_final_reward': self.final_reward})
            wandb.log({'episode_loss': np.array(self.batch_losses).mean()})

        if self.current_episode != 0 and self.current_episode % self.log_interval == 0 and len(self.batch_losses) != 0:
            print(
                "reward of final molecule at episode {} is {:.3f}".format(
                    self.current_episode, self.final_reward
                )
            )
            print(
                "mean loss in episode {} is {:.3f}".format(
                    self.current_episode, np.array(self.batch_losses).mean()
                )
            )

        # The final state
        curr_state = self.environment.state

        if len(self.max_rewards) < 10:
            self.max_rewards.append(self.final_reward)
            self.max_rewards_smiles.append(curr_state)
        elif self.final_reward > self.max_rewards[self.max_rewards_min_idx]:
            self.max_rewards_min_idx = self.max_rewards.index(min(self.max_rewards))

            self.max_rewards.remove(self.max_rewards[self.max_rewards_min_idx])
            self.max_rewards_smiles.remove(self.max_rewards_smiles[self.max_rewards_min_idx])

            self.max_rewards.append(self.final_reward)
            self.max_rewards_smiles.append(curr_state)

        self.validate(3)

        self.current_episode += 1
        wandb.log({"episode": self.current_episode})
        self.eps_threshold *= 0.99907

        self.batch_losses = []

        self.environment.initialize()

    def validate(self, episodes):
        molecules = []
        rewards = []
        for _ in range(episodes):
            self.environment.initialize()
            done = False
            reward = None
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
            molecules.append(self.environment.state)

        wandb.log({'val-reward': np.array(rewards).mean()})
        self.environment.initialize()

    def train(self, config):
        kwargs = {
            'name': self.exp_name,
            'project': "RLDrugDesign",
            "config": config,
            "settings": wandb.Settings(_disable_stats=True),
            'reinit': True,
            'save_code': True
        }

        wandb.init(**kwargs)
        print("Start training...")
        self.start_time = time.time()
        self.environment.initialize()

        while self.current_episode < self.max_episodes:
            self.train_episode()

        outf = open("result_smiles.csv", "w")
        outf.write("Reward;Smiles\n")
        for smiles_text, max_reward in zip(self.max_rewards, self.max_rewards_smiles):
            outf.write(f"{smiles_text};{max_reward}\n")
        outf.close()

        self.end_time = time.time()
        print("Training is done in ", (self.end_time - self.start_time))

        if self.save_model:
            os.makedirs(f"{self.exp_name}", exist_ok=True)
            full_save_path = os.path.join(self.exp_name, self.save_path)
            torch.save(self.agent.dqn.state_dict(), full_save_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', type=int, default=9, help="Seed for training")
    parser.add_argument('--batch_size', type=int, default=128, help="Batch size for training")
    parser.add_argument('--max_steps_per_episode', type=int, default=40)
    parser.add_argument('--max_episodes', type=int, default=100)
    parser.add_argument('--epsilon_start', type=float, default=1.0)
    parser.add_argument('--epsilon_end', type=float, default=0.01)
    parser.add_argument('--epsilon_decay', type=int, default=2000)
    parser.add_argument('--polyak', type=float, default=0.995)
    parser.add_argument('--gamma', type=float, default=0.95)
    parser.add_argument('--discount_factor', type=float, default=0.9)
    parser.add_argument('--replay_buffer_size', type=int, default=10000)
    parser.add_argument('--fingerprint_length', type=int, default=1024)
    parser.add_argument('--fingerprint_radius', type=int, default=3)
    parser.add_argument('--update_interval', type=int, default=20)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--atom_types', type=str, nargs="+", default=["C", "O", "N"])
    parser.add_argument('--allowed_ring_sizes', type=int, nargs="+", default=[3, 4, 5, 6])
    parser.add_argument('--start_molecule', type=str, default=None)

    parser.add_argument('--allow_removal', action="store_true")
    parser.add_argument('--allow_no_modification', action="store_true")
    parser.add_argument('--allow_bonds_between_rings', action="store_true")

    parser.add_argument('--exp_name', type=str, default='RLDrugv1_run1', help='experiment name')
    parser.add_argument('--log_interval', type=int, default=2)
    parser.add_argument('--save_model', action="store_true")
    parser.add_argument('--save_path', type=str, default="dqn_model.pth")

    config = parser.parse_args()
    trainer = Train(config)
    trainer.train(config)
