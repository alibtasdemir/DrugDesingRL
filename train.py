import torch
from agent import QEDRewardMolecule, Agent
import config
import math
import heapq
import utils
import numpy as np
from torch.utils.tensorboard import SummaryWriter

TENSORBOARD_LOG = True
TB_LOG_PATH = "./runs/dqn/run2"
episodes = 0
iterations = 4000
update_interval = 20
batch_size = 64
num_updates_per_it = 1

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

environment = QEDRewardMolecule(
    discount_factor=config.discount_factor,
    atom_types=set(config.atom_types),
    init_mol=config.start_molecule,
    allow_removal=config.allow_removal,
    allow_no_modification=config.allow_no_modification,
    allow_bonds_between_rings=config.allow_bonds_between_rings,
    allowed_ring_sizes=set(config.allowed_ring_sizes),
    max_steps=config.max_steps_per_episode,
)

# DQN Inputs and Outputs:
# input: appended action (fingerprint_length + 1) .
# Output size is (1).

agent = Agent(config.fingerprint_length + 1, 1, device)

if TENSORBOARD_LOG:
    writer = SummaryWriter(TB_LOG_PATH)

environment.initialize()

max_rewards = [-np.inf]
max_rewards_smiles = [None]
max_rewards_min_idx = max_rewards.index(min(max_rewards))

eps_threshold = 1.0
batch_losses = []

for it in range(iterations):

    steps_left = config.max_steps_per_episode - environment.num_steps_taken

    # Compute a list of all possible valid actions. (Here valid_actions stores the states after taking the possible
    # actions)
    valid_actions = list(environment.get_valid_actions())

    # Append each valid action to steps_left and store in observations.
    observations = np.vstack(
        [
            np.append(
                utils.get_fingerprint(
                    act, config.fingerprint_length, config.fingerprint_radius
                ),
                steps_left,
            )
            for act in valid_actions
        ]
    )  # (num_actions, fingerprint_length)

    observations_tensor = torch.Tensor(observations)
    # Get action through epsilon-greedy policy with the following scheduler.
    # eps_threshold = config.epsilon_end + (config.epsilon_start - config.epsilon_end) * \
    #     math.exp(-1. * it / config.epsilon_decay)

    a = agent.get_action(observations_tensor, eps_threshold)

    # Find out the new state (we store the new state in "action" here. Bit confusing but taken from original
    # implementation)
    action = valid_actions[a]
    # Take a step based on the action
    result = environment.step(action)

    action_fingerprint = np.append(
        utils.get_fingerprint(action, config.fingerprint_length, config.fingerprint_radius),
        steps_left,
    )

    next_state, reward, done = result

    # Compute number of steps left
    steps_left = config.max_steps_per_episode - environment.num_steps_taken

    # Append steps_left to the new state and store in next_state
    next_state = utils.get_fingerprint(
        next_state, config.fingerprint_length, config.fingerprint_radius
    )  # (fingerprint_length)

    action_fingerprints = np.vstack(
        [
            np.append(
                utils.get_fingerprint(
                    act, config.fingerprint_length, config.fingerprint_radius
                ),
                steps_left,
            )
            for act in environment.get_valid_actions()
        ]
    )  # (num_actions, fingerprint_length + 1)

    # Update replay buffer (state: (fingerprint_length + 1), action: _, reward: (), next_state: (num_actions,
    # fingerprint_length + 1), done: ()
    # print(action_fingerprints.shape)
    agent.replay_buffer.add(
        obs_t=action_fingerprint,  # (fingerprint_length + 1)
        action=0,  # No use
        reward=reward,
        obs_tp1=action_fingerprints,  # (num_actions, fingerprint_length + 1)
        done=float(result.terminated),
    )

    if done:
        final_reward = reward
        if episodes != 0 and TENSORBOARD_LOG and len(batch_losses) != 0:
            writer.add_scalar("episode_reward", final_reward, episodes)
            writer.add_scalar("episode_loss", np.array(batch_losses).mean(), episodes)
        if episodes != 0 and episodes % 2 == 0 and len(batch_losses) != 0:
            print(
                "reward of final molecule at episode {} is {}".format(
                    episodes, final_reward
                )
            )
            print(
                "mean loss in episode {} is {}".format(
                    episodes, np.array(batch_losses).mean()
                )
            )

        # The final state
        curr_state = environment.state
        # Store file
        if len(max_rewards) < 10:
            max_rewards.append(final_reward)
            max_rewards_smiles.append(curr_state)
        elif final_reward > max_rewards[max_rewards_min_idx]:
            max_rewards_min_idx = max_rewards.index(min(max_rewards))

            max_rewards.remove(max_rewards[max_rewards_min_idx])
            max_rewards_smiles.remove(max_rewards_smiles[max_rewards_min_idx])

            max_rewards.append(final_reward)
            max_rewards_smiles.append(curr_state)

        episodes += 1
        eps_threshold *= 0.99907
        batch_losses = []
        environment.initialize()

    if it % update_interval == 0 and agent.replay_buffer.__len__() >= batch_size:
        for update in range(num_updates_per_it):
            loss = agent.update_params(batch_size, config.gamma, config.polyak)
            loss = loss.item()
            batch_losses.append(loss)

outf = open("result_smiles.csv", "w")
outf.write("Reward;Smiles\n")
for smiles_text, max_reward in zip(max_rewards, max_rewards_smiles):
    outf.write(f"{smiles_text};{max_reward}\n")
outf.close()