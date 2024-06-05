import torch
import numpy as np
import torch.optim as opt
from model import DQN
from rdkit import Chem
from rdkit.Chem import QED
from environment import Molecule
from replay_buffer import ReplayBuffer


class QEDRewardMolecule(Molecule):
    """
    QED optimizer environment
    """

    def __init__(self, discount_factor, **kwargs):
        """
        Initializes the class.
        :param discount_factor: Float. The discount factor
        :param kwargs: The keywords passed to the base class
        """
        super(QEDRewardMolecule, self).__init__(**kwargs)
        self.discount_factor = discount_factor

    def _reward(self):
        """
        Reward of the state.
        :return: Float. QED score of the current state
        """

        molecule = Chem.MolFromSmiles(self._state)
        if molecule is None:
            return 0.0
        qed = QED.qed(molecule)
        return qed * self.discount_factor ** (self.max_steps - self.num_steps_taken)


class Agent(object):
    """
    The Agent to manipulate the molecule. It includes a DQN to decide actions.
    Uses replay buffer. One DQN to learn, the other one to decide (frozen weights).
    """

    def __init__(self, input_length, output_length, device, config, pretrained=False):

        self.fingerprint_length = config.fingerprint_length
        self.learning_rate = config.learning_rate
        self.device = device

        self.dqn, self.target_dqn = (
            DQN(input_length, output_length).to(self.device),
            DQN(input_length, output_length).to(self.device),
        )

        if pretrained:
            self.dqn.load_state_dict(torch.load(config.model_path))

        for p in self.target_dqn.parameters():
            p.requires_grad = False
        # self.replay_buffer = ReplayBuffer(REPLAY_BUFFER_CAPACITY)
        self.replay_buffer = ReplayBuffer(config.replay_buffer_size)
        self.optimizer = getattr(opt, 'Adam')(
            self.dqn.parameters(), lr=self.learning_rate
        )

    def get_action(self, observations, epsilon_threshold):
        """
        Chooses an action from given observation. Returns the decided action
        :param observations: torch.Tensor. The observation state
        :param epsilon_threshold: Float. A threshold value to decide between exploration and exploitation.
        :return: int. Action index
        """
        if np.random.uniform() < epsilon_threshold:
            action = np.random.randint(0, observations.shape[0])
        else:
            q_value = self.dqn.forward(observations.to(self.device)).cpu()
            action = torch.argmax(q_value).numpy()
        return action

    def update_params(self, batch_size, gamma, polyak):
        """
        Updates the parameters of the model. Learning is done here.
        :param batch_size: int. The batch size
        :param gamma: Float. Gamma value for DQN
        :param polyak:
        :return: Loss value
        """

        states, _, rewards, next_states, dones = self.replay_buffer.sample(batch_size)
        q_t = torch.zeros(batch_size, 1, requires_grad=False)
        v_tp1 = torch.zeros(batch_size, 1, requires_grad=False)
        for i in range(batch_size):
            state = (
                torch.FloatTensor(states[i])
                .reshape(-1, self.fingerprint_length + 1)
                .to(self.device)
            )
            q_t[i] = self.dqn(state)

            next_state = (
                torch.FloatTensor(next_states[i])
                .reshape(-1, self.fingerprint_length + 1)
                .to(self.device)
            )
            v_tp1[i] = torch.max(self.target_dqn(next_state))

        rewards = torch.FloatTensor(rewards).reshape(q_t.shape).to(self.device)
        q_t = q_t.to(self.device)
        v_tp1 = v_tp1.to(self.device)
        dones = torch.FloatTensor(dones).reshape(q_t.shape).to(self.device)

        q_tp1_masked = (1 - dones) * v_tp1
        q_t_target = rewards + gamma * q_tp1_masked
        td_error = q_t - q_t_target

        q_loss = torch.where(
            torch.abs(td_error) < 1.0,
            0.5 * td_error * td_error,
            1.0 * (torch.abs(td_error) - 0.5),
        )
        q_loss = q_loss.mean()

        self.optimizer.zero_grad()
        q_loss.backward()
        self.optimizer.step()

        with torch.no_grad():
            for p, p_targ in zip(self.dqn.parameters(), self.target_dqn.parameters()):
                p_targ.data.mul_(polyak)
                p_targ.data.add_((1 - polyak) * p.data)

        return q_loss
