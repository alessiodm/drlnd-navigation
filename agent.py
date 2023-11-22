import numpy as np
import random

from model import QNetwork
from replay_buffer import ReplayBuffer

import torch
import torch.nn.functional as F
import torch.optim as optim

BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 64         # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR = 5e-4               # learning rate
UPDATE_EVERY = 4        # how often to update the network for vanilla fixed Q targets
UPDATE_EVERY_DDQN = 8   # how often to update the network for double DQN and PER

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent:
    """Agent that interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, seed,
                 ddqn=False, PER=False, preload_file: str = None):
        """Initialize an Agent object.

        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
            double_dqn (bool): if True, use a Double DQN update rule
            per (bool): if True, use a Prioritized Experience Replay buffer
            checkpoint_file (str): if set, loads the weight from the file
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)
        self.ddqn = ddqn
        self.PER = PER

        # Q-Network
        self.qnetwork_local = QNetwork(state_size, action_size, seed).to(device)
        self.qnetwork_target = QNetwork(state_size, action_size, seed).to(device)
        self.qnetwork_target.eval()  # Only eval mode on the target network
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)

        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, PER)
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0
        # For Double DQN and Prioritized Experience Replay we use a different value.
        self.update_every = UPDATE_EVERY_DDQN if self.ddqn or self.PER else UPDATE_EVERY

        if preload_file is not None:
            print(f'Loading pre-trained model: {preload_file}')
            self.qnetwork_local.load_state_dict(torch.load(preload_file, map_location=device))

    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)

        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % self.update_every
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > BATCH_SIZE:
                self.__learn(GAMMA)

    def act(self, state, eps=0.):
        """Returns actions for given state as per current policy.
        
        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def checkpoint(self):
        """Save the QNetwork weights in the 'checkpoint.pth' file."""
        torch.save(self.qnetwork_local.state_dict(), 'checkpoint.pth')

    def __learn(self, gamma):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            gamma (float): discount factor
        """
        experiences_selection, experiences, is_factor = self.memory.sample()
        states, actions, rewards, next_states, dones = experiences

        # Get the predicted action values of the *NEXT* states from the target model.
        target_action_values = self.qnetwork_target(next_states).detach()  # (batch_size, action_size)

        # Selct Double-DQN or vanilla Fixed Q-Targets update rule.
        if self.ddqn:
            self.qnetwork_local.eval()
            # max_action_values needs to be picked on the target nework by the local network choice.
            local_action_values = self.qnetwork_local(next_states).detach()
            local_max_action_values_indices = local_action_values.max(1)[1].unsqueeze(1)
            max_action_values = target_action_values.gather(1, local_max_action_values_indices)
            self.qnetwork_local.train()
        else:
            # Select the max action value for each state:
            #   https://pytorch.org/docs/stable/generated/torch.amax.html
            max_action_values = target_action_values.amax(1, keepdim=True)  # (batch_size, 1)

        # Then, compute the Q _targets_ for the current states.
        Q_targets = rewards + (gamma * max_action_values * (1 - dones)) # (batch_size, 1)

        # Get expected Q values from local model.
        predictions = self.qnetwork_local(states)
        # We choose only the action value that was selected in the experience replay.
        Q_expected = predictions.gather(1, actions)

        # Compute the TD error.
        td_error = Q_targets - Q_expected

        # If Prioritized Experience Replay is enabled, update the buffer with new priorities.
        if self.PER:
            with torch.no_grad():
                self.memory.update_p(experiences_selection, td_error)
        is_factor = torch.tensor(is_factor)

        # Compute loss considering the scale_factor for importance sampling.
        scaled_squared_error = is_factor * td_error * td_error
        loss = torch.sum(scaled_squared_error) / torch.numel(scaled_squared_error)

        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        self.__soft_update(self.qnetwork_local, self.qnetwork_target, TAU)                     

    def __soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)
