from collections import deque
import numpy as np
from typing import List
from unityagents import UnityEnvironment

from agent import Agent

class DQN:
    """Deep Q-Learning implementation.
    
    This class encapsulates the general training loop for a DQN agent in a Unity
    environment. Most of the algorithmic logic is contained in the agent itself.
    """

    def __init__(self, env: UnityEnvironment, agent: Agent, solved_score = 13.):
        """Initialize the DQN for training.

        Params:
        =======
          env (Environment): Unity Environment
          agent (Agent):  DQN agent
          soved_score (float): the average score to consider the environment solved
        """
        self.env = env
        self.agent = agent
        self.brain_name: str = env.brain_names[0]
        self.solved_score = solved_score

    def train(self, n_episodes=1500, max_t=1000,
              eps_start=1.0, eps_end=0.01, eps_decay=0.995,
              checkpoint=False) -> List[int]:
        """Deep Q-Learning general training loop.

        Params
        ======
            n_episodes (int): maximum number of training episodes
            max_t (int): maximum number of timesteps per episode
            eps_start (float): starting value of epsilon, for epsilon-greedy action selection
            eps_end (float): minimum value of epsilon
            eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
        
        Returns
        =======
            scores (List[int]): the scores for each training episode.
        """
        scores = []                        # list containing scores from each episode
        scores_window = deque(maxlen=100)  # last 100 scores
        eps = eps_start                    # initialize epsilon
        solved = False                     # flag marking the environment solved
        for i_episode in range(1, n_episodes+1):
            score = 0                      # initialize the score
            state = self.__env_reset()     # get the initial state for the episode
            for t in range(max_t):
                action = self.agent.act(state, eps)
                next_state, reward, done = self.__env_step(action)
                self.agent.step(state, action, reward, next_state, done)
                state = next_state
                score += reward
                if done:
                    break 
            eps = max(eps_end, eps_decay*eps)   # decrease epsilon
            scores.append(score)                # save most recent score
            scores_window.append(score)         # update window for rolling average
            avg_score = np.mean(scores_window)  # compute rolling average score
            # Just logging below :)
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, avg_score), end="")
            if i_episode % 100 == 0:
                print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, avg_score))
            if not solved and avg_score >= self.solved_score:
                print(f'\rEnvironment solved at episode {i_episode} with score {avg_score:.2f}!')
                solved = True
        # checkpoint the agent at the end of training, also save the scores for plotting.
        if checkpoint:
            self.agent.checkpoint()
            np.savetxt("scores.csv", np.asarray(scores, dtype=np.int16), delimiter=",")
        return scores

    def __env_reset(self):
        """Reset the environment for a new training episode."""
        env_info = self.env.reset(train_mode=True)[self.brain_name]   # reset the environment
        state = env_info.vector_observations[0]                       # get the current state
        return state

    def __env_step(self, action):
        """Shortcut method to take an action / step in the Unity environment."""
        env_info = self.env.step(action)[self.brain_name]   # send the action to the environment
        next_state = env_info.vector_observations[0]        # get the next state
        reward = env_info.rewards[0]                        # get the reward
        done = env_info.local_done[0]                       # see if episode has finished
        return next_state, reward, done
