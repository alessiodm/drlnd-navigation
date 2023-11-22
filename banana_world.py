import argparse
from distutils.util import strtobool
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys
from unityagents import UnityEnvironment

from dqn import DQN
from agent import Agent

class BananaWorld:
    """A world full of bananas!
    
    This class encapsulates the Unity Banana environment provided by Udacity for the deep-learning
    nanodegree navigation project. It instantiates the Unity environment, and allows to train and
    simulate an agent via Deep Q Learning.
    """

    def __init__(self, visual=False, seed=0, checkpoint=False):
        """Initialize the banana world.

        Params:
        =======
            visual (bool): whether to run in visual mode (default False for headless)
            seed (int): seed to initialize various randoms.
            checkpoint (bool): whether to save PyTorch model and scores checkpoint files.
        """
        suffix = "" if visual else "_NoVis"
        env_file = f"unity_env/Banana_Linux{suffix}/Banana.x86_64"
        self.seed = seed
        self.env = UnityEnvironment(file_name=env_file)
        self.brain_name = self.env.brain_names[0]
        self.brain = self.env.brains[self.brain_name]
        self.action_size = self.brain.vector_action_space_size
        env_info = self.env.reset(train_mode=False)[self.brain_name]
        state = env_info.vector_observations[0]
        self.state_size = len(state)
        self.checkpoint = checkpoint
        np.random.seed(seed)
        print() # Just for nicer print considering Unity outputs.

    def simulate(self, agent: Agent):
        """Run a banana world simulation for a specific agent."""
        brain_name = self.brain_name
        env_info = self.env.reset(train_mode=False)[brain_name]   # reset the environment
        state = env_info.vector_observations[0]                   # get the current state
        score = 0                                                 # initialize the score
        while True:
            action = agent.act(state)                       # select an action with the agent
            env_info = self.env.step(action)[brain_name]    # send the action to the environment
            next_state = env_info.vector_observations[0]    # get the next state
            reward = env_info.rewards[0]                    # get the reward
            done = env_info.local_done[0]                   # see if episode has finished
            score += reward                                 # update the score
            state = next_state                              # roll over the state to next time step
            if done:                                        # exit loop if episode finished
                break
        print("Score: {}".format(score))

    def train(self, agent: Agent):
        """Train a DQN agent in the banana world."""
        print(f"Training a new agent (ddqn={agent.ddqn}, PER={agent.PER}, save={self.checkpoint})")
        dqn = DQN(self.env, agent)
        scores = dqn.train(checkpoint=self.checkpoint)
        return scores

    def new_agent(self, ddqn = True, PER = True, preload_file: str = None):
        """Create a new raw agent for the banana world.

        Shortcut and convenient method to create agents tailored to the banana world environment.
        """
        return Agent(self.state_size, self.action_size, self.seed, 
                     ddqn=ddqn, PER=PER, preload_file=preload_file)

    def close(self):
        self.env.close()


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train",
        type=lambda x: bool(strtobool(x)),
        default=False,
        nargs="?",
        const=True,
        help="if toggled, run the banana world in training mode",
    )
    parser.add_argument(
        "--simulation",
        type=str,
        default="",
        nargs="?",
        const="pretrained",
        help="if toggled, run a banana world simulation (use the suffix to select it)",
    )
    args = parser.parse_args()
    return args

# Run this file from the command line to see the default simulated agent.
if __name__ == "__main__":
    args = parse_args()

    if args.train and args.simulation:
        print("Only one of --train or --simulation[=<suffix>] can be specified.")
        sys.exit(1)

    if args.simulation:
        print("Showing the plot of the scores achieved during learning.")
        print("Close the plot window to watch the simulation of the agent.")

        # Show the plot of the scores during learning
        scores = np.loadtxt(f'scores_{args.simulation}.csv', delimiter=',', dtype=np.int16)
        avgs = pd.Series(scores).rolling(100).mean()
        x = np.arange(len(scores))
        plt.figure('Episode scores')
        plt.plot(x, scores, label='Scores')
        plt.plot(x, avgs, 'r', label='Running average')
        plt.ylabel('Score')
        plt.xlabel('Episode #')
        plt.show()

        # Simulate the pre-trained agent.
        world = BananaWorld(visual=True)
        agent = world.new_agent(preload_file=f'checkpoint_{args.simulation}.pth')
        world.simulate(agent)

    elif args.train:
        world = BananaWorld(checkpoint=True)
        agent = world.new_agent(ddqn=True, PER=True)
        scores = world.train(agent)
