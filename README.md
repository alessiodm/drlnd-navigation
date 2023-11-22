# Banana World!

Navigation project for the Udacity deep reinforcement learning nanodegree. This repository is
based on the [Value-based-methods](https://github.com/udacity/Value-based-methods) Udacity
repository project.

<video width="320" height="180" autoplay loop controls>
  <source src="pretrained.mp4" type="video/mp4">
</video>

## Getting Started

The instructions below are intended only for Linux environments.

### Python Virtual Environment

[Install Anaconda](https://docs.anaconda.com/free/anaconda/install/linux/), create a new Python
`3.6` virtual environment, and activate it:

```bash
$> conda create --name drlnd python=3.6
$> conda activate drlnd
```

### Install Dependencies

Enter the `python` folder, and install the dependencies:

```bash
(drlnd) $> cd python
(drlnd) $> pip install .
```

Note that Udacity provides a custom setup of Unity ML-Agents. Also, some dependencies have been
updated from the `Value-based-methods` repository because obsolete, notably PyTorch is at version
`1.10.0`.

### Download the Unity Environment

Download the `Banana` Unity environment in the `unity_env` folder. Follow the instructions in the
`unity_env/README.md` file to do that.

### Additional Notes

The configuration used to run this code is Dell XPS-13-9310 with Linux Mint 21.2. MESA rendering
issues with Anaconda have been encountered (see [here](https://askubuntu.com/a/1405450), and
[here](https://stackoverflow.com/questions/71263856/kivy-not-working-mesa-loader-failed-to-open-iris-and-swrast)).
If the aforementioned instructions don't work, it is likely a custom environment issue that needs
to be troubleshooted ad-hoc.

## Instructions

To watch the pretrained agent (Double-DQN + Prioritized Experience Replay), run:

```bash
(drlnd) $> python banana_world.py --simulation
```

You'll see first the scores plot, and then the Unity simulation once the plot window is closed.

To train a new agent instead, run:

```bash
(drlnd) $> python banana_world.py --train
```

Two new files `scores.csv` and `checkpoint.pth` are created. Add a `_<NAME>` suffix to run your
simulation via:

```bash
(drlnd) $> python banana_world.py --simulation=<NAME>
```

## Environment Details

A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for
collecting a blue banana.  Thus, the goal of your agent is to collect as many yellow bananas as
possible while avoiding blue bananas.  

The state space has 37 dimensions and contains the agent's velocity, along with ray-based
perception of objects around agent's forward direction.  Given this information, the agent has to
learn how to best select actions.  Four discrete actions are available, corresponding to:

- **`0`** - move forward.
- **`1`** - move backward.
- **`2`** - turn left.
- **`3`** - turn right.

### When Is The Environment Solved?

The task is episodic, and in order to solve the environment, your agent must get an average score
of +13 over 100 consecutive episodes.
