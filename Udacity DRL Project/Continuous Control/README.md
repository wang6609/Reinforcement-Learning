# Continuous Control

In this project, we work with the [Reacher](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#reacher) envirionment.

![1](https://user-images.githubusercontent.com/33606479/48993513-9910a180-f103-11e8-82e3-949d40c32045.gif)

The following environment description is from Udacity. 

In this environment, a double-jointed arm can move to target locations. A reward of +0.1 is provided for each step that the agent's hand is in the goal location. Thus, the goal of your agent is to maintain its position at the target location for as many time steps as possible.

The observation space consists of 33 variables corresponding to position, rotation, velocity, and angular velocities of the arm. Each action is a vector with four numbers, corresponding to torque applicable to two joints. Every entry in the action vector should be a number between -1 and 1.

## Distributed Training

For this project, we will provide you with two separate versions of the Unity environment:

**The first version** contains a single agent.
The **second version** contains 20 identical agents, each with its own copy of the environment.
The second version is useful for algorithms like [PPO](https://arxiv.org/pdf/1707.06347.pdf), [A3C](https://arxiv.org/pdf/1602.01783.pdf), and [D4PG](https://openreview.net/pdf?id=SyZipzbCb) that use multiple (non-interacting, parallel) copies of the same agent to distribute the task of gathering experience.

Option 1: Solve the First Version
The task is episodic, and in order to solve the environment, your agent must get an average score of +30 over 100 consecutive episodes.

Option 2: Solve the Second Version
The barrier for solving the second version of the environment is slightly different, to take into account the presence of many agents. In particular, your agents must get an average score of +30 (over 100 consecutive episodes, and over all agents). Specifically,

After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent. This yields 20 (potentially different) scores. We then take the average of these 20 scores.
This yields an average score for each episode (where the average is over all 20 agents).

## Setting up the envirionment
The environment is provided by **Unity**. To play with this environment, please follow the described two steps:

*Step 1: Clone the DRLND Repository*

Please follow the [instructions in the DRLND GitHub repository](https://github.com/udacity/deep-reinforcement-learning#dependencies) to set up your Python environment. These instructions can be found in 'README.md' at the root of the repository. By following these instructions, you will install PyTorch, the ML-Agents toolkit, and a few more Python packages required to complete the project.

*Step 2: Download the Unity Environment*

For this project, we can use the environment Udacity build. Put the **Reacher.app.zip** (can be found in this folder and this is for Mac) in the p2_continuous_control/ folder in the DRLND GitHub repository, and unzip (or decompress) the file.

## How to use my code
Please put my code into the p2_continuous_control/ folder and then open the Jupyternote book **Report**. First, run the code under **Start the Environment**. Then, you can either use the saved weights and see the performance directly (run the code under **Use the trained agent**) or train a new agent from random initialization (run the code under **Train an agent by DDPG**). Also, you can see the performance of an agent with random behavior (run the code under **Take Random Actions in the Envirionment**).

You can play with the hyperparameters in the **Agent_NN.py** and change the structures of the neural network in the **NN_model.py**.
