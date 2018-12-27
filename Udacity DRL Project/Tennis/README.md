# Tennis

In this project, we work with the [Tennis](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#tennis) envirionment.

The following introductions are from Udacity. 

The observation space consists of 8 variables corresponding to the position and velocity of the ball and racket. Each agent receives its own, local observation. Two continuous actions are available, corresponding to movement toward (or away from) the net, and jumping.

The task is episodic, and in order to solve the environment, your agents must get an average score of +0.5 (over 100 consecutive episodes, after taking the maximum over both agents). Specifically,

* After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent. This yields 2 (potentially different) scores. We then take the maximum of these 2 scores.
* This yields a single score for each episode.

The environment is considered solved, when the average (over 100 episodes) of those scores is at least +0.5.

![image](https://user-images.githubusercontent.com/33606479/50485572-77812f00-09bb-11e9-8409-951041f71d02.png)

## Setting up the envirionment
The environment is provided by **Unity**. To play with this environment, please follow the described two steps:

*Step 1: Clone the DRLND Repository*

Please follow the [instructions in the DRLND GitHub repository](https://github.com/udacity/deep-reinforcement-learning#dependencies) to set up your Python environment. These instructions can be found in 'README.md' at the root of the repository. By following these instructions, you will install PyTorch, the ML-Agents toolkit, and a few more Python packages required to complete the project.

*Step 2: Download the Unity Environment*

For this project, we can use the environment Udacity build. Put the **Tennis.app.zip** (can be found in this folder and this is for Mac) in the p3_collab_compet/ folder in the DRLND GitHub repository, and unzip (or decompress) the file.

## How to use my code
Please put my code into the p3_collab_compet/ folder and then open the Jupyternote book **Report**. First, run the code under **Start the Environment**. Then, you can either use the saved weights and see the performance directly (run the code under **Use the trained agents**) or train a new agent from random initialization (run the code under **Train the agents**). Also, you can see the performance of agents with random behavior (run the code under **Take Random Actions in the Envirionment**).

You can play with the hyperparameters in the **Agent_NN.py** and change the structures of the neural network in the **NN_model.py**.
