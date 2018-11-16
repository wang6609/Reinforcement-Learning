# Navigation
## Deep Reinforcement Learning
Using Reinforcement Learning to solve a navigation problem. **Python3** and **PyTorch** is used. This is the first project in the Deep Reinforcement Learning Nanodegree from Udacity.

The following game environment description and programming envirionment setting is provided by Udacity.

For this project, you will train an agent to navigate (and collect bananas!) in a large, square world.

A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana. Thus, the goal of your agent is to collect as many yellow bananas as possible while avoiding blue bananas.

The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around the agent's forward direction. Given this information, the agent has to learn how to best select actions. Four discrete actions are available, corresponding to:

* 0 - move forward
* 1 - move backward
* 2 - turn left
* 3 - turn right

The task is episodic, and in order to solve the environment, your agent must get an average score of +13 over 100 consecutive episodes.

![banana](https://user-images.githubusercontent.com/33606479/48532564-8924d200-e866-11e8-96eb-b006a2505800.gif)

The environment is provided by **Unity**. To play with this environment, please follow the described two steps:

*Step 1: Clone the DRLND Repository*

Please follow the [instructions in the DRLND GitHub repository](https://github.com/udacity/deep-reinforcement-learning#dependencies) to set up your Python environment. These instructions can be found in 'README.md' at the root of the repository. By following these instructions, you will install PyTorch, the ML-Agents toolkit, and a few more Python packages required to complete the project.

*Step 2: Download the Unity Environment*

For this project, we can use the environment Udacity build. Put the **Banana.app.zip** (can be found in this folder and this is for Mac) in the p1_navigation/ folder in the DRLND GitHub repository, and unzip (or decompress) the file.

In this video, the agent I trained get a score of +17.
https://www.youtube.com/watch?v=0dpAEdseCRQ
