Stochastic Policy Gradient Methods

For a detailed discussion, visit : https://sridhartee.blogspot.in/2016/11/policy-gradient-methods.html
Environment Simulator Used : OpenAI Gym.

![cartpole-actorcritic](https://user-images.githubusercontent.com/25884784/60865345-b568d680-a1da-11e9-9299-b55ee0d6bcee.png)


We design and test 3 policy gradient methods in this repository

1) Monte Carlo Policy Gradient : Baseline used is average of rewards obtained, no baseline results in high variance

2) Actor Critic Method : Using Softmax policy and Q-learning Critic for value function estimation

3) Numerical Gradient Estimation : perturb the parameters and estimate the gradient using regression (X'X)^-1X'y. 
Change num_rollouts to change the number of training examples we learn the gradient from.
Note that the actual number of runs is number of episodes*num_rollouts
