# Reinforcement Learning Algorithm Implementation
We use python software and NumPy library to implement the Q-learning method to solve a Reinforcement Learning Problem. 
Specifically, we solve a autonomous-transportation problem by using Q-learning algorithm. We train our Agent and evaluate its performance by comparing two cases. 
To evaluate the Agent’s performance, we define a variable called "Score over time." "Score overtime" is defined as the sum of the rewards divided by
the total test episodes. The bigger this value is, the better the performance of our Agent.

# Q-learning Implementation In OpenAI Gym’s "Taxi-v3" Environment.

Briefly, the steps of implementing the Q-learning method in the Taxi-v3 environment in python are below.
- Step 1 Import libraries
- Step 2 Create the environment of taxi-v3
- Step 3: Create the Q-table and initialize it
- Step 4 Specify the hyper-parameters
- Step 5 Implement The Q learning algorithm and train agent
- Step 6 Evaluation


# Performance Evaluation

We are considering two cases:

- Case 1: Train the agent for 10 episodes.
then we use the Q-table to play the game, and we receive the Score over time equal to 5.1.
(in python we set, total episodes = 500 and total test episodes = 10)

- Case 2: Train the agent for 100 episodes.
then we use the Q-table to play the game, and we receive the Score over time equal to 8.27.
(in python we set , total episodes = 5000 and total test episodes = 100)

By comparing the two Score over times , we notice a significant improvement in the agent’s performance
as we increased the amount of training.

# Improvements
To further improve the performance of our agent using Q-learning there are few things that we could consider.
Tune differently α (Learning Rate) , γ (Discounted factor) , and ϵ (Exploration Rate) . We could
implement a grid search to discover the best hyper-parameters.


# Conclusion
We implemented the Q-learning algorithm; we observed how terrible our agent was when using a
small number of training episodes to play the game. The agent’s performance improved significantly
after increasing the total number of episodes in Q-learning. However, the problem with Q-learning
is that once the number of states in the environment is very high, it becomes difficult to implement
them with a Q table as the size would become very, very large. State-of-the-art techniques use Deep
neural networks instead of the Q-table (Deep Reinforcement Learning).
