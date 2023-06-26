# RLforFD
A Reinforcement Learning Environment for credit card fraud detection.
- `sb3.ipynb` is a test using [stable-baselines3](https://github.com/DLR-RM/stable-baselines3). off-policy methods work better than on-policy algos. (algos with replay buffer). 
- `test.ipynb` is a test train a self-built RL agent and gives the corresponding test results. We trained six algorithm models respectively.
- `agent.py` is the RL agent(DQN & DoubleDQN) script file.
- `env.py` is the self-built fraud detection RL training environment.  
- `train_func.py` is the training setup.
- `q_net.py` is a neural network (Q network) built using pytorch.
- `replay_buffer.py` is the replay buffer setup.
- download the dataset from https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud, and put it in `dataset`


| ![DQN_original_on_Returns](figures/DQN_original_on_Returns.png) | ![DQN_original_on_Q_value](figures/DQN_original_on_Q_value.png) |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| ![DQN_q_values_on_Returns](figures/DQN_q_values_on_Returns.png) | ![DQN_q_values_on_Q_value](figures/DQN_q_values_on_Q_value.png) |
| ![DQN_q_targets_on_Returns](figures/DQN_q_targets_on_Returns.png) | ![DQN_q_targets_on_Q_value](figures/DQN_q_targets_on_Q_value.png) |
