import random
import numpy as np
import torch
import torch.nn.functional as F
from q_net import Qnet

class CDQN:
    def __init__(self,
                 state_dim,
                 hidden_dim,
                 action_dim,
                 learning_rate,
                 gamma,
                 epsilon,
                 target_update,
                 device,
                 dqn_type='DQN',
                 q_estimate_type='q_targets'):
        self.action_dim = action_dim
        self.q_net = Qnet(state_dim, hidden_dim, self.action_dim).to(device)
        self.target_q_net = Qnet(state_dim, hidden_dim, self.action_dim).to(device)
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=learning_rate)
        self.gamma = gamma
        self.epsilon = epsilon
        self.target_update = target_update
        self.count = 0
        self.dqn_type = dqn_type
        self.device = device
        self.q_estimate_type = q_estimate_type

    def take_action(self, state):
        if np.random.random() < self.epsilon:
            action = np.random.randint(self.action_dim)
        else:
            state = torch.tensor([state], dtype=torch.float).to(self.device)
            action = self.q_net(state).argmax().item()
        return action

    def max_q_value(self, state):
        state = torch.tensor([state], dtype=torch.float).to(self.device)
        return self.q_net(state).max().item()
    
    def dr_estimate(self, state, action, reward, next_state):
        # First calculate the Q value estimated based on the model
        q_value = self.q_net(state).gather(1, action)
        q_value_next = self.q_net(next_state).detach().max(1)[0].unsqueeze(1)

        # Computes an estimator based on a target strategy
        action_probs = self.target_q_net(next_state).detach()

        log_action_prob = F.log_softmax(action_probs, dim=1)

        log_chose_action_prob = log_action_prob.gather(1, action.reshape((-1, 1)).long())
        log_chose_action_prob = log_chose_action_prob.squeeze()

        q_estimate_next = (
            action_probs.exp() * self.q_net(next_state).detach()
        ).sum(dim=1, keepdim=True)

        # Calculate the DR Estimator
        pi = torch.exp(log_chose_action_prob)
        dr_estimates = ((q_value - q_estimate_next + reward) / pi) * \
                        pi.detach() + (1 - pi.detach()) * q_estimate_next
        # dr_estimates = dr_estimates.gather(1, action)
        return dr_estimates
    
    def update(self, transition_dict):
        states = torch.tensor(transition_dict['states'], dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions']).view(-1, 1).to(self.device)
        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'], dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'], dtype=torch.float).view(-1, 1).to(self.device)
        
        # states = torch.FloatTensor(np.float32(transition_dict['states'])).to(self.device)
        # actions = torch.LongTensor(transition_dict['actions']).unsqueeze(1).to(self.device)
        # rewards = torch.FloatTensor(transition_dict['rewards']).unsqueeze(1).to(self.device)
        # next_states = torch.FloatTensor(np.float32(transition_dict['next_states'])).to(self.device)
        # dones = torch.tensor(transition_dict['dones'], dtype=torch.float).view(-1, 1).to(self.device)
        
        # Calculate the DR Estimator
        dr_estimates = self.dr_estimate(states, actions, rewards, next_states)
        
        if self.q_estimate_type == 'original': # Original version
            q_values = self.q_net(states).gather(1, actions)  # Q value
            # The maximum Q value of the next state
            if self.dqn_type == 'DoubleDQN': # Difference between DQN and Double DQN
                max_action = self.q_net(next_states).max(1)[1].view(-1, 1)
                max_next_q_values = self.target_q_net(next_states).gather(1, max_action)
            elif self.dqn_type == 'DQN': # The DQN situation
                max_next_q_values = self.target_q_net(next_states).max(1)[0].view(-1, 1)
            q_targets = rewards + self.gamma * max_next_q_values * (1 - dones) # TD error target
            loss = torch.mean(F.mse_loss(q_values, q_targets)) # Calculate the MSE loss estimated by Q value
        elif self.q_estimate_type == 'q_values': # Estimate of q_values
            q_values = dr_estimates.gather(1, actions)
            if self.dqn_type == 'DoubleDQN':
                max_action = self.q_net(next_states).max(1)[1].view(-1, 1)
                max_next_q_values = self.target_q_net(next_states).gather(1, max_action)
            elif self.dqn_type == 'DQN':
                max_next_q_values = self.target_q_net(next_states).max(1)[0].view(-1, 1)
            q_targets = rewards + self.gamma * max_next_q_values * (1 - dones)
            loss = torch.mean(F.mse_loss(q_values, q_targets))
        elif self.q_estimate_type == 'q_targets': # Estimate of q_targets
            q_values = self.q_net(states).gather(1, actions)
            if self.dqn_type == 'DoubleDQN':
                max_action = self.q_net(next_states).max(1)[1].view(-1, 1) 
                max_next_q_values = dr_estimates.gather(1, max_action)
            elif self.dqn_type == 'DQN':
                max_next_q_values = dr_estimates.max(1)[0].view(-1, 1)
            q_targets = rewards + self.gamma * max_next_q_values * (1 - dones)
            loss = torch.mean(F.mse_loss(q_values, q_targets))
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        if self.count % self.target_update == 0:
            self.target_q_net.load_state_dict(self.q_net.state_dict())  # Update target network
        self.count += 1