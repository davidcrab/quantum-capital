import gym
import numpy as np
import pandas as pd
import yfinance as yf

class SimpleTradingEnvironment(gym.Env):
    def __init__(self, data, window_size, initial_balance=10000):
        super(SimpleTradingEnvironment, self).__init__()

        self.data = data
        self.window_size = window_size
        self.current_step = self.window_size

        self.action_space = gym.spaces.Discrete(3)  # buy, sell, hold
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(2,), dtype=np.float32)

        self.balance = initial_balance
        self.portfolio_value = initial_balance
        self.position = 0

    def _get_state(self):
        price = self.data['Close'][self.current_step]
        moving_average = self.data['Close'][self.current_step - self.window_size:self.current_step].mean()
        return np.array([price, moving_average])

    def step(self, action):
        self.current_step += 1

        if self.current_step >= len(self.data) - 1:
            done = True
        else:
            done = False

        # Calculate the reward
        prev_value = self.portfolio_value
        current_price = self.data['Close'][self.current_step]
        
        if action == 0:  # buy
            self.position += 1
            self.balance -= current_price
        elif action == 1:  # sell
            self.position -= 1
            self.balance += current_price
        
        self.portfolio_value = self.balance + self.position * current_price
        reward = self.portfolio_value - prev_value

        state = self._get_state()

        return state, reward, done, {}

    def reset(self):
        self.current_step = self.window_size
        return self._get_state()

import torch
import torch.nn as nn

class SimpleDQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(SimpleDQN, self).__init__()

        self.layers = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )

    def forward(self, x):
        return self.layers(x)


import random
from collections import deque
import torch.optim as optim

def train_agent(env, dqn, episodes, epsilon_start, epsilon_end, epsilon_decay, batch_size, buffer_size, gamma, learning_rate):
    memory = deque(maxlen=buffer_size)
    optimizer = optim.Adam(dqn.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    epsilon = epsilon_start
    total_rewards = []

    for episode in range(episodes):
        state = env.reset()
        done = False
        episode_reward = 0

        while not done:
            if random.random() < epsilon:  # Exploration
                action = env.action_space.sample()
            else:  # Exploitation
                with torch.no_grad():
                    q_values = dqn(torch.tensor(state, dtype=torch.float32))
                    action = torch.argmax(q_values).item()

            next_state, reward, done, _ = env.step(action)
            memory.append((state, action, reward, next_state, done))

            state = next_state
            episode_reward += reward

            # Training the DQN with a batch of experiences
            if len(memory) >= batch_size:
                batch = random.sample(memory, batch_size)
                states, actions, rewards, next_states, dones = zip(*batch)
                
                states = torch.tensor(states, dtype=torch.float32)
                actions = torch.tensor(actions, dtype=torch.int64).unsqueeze(-1)
                rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(-1)
                next_states = torch.tensor(next_states, dtype=torch.float32)
                dones = torch.tensor(dones, dtype=torch.bool)

                q_values = dqn(states).gather(1, actions)
                next_q_values = dqn(next_states).detach().max(1)[0].unsqueeze(-1)
                target_q_values = rewards + gamma * next_q_values * (~dones)

                loss = criterion(q_values, target_q_values)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            epsilon = max(epsilon_end, epsilon * epsilon_decay)

        total_rewards.append(episode_reward)
        print(f"Episode: {episode}, Total Reward: {episode_reward:.2f}, Epsilon: {epsilon:.3f}")

    return total_rewards


# Download stock data
ticker = 'AAPL'
start_date = '2015-01-01'
end_date = '2023-01-01'
data = yf.download(ticker, start=start_date, end=end_date)

# Replace this with your actual dataset
# data = pd.read_csv('your_dataset.csv')
data['Close'] = (data['Close'] - data['Close'].min()) / (data['Close'].max() - data['Close'].min())  # Normalize price data

env = SimpleTradingEnvironment(data, window_size=10)

# DQN and training parameters
input_dim = env.observation_space.shape[0]
output_dim = env.action_space.n
dqn = SimpleDQN(input_dim, output_dim)

# Training hyperparameters
episodes = 1000
epsilon_start = 1.0
epsilon_end = 0.01
epsilon_decay = 0.995
batch_size = 64
buffer_size = 10000
gamma = 0.99
learning_rate = 0.001

# Train the agent
total_rewards = train_agent(env, dqn, episodes, epsilon_start, epsilon_end, epsilon_decay, batch_size, buffer_size, gamma, learning_rate)

import matplotlib.pyplot as plt

plt.plot(total_rewards)
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("Training Rewards")
plt.show()
