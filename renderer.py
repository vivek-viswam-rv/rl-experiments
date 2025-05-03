import os
import time
from abc import ABC, abstractmethod
from enum import Enum
from gymnasium import spaces
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque

from gymnasium.wrappers import RecordVideo

class PPONet(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(PPONet, self).__init__()
        self.shared = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()
        )
        self.actor = nn.Linear(128, action_dim)
        self.critic = nn.Linear(128, 1)

    def forward(self, x):
        x = self.shared(x)
        logits = self.actor(x)
        value = self.critic(x)
        return logits, value

class PPOAgent:
    BATCH_SIZE = 32
    CLIP_PARAM = 0.2
    GAMMA = 0.99
    LR = 1e-4
    GAE_LAMBDA = 0.95
    PPO_EPOCHS = 10
    VALUE_LOSS_COEF = 0.5
    ENTROPY_COEF = 0.01

    def __init__(self, state_dim, action_dim):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net = PPONet(state_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.LR)
        self.memory = []
        self.clip = self.CLIP_PARAM

    def store_transition(self, state, action, reward, next_state, done, log_prob, value):
        self.memory.append((state, action, reward, next_state, done, log_prob, value))

    def select_action(self, state):
        state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        logits, value = self.policy_net(state_tensor)
        probs = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)
        action = torch.multinomial(probs, 1).item()
        log_prob = log_probs[0, action].item()

        return action, log_prob, value.item()

    def compute_gae(self, rewards, values, next_value, dones):
        advantages = []
        gae = 0
        values = values + [next_value]
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.GAMMA * values[t + 1] * (1 - dones[t]) - values[t]
            gae = delta + self.GAMMA * self.GAE_LAMBDA * (1 - dones[t]) * gae
            advantages.insert(0, gae)
        advantages = torch.tensor(advantages, dtype=torch.float32, device=self.device)
        returns = advantages + torch.tensor(values[:-1], dtype=torch.float32, device=self.device)

        return advantages, returns

    def step(self):
        if len(self.memory) < self.BATCH_SIZE:
            return

        states, actions, rewards, next_states, dones, old_log_probs, old_values = zip(*self.memory)
        states = torch.tensor(np.array(states), dtype=torch.float32, device=self.device)
        actions = torch.tensor(actions, dtype=torch.int64, device=self.device)
        old_log_probs = torch.tensor(old_log_probs, dtype=torch.float32, device=self.device)
        old_values = torch.tensor(old_values, dtype=torch.float32, device=self.device)

        with torch.no_grad():
            _, next_value = self.policy_net(states[-1:])
        advantages, returns = self.compute_gae(rewards, old_values.tolist(), next_value.item(), dones)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        for _ in range(self.PPO_EPOCHS):
            logits, values = self.policy_net(states)
            probs = F.softmax(logits, dim=-1)
            log_probs = F.log_softmax(logits, dim=-1)
            selected_log_probs = log_probs[range(len(actions)), actions]

            ratios = torch.exp(selected_log_probs - old_log_probs)
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.clip, 1 + self.clip) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()

            value_loss = F.mse_loss(values.squeeze(), returns)
            entropy = -(probs * log_probs).sum(dim=-1).mean()

            loss = policy_loss + self.VALUE_LOSS_COEF * value_loss - self.ENTROPY_COEF * entropy

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        self.memory.clear()

def load_model(model_path, env):
    # Initialize the DQN model
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    model = PPOAgent(state_dim, action_dim)  # Adjust parameters as per your DQNclass
    # Load the saved state dictionary
    model.policy_net.load_state_dict(torch.load(model_path)) # Set to evaluation mode
    return model

def render_lunar_lander(model_path):
    # Initialize the Lunar Lander environment with rendering
    env = gym.make('Acrobot-v1', render_mode='rgb_array')
    env = RecordVideo(env, video_folder="C:\\Users\\viswa\Projects\\rl_final_project",  name_prefix = "ppo_acrobot", episode_trigger=lambda x: True)

    # Load the pre-trained DQN model
    model = load_model(model_path, env)

    # Reset the environment
    state, _ = env.reset()
    state = np.array(state, dtype=np.float32)  # Add batch dimension

    done = False
    total_reward = 0

    # Run one episode
    while not done:
        # Select action using the DQN model (assuming model has an act/get_action method)
        with torch.no_grad():
            action, _, _ = model.select_action(state)  # Replace with your DQNclass's action selection method

        # Step the environment
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        total_reward += reward

        # Update state
        state = np.array(next_state, dtype=np.float32)

    print(f"Episode finished with total reward: {total_reward}")
    env.close()


if __name__ == "__main__":
    # Path to your .pth file
    model_path = "ppo_acrobot_v1.pth"  # Replace with the actual path to your .pth file
    render_lunar_lander(model_path)
