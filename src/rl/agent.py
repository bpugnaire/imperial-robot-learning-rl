# rl/agent.py

import numpy as np
import random

class QLearningAgent:
    def __init__(self, state_bins, num_actions, alpha=0.1, gamma=0.9, epsilon=0.1):
        """
        state_bins: list of bin edges to discretize the lid angle
        num_actions: number of discrete actions
        """
        self.state_bins = state_bins
        self.num_actions = num_actions
        self.q_table = np.zeros((len(state_bins) + 1, num_actions))

        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

    def discretize_state(self, angle):
        """Convert continuous angle into discrete state"""
        print("supposed to be an angle" , angle)
        return np.digitize(angle, self.state_bins)

    def choose_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.num_actions - 1)
        return np.argmax(self.q_table[state])

    def update(self, state, action, reward, next_state):
        predict = self.q_table[state, action]
        target = reward + self.gamma * np.max(self.q_table[next_state])
        self.q_table[state, action] += self.alpha * (target - predict)
