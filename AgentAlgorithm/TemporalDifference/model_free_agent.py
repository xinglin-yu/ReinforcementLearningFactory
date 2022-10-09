import numpy as np


class ModelFreeAgent(object):
    def __init__(self, env, gamma=0.8):
        self.env = env
        self.state_length = env.observation_space.n
        self.action_length = env.action_space.n

        self.pi = np.array([0 for s in range(0, self.state_length)])
        self.value_q = np.zeros((self.state_length, self.action_length))
        self.value_n = np.zeros((self.state_length, self.action_length))
        self.gamma = gamma

    def epsilon_greedy_policy(self, state, epsilon=0):
        if np.random.rand() < epsilon:
            return np.random.randint(self.action_length)
        else:
            return self.pi[state]

    def policy_improve(self):
        new_policy = np.zeros_like(self.pi)

        for i in range(self.state_length):
            new_policy[i] = np.argmax(self.value_q[i, :])

        if np.all(np.equal(new_policy, self.pi)):
            return False
        else:
            self.pi = new_policy
            return True
