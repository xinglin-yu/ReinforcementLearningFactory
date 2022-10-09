import numpy as np


class QLearning(object):
    def __init__(self, epsilon=0.0):
        self.epsilon = epsilon

    def q_learn_eval(self, agent, env):
        state = env.reset()
        while True:
            # choose action
            action = agent.epsilon_greedy_policy(state, self.epsilon)
            next_state, reward, terminate, _ = env.step(action)

            # off policy
            # update q table
            agent.value_n[state][action] += 1
            agent.value_q[state][action] += (reward + agent.gamma * np.max(agent.value_q[next_state, :]) -
                                             agent.value_q[state][action]) / \
                                            agent.value_n[state][action]

            if terminate:
                break
            state = next_state

    def iteration(self, agent, env):
        for i in range(10):
            for j in range(3000):
                self.q_learn_eval(agent, env)
            agent.policy_improve()

        return agent.pi
