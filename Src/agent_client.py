import time

import gym
import numpy as np


class AgentClient:

    def __init__(self, env):
        self.env = env
        self.observation_spaces = env.observation_space.n
        self.action_spaces = env.action_space.n

    def run_agent(self, policy, episode=10, render=True):
        # 记录到到end point的次数
        end_times = 0
        end = self.env.observation_space.n - 1

        total_rewards = []

        for i in range(episode):
            # reset agent
            done = False
            obs = self.env.reset()
            if render:
                print(f"iter={i}")
                self.env.render()

            total_reward = 0

            gamma = 1
            step_idx = 0

            while not done:
                obs, reward, done, info = self.env.step(int(policy[obs]))

                if render:
                    print("       step_idx=", step_idx)
                    self.env.render()

                total_reward += (gamma ** step_idx * reward)
                step_idx += 1

            if obs == end:
                # time.sleep(1)
                end_times += 1

            total_rewards.append(total_reward)
        self.env.close()

        return episode, end_times, end_times / episode

    def calcute_possibility(self, policy):

        # 转移概率矩阵
        possibility_matrix = np.zeros(shape=(self.observation_spaces, self.observation_spaces))

        # 遍历每个状态
        for state in range(self.observation_spaces):
            # 根据策略取动作
            action = policy[state]
            for next_sr in self.env.P[state][action]:
                trans_prob, next_state, reward, done = next_sr
                possibility_matrix[int(next_state), state] += trans_prob

        # 初始状态向量
        state_vector = np.zeros(shape=(self.observation_spaces, 1))
        state_vector[0] = 1

        # 开始迭代, 至收敛
        threshold = 1e-10  # 收敛判断阈值
        err = threshold + 1  # 误差
        times = 0  # 迭代次数
        # 循环直到收敛
        while err > threshold:
            state_vector_next = np.dot(possibility_matrix, state_vector)
            # 计算误差
            err = np.linalg.norm(np.subtract(state_vector_next, state_vector), ord=2)
            times += 1
            state_vector = state_vector_next

        return times, possibility_matrix, state_vector


if __name__ == '__main__':

    env = gym.make('FrozenLake-v1', map_name="4x4", is_slippery=True)
    env.reset()
    env.render()

    # given policy
    policy = [0, 3, 3, 3, 0, 0, 0, 0, 3, 1, 0, 0, 0, 2, 1, 0]
    print(policy)

    obj = AgentClient(env)
    times, possibility_matrix, state_vector = obj.calcute_possibility(policy)
    print("times=", times)
    print("state_vector=", state_vector)

    print(obj.run_agent(policy, episode=10000, render=False))
    # print(obj.run_agent(policy, episode=10, render=True))
