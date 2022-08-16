import gym
import numpy as np


class ValuePolicyBase:

    def __init__(self, env):
        self.env = env
        self.observation_spaces = env.observation_space.n
        self.action_spaces = env.action_space.n

    # 策略选取
    def extract_policy(self, value_table, gamma=1.0):
        # 初始化存储策略的数组
        policy = np.zeros(self.observation_spaces)
        # 对每个状态构建Q表，并在该状态下对每个行为计算Q值: 3重循环
        for state in range(self.observation_spaces):
            # 初始化Q表
            Q_table = np.zeros(self.action_spaces)
            # 对每个动作计算
            for action in range(self.action_spaces):
                # 同上
                for next_sr in self.env.P[state][action]:
                    trans_prob, next_state, reward, done = next_sr
                    # 更新Q表，即更新动作对应的Q值（4个动作分别由0-3表示）
                    Q_table[action] += (trans_prob *
                                        (reward+gamma*value_table[next_state]))
            # 当前状态下，选取使Q值最大的那个动作
            policy[state] = np.argmax(Q_table)
        # 返回动作
        return policy

    def run_agent(self, policy, episode=10, render=True):
        # 记录到到end point的次数
        end_times = 0
        end = self.env.observation_space.n - 1

        total_rewards = []

        for i in range(episode):
            # print(f"iter={i}")

            # reset agent
            done = False
            obs = self.env.reset()
            total_reward = 0

            gamma = 1
            step_idx = 0

            while not done:
                if render:
                    self.env.render()

                obs, reward, done, info = self.env.step(int(policy[obs]))

                total_reward += (gamma ** step_idx * reward)
                step_idx += 1

            if obs == end:
                end_times += 1

            total_rewards.append(total_reward)
        self.env.close()

        return episode, end_times, end_times / episode


if __name__ == '__main__':

    env = gym.make('FrozenLake-v1', map_name="4x4", is_slippery=True)
    env.reset()
    env.render()

    # given policy
    policy = [0, 3, 3, 3, 0, 0, 0, 0, 3, 1, 0, 0, 0, 2, 1, 0]
    print(policy)
    print(ValuePolicyBase(env).run_agent(policy, episode=100000, render=False))
