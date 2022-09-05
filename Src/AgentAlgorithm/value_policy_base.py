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
