import numpy as np


class ValuePolicyBase:

    def __init__(self, env, gamma=1):
        self.env = env
        self.observation_spaces = env.observation_space.n
        self.action_spaces = env.action_space.n

        # 衰减率
        self.gamma = gamma

    def policy_extract(self, value_table):
        """
        策略提取: 固定状态值函数 => 求最优策略
        :param value_table: 状态值函数
        :return: 策略list
        """
        # 初始化存储策略的数组
        policy = np.zeros(self.observation_spaces, dtype=int)

        # 对每个状态构建Q表，并在该状态下对每个行为计算Q值: 3重循环
        for state in range(self.observation_spaces):
            # 初始化Q表
            q_table = np.zeros(self.action_spaces)
            # 对每个动作计算
            for action in range(self.action_spaces):
                for next_sr in self.env.P[state][action]:
                    trans_prob, next_state, reward, done = next_sr
                    # 更新Q表，即更新动作对应的Q值（4个动作分别由0-3表示）
                    q_table[action] += (trans_prob * (reward + self.gamma * value_table[next_state]))
            # 当前状态下，选取使Q值最大的那个动作
            policy[state] = int(np.argmax(q_table))
        return policy
