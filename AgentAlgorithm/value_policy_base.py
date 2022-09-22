import numpy as np


class ValuePolicyBase:

    def __init__(self, env, gamma=1):
        self.env = env
        self.observation_spaces = env.observation_space.n
        self.action_spaces = env.action_space.n

        # 衰减率
        self.gamma = gamma

    def q_value_calculate(self, state, action, value_table, default_q_value=0):
        """
        q值计算
        :param state: 当前状态
        :param action: 当前动作
        :param value_table: 状态值函数表
        :param default_q_value: 默认值
        :return: int. 即 q值: q(state, action)
        """
        # 不在env可以转移的位置, 便直接返回默认值
        # 对于reward始终为正的, 可以设置default_value=0;
        # 对于reward存在负数的情况, 可以设置default_value=-2**31
        # 目的: (1)不影响value_table的收敛判断; (2)不影响取max和argmax
        if action not in self.env.P[state]:
            return default_q_value

        q_value = 0
        for next_sr in self.env.P[state][action]:
            # P[][]是环境定义的变量,存储状态s下采取动作a得到的元组数据（转移概率，下一步状态，奖励，完成标志）
            trans_prob, next_state, reward, done = next_sr

            # 更新q表，即更新动作对应的q值
            q_value += (trans_prob * (reward + self.gamma * value_table[next_state]))

        return q_value

    def policy_extract(self, value_table):
        """
        策略提取: 固定状态值函数 => 求最优策略
        :param value_table: 状态值函数
        :return: 策略list
        """
        # 初始化存储策略的数组
        policy = np.zeros(self.observation_spaces, dtype=int)

        # 对每个状态构建Q表，并在该状态下对每个行为计算Q值
        for state in range(self.observation_spaces):
            # 初始化Q表
            q_table = np.zeros(self.action_spaces)

            # 对每个动作计算
            for action in range(self.action_spaces):
                # 计算q值, 并更新q表
                q_table[action] = self.q_value_calculate(state, action, value_table, default_q_value=-2**31)

            # 当前状态下，选取使Q值最大的那个动作
            policy[state] = int(np.argmax(q_table))
        return policy
