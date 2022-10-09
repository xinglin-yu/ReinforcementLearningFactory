import numpy as np
from AgentAlgorithm.PolicyValueIteration.value_policy_base import ValuePolicyBase


class ValueIterationClient(ValuePolicyBase):

    def __init__(self, env, gamma=1):
        super().__init__(env, gamma)

    def value_iteration(self, max_iter=-1):
        """
        值迭代 算法
        :param max_iter:
        :return:
        """
        # 初始化状态值表（V表）
        value_table = np.zeros(self.observation_spaces)

        # 收敛判断阈值
        threshold = 1e-6
        iters = 0
        # 开始迭代: 4层循环
        while True:
            iters += 1
            # 初始化更新后的V表（旧表复制过来）
            updated_value_table = np.copy(value_table)

            # 计算每个状态下所有行为的状态动作值表（q表），最后取最大q值更新v表
            # 遍历每个状态
            for state in range(self.observation_spaces):
                # 初始化Q表
                q_table = np.zeros(self.action_spaces)

                # 遍历每个动作
                for action in range(self.action_spaces):
                    # 计算q值, 并更新q表
                    q_table[action] = self.q_value_calculate(state, action, value_table, default_q_value=-2**31)

                # 取最大Q值更新V表，即更新当前状态的V值
                value_table[state] = max(q_table)

            # 收敛判断
            if np.sum(np.fabs(updated_value_table - value_table)) <= threshold:
                print('value_iteration converges at {} rounds'.format(iters))
                break
            if iters == max_iter:
                print('value_iteration terminates at {} rounds'.format(iters))
                break

        # 返回确定性策略
        return self.policy_extract(value_table)

