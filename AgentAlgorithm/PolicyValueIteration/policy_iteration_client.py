import numpy as np
from CommonUtils.decorator import timer
from AgentAlgorithm.PolicyValueIteration.value_policy_base import ValuePolicyBase


class PolicyIterationClient(ValuePolicyBase):

    def __init__(self, env, gamma=1):
        super().__init__(env, gamma)

    def policy_evaluation(self, policy, max_iter=-1):
        """
        策略评估: 固定策略 => 求最优状态值函数
        :param policy: 策略
        :param max_iter: 最大迭代步数
        :return: value_table
        """
        # 初始化V表
        value_table = np.zeros(self.observation_spaces, dtype=float)

        # 收敛判断阈值
        threshold = 1e-6
        iters = 0
        while True:  # 循环直到收敛
            iters += 1

            # 初始化更新后的V表（旧表复制过来）
            updated_value_table = np.copy(value_table)

            # 计算每个状态从策略中得到的动作，然后计算值函数
            # 遍历每个状态
            for state in range(self.observation_spaces):
                # 根据策略取动作
                action = policy[state]

                # 计算q值, 并更新v表
                value_table[state] = self.q_value_calculate(state, action, updated_value_table,
                                                            default_q_value=updated_value_table[state])

            # 收敛判断
            if np.sum((np.fabs(updated_value_table - value_table))) <= threshold:
                print('policy_evaluation converges at {} rounds'.format(iters))
                break
            if iters == max_iter:
                print('policy_evaluation terminates at {} rounds'.format(iters))
                break
        # 返回V表
        return value_table

    def policy_improvement(self, value_table):
        """
        策略改进: 固定状态值函数 => 求最优策略
        :param value_table:
        :return:
        """
        return self.policy_extract(value_table)

    def policy_iteration(self, max_iter=-1, initial_policy=None):
        """
        策略迭代
        :return:
        """
        # 初始化策略
        if initial_policy is None:
            initial_policy = np.zeros(self.observation_spaces, dtype=int)

        iters = 0
        while True:
            iters += 1
            # 策略评估
            value_table = self.policy_evaluation(initial_policy, max_iter)
            # 策略改进
            new_policy = self.policy_improvement(value_table)

            if np.all(initial_policy == new_policy):
                break
            initial_policy = new_policy
        print('policy_iteration converges at {} rounds'.format(iters))

        return new_policy

    def policy_iteration_timer(self, max_iter=-1, initial_policy=None):
        # 初始化策略
        if initial_policy is None:
            initial_policy = np.zeros(self.observation_spaces, dtype=int)

        iters = 0
        while True:
            iters += 1
            print(f'***iters={iters}')
            with timer('policy_evaluation'):
                value_table = self.policy_evaluation(initial_policy, max_iter)
            with timer('policy_improvement'):
                new_policy = self.policy_improvement(value_table)

            if np.all(initial_policy == new_policy):
                break
            initial_policy = new_policy
        print('policy_iteration_timer converges at {} rounds'.format(iters))

        return new_policy
