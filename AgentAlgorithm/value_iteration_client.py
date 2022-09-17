import numpy as np
from AgentAlgorithm.value_policy_base import ValuePolicyBase


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
            # 计算每个状态下所有行为的next_state_rewards,并更新状态动作值表（Q表），最后取最大Q值更新V表
            # 遍历每个状态
            for state in range(self.observation_spaces):
                # 初始化存储Q值的列表
                Q_value = []
                # 遍历每个动作
                for action in range(self.action_spaces):
                    # 初始化存储下一个状态的奖励的列表
                    next_states_rewards = []
                    # P[][]是环境定义的变量,存储状态s下采取动作a得到的元组数据（转移概率，下一步状态，奖励，完成标志）
                    for next_sr in self.env.P[state][action]:
                        # next_state是否是终止状态？if Yes：done=True; else：done=False
                        trans_prob, next_state, reward, done = next_sr
                        # 计算next_states_reward（公式）
                        next_states_rewards.append(
                            (trans_prob*(reward + self.gamma * updated_value_table[next_state])))
                    # 计算Q值（公式）
                    Q_value.append(np.sum(next_states_rewards))
                    # 取最大Q值更新V表，即更新当前状态的V值
                    value_table[state] = max(Q_value)

            # 收敛判断
            if np.sum(np.fabs(updated_value_table - value_table)) <= threshold:
                print('value_iteration converges at {} rounds'.format(iters))
                break
            if iters == max_iter:
                print('value_iteration terminates at {} rounds'.format(iters))
                break

        # 返回确定性策略
        return self.policy_extract(value_table)


# https://blog.csdn.net/njshaka/article/details/89237941
