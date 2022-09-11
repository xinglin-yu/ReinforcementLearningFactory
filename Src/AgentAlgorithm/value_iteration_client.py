import gym
import numpy as np
from value_policy_base import ValuePolicyBase
from Src.PolicyHelper.policy_helper_client import PolicyHelperClient


class ValueIterationClient(ValuePolicyBase):

    def __init__(self, env):
        super().__init__(env)

    # 值迭代
    def value_iteration(self, gamma=1.0):
        # 初始化状态值表（V表）
        value_table = np.zeros(self.observation_spaces)
        # 迭代次数
        no_of_iterations = 100000
        # 收敛判断阈值
        threshold = 1e-20
        # 开始迭代: 4层循环
        for i in range(no_of_iterations):
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
                        # next_state是否是终止状态？if Yes：done=True；else：done=False
                        trans_prob, next_state, reward, done = next_sr
                        # 计算next_states_reward（公式）
                        next_states_rewards.append(
                            (trans_prob*(reward+gamma*updated_value_table[next_state])))
                        # 计算Q值（公式）
                        Q_value.append(np.sum(next_states_rewards))
                        # 取最大Q值更新V表，即更新当前状态的V值
                        value_table[state] = max(Q_value)

            # 收敛判断
            if np.sum(np.fabs(updated_value_table - value_table)) <= threshold:
                print("Value-itration converged at itration # %d" % (i+1))
                break
        # 返回V表
        return value_table


if __name__ == '__main__':

    # frozen lake order
    order = 4
    # preloaded maps
    env = gym.make('FrozenLake-v1', map_name=f"{order}x{order}", is_slippery=True, max_episode_steps=1000)
    file_suffix = "_ValueIteration"

    # random maps
    np.random.seed(0)  # random seed can enable same env in every run
    # env = gym.make('FrozenLake-v1', desc=generate_random_map(size=order), is_slippery=True, max_episode_steps=1000)
    # file_suffix = "_Random_ValueIteration"

    env.reset()
    env.render(mode='rgb_array')

    value_client = ValueIterationClient(env)
    value_table = value_client.value_iteration()
    policy = value_client.extract_policy(value_table)

    # policy analysis and show
    PolicyHelperClient.show(env, policy, file_suffix=file_suffix)


# https://blog.csdn.net/njshaka/article/details/89237941
