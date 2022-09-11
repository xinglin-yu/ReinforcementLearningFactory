import gym
from gym.envs.toy_text.frozen_lake import generate_random_map
import numpy as np
from value_policy_base import ValuePolicyBase
from Src.PolicyHelper.policy_helper_client import PolicyHelperClient


class PolicyIterationClient(ValuePolicyBase):

    def __init__(self, env):
        super().__init__(env)

    # 计算值函数: 策略评估
    def policy_evaluation(self, policy, gamma=1.0):
        # 初始化V表
        value_table = np.zeros(self.observation_spaces)
        # 收敛判断阈值
        threshold = 1e-10
        # 循环直到收敛
        while True:
            # 初始化更新后的V表（旧表复制过来）
            updated_value_table = np.copy(value_table)
            # 计算每个状态从策略中得到的动作，然后计算值函数
            # 遍历每个状态
            for state in range(self.observation_spaces):
                # 根据策略取动作
                action = policy[state]

                # 更新该状态的V值（公式）
                value_table[state] = 0  # 每次都要重新赋值
                for next_sr in self.env.P[state][action]:
                    # next_state是否是终止状态？if Yes：done=True；else：done=False
                    trans_prob, next_state, reward, done = next_sr
                    value_table[state] += trans_prob * (reward + gamma * updated_value_table[next_state])

                # # 更新该状态的V值（公式）
                # value_table[state] = sum([trans_prob*(reward+gamma*updated_value_table[next_state])
                #                           for trans_prob, next_state, reward, done in env.P[state][action]])
            # 收敛判断
            if np.sum((np.fabs(updated_value_table - value_table))) <= threshold:
                break
        # 返回V表
        return value_table

    # 策略改进
    def policy_improvement(self, gamma=1.0):
        # 初始化随机策略，下句代码即为初始策略全为0（向左走）
        random_policy = np.zeros(self.observation_spaces)
        new_policy = random_policy
        # 设置迭代次数
        no_of_iterations = 200000
        # 开始迭代
        for i in range(no_of_iterations):
            # 计算新的值函数
            new_value_function = self.policy_evaluation(random_policy, gamma)
            # 得到新的策略
            new_policy = self.extract_policy(new_value_function, gamma)
            # 判断迭代终止条件（策略不变时）
            if np.all(random_policy == new_policy):
                print('Policy-Iteration converged as step %d.' % (i + 1))
                break
            # 新的策略为下一次的执行策略
            random_policy = new_policy
        # 返回新的策略
        return new_policy


if __name__ == '__main__':
    # frozen lake order
    order = 4
    # preloaded maps
    env = gym.make('FrozenLake-v1', map_name=f"{order}x{order}", is_slippery=True, max_episode_steps=1000)
    file_suffix = "_PolicyIteration"

    # random maps
    np.random.seed(0)  # random seed can enable same env in every run
    # env = gym.make('FrozenLake-v1', desc=generate_random_map(size=order), is_slippery=True, max_episode_steps=1000)
    # file_suffix = "_Random_PolicyIteration"

    env.reset()
    env.render(mode='rgb_array')

    policy_client = PolicyIterationClient(env)
    policy = policy_client.policy_improvement()

    # policy analysis and show
    PolicyHelperClient.show(env, policy, file_suffix=file_suffix)
