import gym
import numpy as np
from Src.value_policy_base import ValuePolicyBase
from Src.agent_client import AgentClient


class PolicyIterationClient(ValuePolicyBase):

    def __init__(self, env):
        super().__init__(env)

    # 计算值函数
    def compute_value_function(self, policy, gamma=1.0):
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

    # 策略迭代
    def policy_iteration(self, gamma=1.0):
        # 初始化随机策略，下句代码即为初始策略全为0（向左走）
        random_policy = np.zeros(self.observation_spaces)
        new_policy = random_policy
        # 设置迭代次数
        no_of_iterations = 200000
        # 开始迭代
        for i in range(no_of_iterations):
            # 计算新的值函数
            new_value_function = self.compute_value_function(random_policy, gamma)
            # 得到新的策略
            new_policy = self.extract_policy(new_value_function, gamma)
            # 判断迭代终止条件（策略不变时）
            if np.all(random_policy == new_policy):
                print('Policy-Iteration converged as step %d.' % (i+1))
                break
            # 新的策略为下一次的执行策略
            random_policy = new_policy
        # 返回新的策略
        return new_policy


if __name__ == '__main__':

    env = gym.make('FrozenLake-v1', map_name="4x4", is_slippery=True)
    env.reset()
    env.render()

    policy_client = PolicyIterationClient(env)
    policy = policy_client.policy_iteration()

    print(policy)
    print(AgentClient(env).run_agent(policy, episode=10000, render=False))
