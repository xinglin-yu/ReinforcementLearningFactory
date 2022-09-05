import gym
import numpy as np


class AgentClient:

    def __init__(self, env):
        self.env = env
        self.observation_spaces = env.observation_space.n
        self.action_spaces = env.action_space.n

    def run_agent(self, policy, episode=10, render=True):
        # 记录到到end point的次数
        end_times = 0
        end = self.env.observation_space.n - 1

        done_dict = dict()

        for i in range(episode):
            obs = self.env.reset()
            if render:
                print(f"iter={i}")
                self.env.render()
            # 步数
            step_idx = 0

            while True:
                obs, reward, done, info = self.env.step(int(policy[obs]))
                step_idx += 1

                if render:
                    print("       step_idx=", step_idx)
                    self.env.render()

                if done:
                    if obs == end:
                        # time.sleep(1)
                        end_times += 1

                    # record terminate point
                    if obs not in done_dict:
                        done_dict[obs] = 0
                    done_dict[obs] += 1

                    break

            self.env.close()

        return episode, end_times, end_times / episode, done_dict

    def calculate_possibility(self, policy):

        # 转移概率矩阵
        possibility_matrix = np.zeros(shape=(self.observation_spaces, self.observation_spaces))

        # 遍历每个状态
        for state in range(self.observation_spaces):
            # 根据策略取动作
            action = policy[state]
            for next_sr in self.env.P[state][action]:
                trans_prob, next_state, reward, done = next_sr
                possibility_matrix[int(next_state), state] += trans_prob

        # 初始状态向量
        state_vector = np.zeros(shape=(self.observation_spaces, 1))
        state_vector[0] = 1

        # 开始迭代, 至收敛
        threshold = 1e-10  # 收敛判断阈值
        err = threshold + 1  # 误差
        times = 0  # 迭代次数
        # 循环直到收敛
        while err > threshold:
            state_vector_next = np.dot(possibility_matrix, state_vector)
            # 计算误差
            err = np.linalg.norm(np.subtract(state_vector_next, state_vector), ord=2)
            times += 1
            state_vector = state_vector_next

        # format the output
        state_vector_dict = {i: float(np.around(state_vector[i], 3)[0]) for i in range(len(state_vector))}
        episode_terminate_points = {key: value for key, value in state_vector_dict.items() if value > 0}

        return episode_terminate_points, times


if __name__ == '__main__':
    # increase the max_episode_steps from 100 to 1000, so that the agent will not end prematurely
    # https://github.com/openai/gym/blob/master/gym/wrappers/time_limit.py

    order = 4
    env = gym.make('FrozenLake-v1', map_name=f"{order}x{order}", is_slippery=True, max_episode_steps=1000)
    env.reset()
    frame = env.render(mode='rgb_array')

    # given policy
    policy = [0, 3, 3, 3, 0, 0, 0, 0, 3, 1, 0, 0, 0, 2, 1, 0]
    print(policy)

    # calculate arriving possibility
    agent_client = AgentClient(env)
    print(agent_client.calculate_possibility(policy))

    # iteration verification
    print(agent_client.run_agent(policy, episode=10000, render=False))

