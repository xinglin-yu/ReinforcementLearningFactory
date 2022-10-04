import gym
import numpy as np
from typing import List
from gym.envs.toy_text.frozen_lake import generate_random_map
from Env.FrozenLakeEnv.image_client import ImageClient
from Env.FrozenLakeEnv.maze_path_client import MazePathClient


class FrozenLakeClient:

    @staticmethod
    def create_preload_env_obj(order=4):
        """
        :param order: frozen lake order
        :return:
        """
        # increase the max_episode_steps from 100 to 1000, so that the agent will not end prematurely
        # https://github.com/openai/gym/blob/master/gym/wrappers/time_limit.py

        # preloaded maps
        env = gym.make('FrozenLake-v1',
                       map_name=f"{order}x{order}",
                       is_slippery=True,
                       max_episode_steps=1000,
                       render_mode='rgb_array')

        env.reset()
        env.render()

        return env

    @staticmethod
    def create_random_env_obj(order=4):
        """
        :param order: frozen lake order
        :return:
        """
        # random maps
        np.random.seed(0)  # random seed can enable same env in every run
        env = gym.make('FrozenLake-v1',
                       desc=generate_random_map(size=order),
                       is_slippery=True,
                       max_episode_steps=1000,
                       render_mode='rgb_array')

        env.reset()
        env.render()

        return env

    @staticmethod
    def run_episodes(env, policy, episodes=10, render=True):
        # 记录到到end point的次数
        end_times = 0
        end = env.observation_space.n - 1

        done_dict = dict()

        for i in range(episodes):
            observation = env.reset()[0]
            if render:
                print(f"iter={i}")
                env.render()
            # 步数
            step_idx = 0

            while True:
                action = int(policy[observation])
                observation, reward, terminated, truncated, info = env.step(action)
                step_idx += 1

                if render:
                    print("       step_idx=", step_idx)
                    env.render()

                if terminated:
                    if observation == end:
                        # time.sleep(1)
                        end_times += 1
                    # record terminate point
                    if observation not in done_dict:
                        done_dict[observation] = 0
                    done_dict[observation] += 1
                    break
            env.close()

        return episodes, end_times, end_times / episodes, done_dict

    @staticmethod
    def calculate_possibility(env, policy):
        """
        计算理论成功率
        :param env:
        :param policy:
        :return:
        """
        # exact env info
        observation_spaces = env.observation_space.n
        action_spaces = env.action_space.n
        trans_possibility = env.P

        # 转移概率矩阵
        possibility_matrix = np.zeros(shape=(observation_spaces, observation_spaces))

        # 遍历每个状态
        for state in range(observation_spaces):
            # 根据策略取动作
            action = policy[state]
            for next_sr in trans_possibility[state][action]:
                trans_prob, next_state, reward, done = next_sr
                possibility_matrix[int(next_state), state] += trans_prob

        # 初始状态向量
        state_vector = np.zeros(shape=(observation_spaces, 1))
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

    @staticmethod
    def show(env, policy: List[int], file_suffix="", snapshot_folder="../../Doc/Env.FrozenLakeEnv/Snapshot/"):
        """
        This method is for post show of an policy
        :param env: the frozen lake env
        :param policy: a = pi(s), the policy of env interaction
        :param file_suffix: file suffix to save
        :param snapshot_folder: the path to save result
        :return:
        """
        print("(1) Trained policy is: ", policy)

        print("(2) Test success rate")
        episode_terminate_points, _ = FrozenLakeClient.calculate_possibility(env, policy)
        print("Test Theoretical Success Rate", episode_terminate_points)
        print("Test Experimental Success Rate", FrozenLakeClient.run_episodes(env, policy, episodes=1000, render=False))

        # 获取所有路径
        print("(3) Get all paths under policy: ", end=" ")
        maze_path_client = MazePathClient(env)
        maze_path_client.get_paths_to_hole_goal(state=0, policy=policy)
        print(len(maze_path_client.paths))

        # 绘制路径
        print("(4) Render figure")
        frame = env.render()
        order = int(np.sqrt(env.observation_space.n))
        ImageClient.save_frame(frame, filename=f"{snapshot_folder}FrozenLake{order}x{order}{file_suffix}.png")
        ImageClient.add_arrow(policy, order=order,
                              in_img=f"{snapshot_folder}FrozenLake{order}x{order}{file_suffix}.png",
                              out_img=f"{snapshot_folder}FrozenLake{order}x{order}{file_suffix}_Policy.png")
        ImageClient.add_path(maze_path_client.paths, order=order,
                             episode_terminate_points=episode_terminate_points,
                             in_img=f"{snapshot_folder}FrozenLake{order}x{order}{file_suffix}.png",
                             out_img=f"{snapshot_folder}FrozenLake{order}x{order}{file_suffix}_Poths.png")

        print("All successful")


if __name__ == '__main__':

    env = FrozenLakeClient.create_preload_env_obj()
