import gym
import numpy as np
from typing import List

from Src.PolicyHelper.agent_client import AgentClient
from Src.PolicyHelper.image_client import ImageClient
from Src.PolicyHelper.maze_path_client import MazePathClient


class PolicyHelperClient:

    @staticmethod
    def show(env, policy: List[int], file_suffix="", snapshot_folder = "../../Doc/Snapshot/"):
        """
        This method is for post show of an policy
        :param env: the frozen lake env
        :param policy: a = pi(s), the policy of env interaction
        :param file_suffix: file suffix to save
        :return:
        """
        print("(1) Trained policy is: ", policy)

        print("(2) Test success rate")
        agent_client = AgentClient(env)
        episode_terminate_points, _ = agent_client.calculate_possibility(policy)
        print("Test Theoretical Success Rate", episode_terminate_points)
        print("Test Experimental Success Rate", agent_client.run_agent(policy, episode=1000, render=False))

        # 获取所有路径
        print("(3) Get all paths under policy: ", end=" ")
        maze_path_client = MazePathClient(env)
        maze_path_client.get_paths_to_hole_goal(state=0, policy=policy)
        print(len(maze_path_client.paths))
        # for path in agent_client.maze_path_client.paths:
        #     print(path)

        # 绘制路径
        print("(4) Render figure")
        frame = env.render(mode='rgb_array')
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
    # increase the max_episode_steps from 100 to 1000, so that the agent will not end prematurely
    # https://github.com/openai/gym/blob/master/gym/wrappers/time_limit.py

    order = 4
    env = gym.make('FrozenLake-v1', map_name=f"{order}x{order}", is_slippery=True, max_episode_steps=1000)
    env.reset()
    frame = env.render(mode='rgb_array')

    # given policy
    policy = [0, 3, 3, 3, 0, 0, 0, 0, 3, 1, 0, 0, 0, 2, 1, 0]

    # policy analysis and show
    PolicyHelperClient.show(env, policy, file_suffix="")
