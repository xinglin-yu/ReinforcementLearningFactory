import gym


class MazePathClient:

    def __init__(self, env):
        """
        initial maze.
        F: Frozen: can arrive
        H: hole: can not arrive
        S: Start
        G: Goal
        :param env: frozen lake env
        """
        self.env = env
        self.maze = env.unwrapped.desc
        self.order = len(self.maze)

        self.path = []  # one path
        self.paths = []  # all paths

    def reset(self):
        self.path = []  # one path
        self.paths = []  # all paths

    def is_safe(self, x, y):
        """
        判断是否在范围内
        :param x:
        :param y:
        :return:
        """
        return 0 <= x < self.order and 0 <= y < self.order

    def get_paths_to_goal(self, pos):
        """
        get all paths to goal
        state 和 pos=(x,y) 是唯一映射的
        x, y = state // self.order, state % self.order
        state = x * self.order + y
        :param pos: (x,y) current position
        :return:
        """
        x, y = pos[0], pos[1]
        state = x * self.order + y

        # 超过边界 or 已走过 or 到达hole, 便结束
        if not self.is_safe(x, y) or (state in self.path) or self.maze[x][y] == b'H':
            return

        # 到达goal
        if self.maze[x][y] == b'G':
            # 存储路径
            sol_path = [i for i in self.path]
            sol_path.append(state)
            self.paths.append(sol_path)
            return

        # 可行的区域
        self.path.append(state)  # 入栈标记

        # 向各个可能的方向走
        self.get_paths_to_goal((x - 1, y))  # left
        self.get_paths_to_goal((x + 1, y))  # right
        self.get_paths_to_goal((x, y + 1))  # up
        self.get_paths_to_goal((x, y - 1))  # down

        # 标记当前单元不通, 然后回溯
        self.path.pop()  # 回退

    def get_paths_to_hole_goal(self, state, policy):
        """
        get all paths to Hole and Goal within a policy
        :param state: current state
        :param policy: a=pi(s)
        :return:
        """
        x, y = state // self.order, state % self.order

        # 超过边界 or 已走过, 便结束
        if not self.is_safe(x, y) or (state in self.path):
            return

        # 给定策略下, 到达goal或hole 都算可能的路径
        if self.maze[x][y] == b'G' or self.maze[x][y] == b'H':
            # 存储路径
            sol_path = [i for i in self.path]
            sol_path.append(state)
            self.paths.append(sol_path)
            return

        # 记录当前单元: 即入栈
        self.path.append(state)

        # 向各个可能的方向走
        action = policy[state]
        for next_sr in self.env.P[state][action]:
            trans_prob, next_state, reward, done = next_sr
            self.get_paths_to_hole_goal(next_state, policy)

        # 标记当前单元不通, 然后回溯: 即出栈
        self.path.pop()


if __name__ == '__main__':
    order = 4
    env = gym.make('FrozenLake-v1', map_name=f"{order}x{order}", is_slippery=True, max_episode_steps=1000)
    env.reset()
    frame = env.render(mode='rgb_array')

    obj = MazePathClient(env)
    obj.reset()

    # 获取所有路径
    print("all paths")
    obj.get_paths_to_hole_goal(state=0, policy=[0, 3, 3, 3, 0, 0, 0, 0, 3, 1, 0, 0, 0, 2, 1, 0])
    for path in obj.paths:
        print(path)

