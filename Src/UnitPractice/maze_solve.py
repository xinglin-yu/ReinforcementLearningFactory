class Maze:

    def __init__(self, maze):
        """
        initial maze.
        0 can not arrive
        1 can arrive
        :param maze: input maze 2D matrix
        """
        self.order = len(maze)
        self.maze = maze

        self.path = []  # 栈: 保存当前路径
        self.paths = []  # 队列: 保存所有路径

    def rest(self):
        """
        for multi calls.
        """
        self.path = []
        self.paths = []

    def is_safe(self, x, y):
        return 0 <= x < self.order and \
               0 <= y < self.order \
               and self.maze[x][y] == 1

    def get_path(self, x, y):
        """
        获取一条路径
        """
        # (1) 终止条件-到达目的地
        if x == self.order - 1 and y == self.order - 1:  # 到达目的地
            self.path.append((x, y))  # 记录位置
            return True

        # (1) 终止条件-当前位置不可行
        if not self.is_safe(x, y):
            return False

        # (2) 当前位置`入栈`
        self.path.append((x, y))  # 入栈

        # (3) `递归`地向各个可能的方向走, 找到路径便返回true
        if self.get_path(x + 1, y):  # 向右
            return True
        if self.get_path(x, y + 1):  # 向下
            return True

        # (4) 从当前位置无法到达终点, 于是`回溯` (即当前位置出栈)
        self.path.pop()  # 回溯=出栈

        # (5) 返回
        return False

    def get_paths(self, x, y):
        """
        获取所有路径
        """
        # (1) 终止条件-到达目的地
        if x == self.order - 1 and y == self.order - 1:
            self.path.append((x, y))  # 记录位置
            self.paths.append([i for i in self.path])
            return

        # (1) 终止条件-当前位置不可行
        if not self.is_safe(x, y):
            return

        # (2) 当前位置`入栈`
        self.path.append((x, y))  # 入栈

        # (3)`递归`地向各个可能的方向前进
        self.get_paths(x + 1, y)
        self.get_paths(x, y + 1)

        # (4) 从当前位置出发无可行路径, 于是`回溯`(即当前位置出栈)
        self.path.pop()  # 回溯=出栈


if __name__ == '__main__':
    maze = [[1, 1, 1, 1],
            [1, 0, 1, 0],
            [1, 1, 1, 0],
            [0, 1, 1, 1]]

    obj = Maze(maze)

    # 获取一条路径
    print("one path")
    if obj.get_path(0, 0):
        print(obj.path)

    # 获取所有路径
    obj.rest()
    print("all paths")
    obj.get_paths(0, 0)
    for path in obj.paths:
        print(path)
