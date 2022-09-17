import numpy as np
import gym
from gym.spaces import Discrete
import matplotlib.pyplot as plt


class SnakeEnv(gym.Env):

    SIZE = 100  # 蛇棋状态空间大小
    DICES = [3, 6]  # 蛇棋动作空间 骰子最大取值

    def __init__(self, random_ladder_num=-1):
        """
        :param random_ladder_num: 随机产生的梯子数目. -1: 使用预置的梯子. >=0: 随机产生梯子
        """
        self.observation_space = Discrete(self.SIZE)
        self.action_space = Discrete(len(self.DICES))

        # 生成梯子, eg. {78:33, 52:97, 71:64, 51:32}
        if random_ladder_num >= 0:
            self.ladders = dict(np.random.randint(0, self.SIZE, size=(random_ladder_num, 2)))
        else:
            # 预置的梯子, 不会交叉
            self.ladders = {4: 17, 10: 32, 30: 51, 62: 80, 26: 83, 8: 39, 59: 66, 79: 87}

        # 将梯子反过来存一遍
        keys = self.ladders.keys()
        for k in list(keys):
            self.ladders[self.ladders[k]] = k
        print('ladders info:', self.ladders)

        # 初始位置
        self.pos = 0

        # 环境转移概率矩阵
        self.P = self.__transform_possibility_matrix()

    def reset(self):
        self.pos = 0
        return self.pos

    def step(self, action: int):
        # 取消random的影响, 使得每次能够随机选择
        np.random.seed()
        step = np.random.randint(self.DICES[action]) + 1
        self.pos = self.__step_forward(self.pos, step)
        return self.pos, self.reward(self.pos), self.done(self.pos), {}

    def reward(self, state: int):
        if self.done(state):
            return 100
        else:
            return -1

    def done(self, state: int):
        return state == self.SIZE - 1

    def render(self):
        """
        绘图蛇棋网格界面
        :return:
        """
        self.__render_ui()
        plt.show()

    def render_policy(self, policy):
        """
        绘图蛇棋网格界面+策略
        :return:
        """
        self.__render_ui()

        # 行动可视化
        order = int(np.sqrt(self.SIZE))
        offset = 0
        action_dict = {0: "A", 1: "B"}
        color_dict = {0: "red", 1: "green"}
        for state in range(len(policy)):
            action = policy[state]
            x, y = state % order + offset, state // order + offset
            plt.text(x + offset, y + offset, action_dict[action], color=color_dict[action])

        plt.show()

    # 以下为私有函数区
    def __render_ui(self):
        """
        绘图蛇棋网格界面
        :return:
        """
        order = int(np.sqrt(self.SIZE))
        # 画网格线但不显示刻度值
        plt.xlim([0, order])
        plt.ylim([0, order])
        plt.xticks(list(range(0, order + 1)))
        plt.yticks(list(range(0, order + 1)))
        plt.gca().axes.xaxis.set_ticklabels([])
        plt.gca().axes.yaxis.set_ticklabels([])
        plt.grid(True)

        # 显示网格的数字
        offset = 0.4
        for y in range(order):
            for x in range(order):
                plt.text(x + offset, y + offset, x + y * order)

        # 梯子
        for start, end in self.ladders.items():
            xstart, ystart = start % order + offset, start // order + offset
            xend, yend = end % order + offset, end // order + offset
            plt.plot([xstart, xend], [ystart, yend], color="black")
        plt.legend(["ladders"], bbox_to_anchor=(0.2, 1.1))

    def __step_forward(self, cur_state, step):
        """
        执行step
        :param cur_state: 当前位置
        :param step: 前进步数
        :return: next_state: 下一时刻位置
        """
        next_state = cur_state + step
        # 超过范围便回退
        if next_state >= self.SIZE:
            # index从0开始, 终点是99
            next_state = 2 * (self.SIZE - 1) - next_state

        # 梯子
        if next_state in self.ladders:
            next_state = self.ladders[next_state]

        return next_state

    def __transform_possibility_matrix(self):
        """
        定义环境的转移概率矩阵
        next_iter = p(s,a,s1)
        next_iter = p, next_state, reward, done
        :return: P
        """
        # 概率转移矩阵
        P = dict()

        for state in range(self.SIZE):
            action_dict = dict()
            for action, dice in enumerate(self.DICES):
                # 概率
                prob = 1.0 / dice
                # 骰子取值为[1, 2, 3,...]
                steps = np.arange(dice) + 1

                next_state_vector = list()
                for step in steps:
                    next_state = self.__step_forward(state, step)
                    next_state_vector.append((prob, next_state, self.reward(next_state), self.done(next_state)))

                action_dict[action] = next_state_vector
            P[state] = action_dict

        return P


if __name__ == '__main__':

    # 预置梯子
    env = SnakeEnv()

    # 设置随机数种子, 保证每次运行时, 都是相同的随机数
    # np.random.seed(1)
    # env = SnakeEnv(random_ladder_num=10)

    env.reset()
    env.render()

    iters = 0
    rewards = 0
    while True:
        iters += 1
        nxt_state, reward, terminate, _ = env.step(0)
        rewards += reward
        if terminate:
            break
    print("iters=", iters)
    print("rewards=", rewards)
