

class SnakeClient:

    @staticmethod
    def run_one_episode(env, policy):
        """
        运行一个回合, 直到结束
        :param env: 环境
        :param policy: 策略
        :return: 累计奖赏值
        """
        state = env.reset()
        return_val = 0
        while True:
            act = policy[state]
            state, reward, terminate, _ = env.step(act)
            return_val += reward
            if terminate:
                break
        return return_val

    @staticmethod
    def run_episodes(env, policy, episodes=1000):
        """
        运行多个回合, 计算平均值
        :param env: 环境
        :param policy: 策略
        :param episodes: 回合数
        :return: 累计奖赏值
        """
        sum_rewards = 0

        for i in range(episodes):
            sum_rewards += SnakeClient.run_one_episode(env, policy)

        return episodes, int(sum_rewards / episodes)
