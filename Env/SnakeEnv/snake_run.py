from CommonUtils.decorator import time_wrapper
from Env.SnakeEnv.snake_env import SnakeEnv
from Env.SnakeEnv.snake_client import SnakeClient
from AgentAlgorithm.policy_iteration_client import PolicyIterationClient
from AgentAlgorithm.value_iteration_client import ValueIterationClient
from AgentAlgorithm.general_iteration_client import GeneralIterationClient


class SnakeRun(object):

    @staticmethod
    def base_policy_run(env=None):
        """
        预置的策略
        :param env: 环境
        :return:
        """
        if env is None:
            env = SnakeEnv(random_ladder_num=0, dices=[3, 6])

        # 3 种不同的策略
        policy_0 = [0] * 100  # 全部投掷1-3的策略
        policy_1 = [1] * 100  # 全部投掷1-6的策略
        policy_ref = [1] * 97 + [0] * 3  # 参考策略

        print('policy_0: (max_episodes, avg_rewards)={}'.format(SnakeClient.run_episodes(env, policy_0)))
        print('policy_1: (max_episodes, avg_rewards)={}'.format(SnakeClient.run_episodes(env, policy_1)))
        print('policy_ref: (max_episodes, avg_rewards)={}'.format(SnakeClient.run_episodes(env, policy_ref)))

    @staticmethod
    @time_wrapper
    def policy_iteration_run1():
        """
        0个梯子时的 策略迭代
        :return:
        """
        env = SnakeEnv(random_ladder_num=0)

        # base policy
        SnakeRun.base_policy_run(env)

        # optimal policy
        pi_algo = PolicyIterationClient(env, gamma=0.8)
        policy = pi_algo.policy_iteration()
        print('policy_opt={}'.format(policy))
        print('policy_opt: (max_episodes, avg_rewards)={}'.format(SnakeClient.run_episodes(env, policy)))
        env.render_policy(policy, filename="../../Doc/Env.SnakeEnv/SnakeEnv_ladders0_PolicyIteration_Policy.png")

    @staticmethod
    @time_wrapper
    def policy_iteration_run2():
        """
        多个梯子时的 策略迭代
        :return:
        """
        # 有多个梯子的环境
        env = SnakeEnv()

        # base policy
        SnakeRun.base_policy_run(env)

        # optimal policy
        pi_algo = PolicyIterationClient(env, gamma=0.8)
        policy = pi_algo.policy_iteration()
        # policy = pi_algo.policy_iteration_timer(max_iter=-1)
        print('policy_opt={}'.format(policy))
        print('policy_opt: (max_episodes, avg_rewards)={}'.format(SnakeClient.run_episodes(env, policy)))
        env.render_policy(policy, filename="../../Doc/Env.SnakeEnv/SnakeEnv_ladders8_PolicyIteration_Policy.png")

    @staticmethod
    @time_wrapper
    def value_iteration_run():
        env = SnakeEnv()
        vi_algo = ValueIterationClient(env, gamma=0.8)
        policy = vi_algo.value_iteration()
        print('policy_opt={}'.format(policy))
        print('policy_opt: (max_episodes, avg_rewards)={}'.format(SnakeClient.run_episodes(env, policy)))
        env.render_policy(policy)

    @staticmethod
    @time_wrapper
    def general_iteration_run():
        env = SnakeEnv()
        algo = GeneralIterationClient(env, gamma=0.8)
        policy = algo.general_iteration()
        print('policy_opt={}'.format(policy))
        print('policy_opt: (max_episodes, avg_rewards)={}'.format(SnakeClient.run_episodes(env, policy)))
        env.render_policy(policy)


if __name__ == '__main__':
    # SnakeRun.base_policy_run()
    # SnakeRun.policy_iteration_run1()
    SnakeRun.policy_iteration_run2()
    # SnakeRun.value_iteration_run()
    # SnakeRun.general_iteration_run()

# TODO: 对于给定的环境, 最优的step是什么呢?
