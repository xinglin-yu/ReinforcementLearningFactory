import numpy as np
from CommonUtils.decorator import time_wrapper
from CommonUtils.decorator import timer
from Env.SnakeEnv.snake_env import SnakeEnv
from Env.SnakeEnv.snake_client import SnakeClient

from AgentAlgorithm.TemporalDifference.model_free_agent import ModelFreeAgent
from AgentAlgorithm.TemporalDifference.monte_carlo import MonteCarlo
from AgentAlgorithm.TemporalDifference.sarsa import SARSA
from AgentAlgorithm.TemporalDifference.qlearning import QLearning


class SnakeRun(object):

    @staticmethod
    @time_wrapper
    def monte_carlo_demo():
        env = SnakeEnv()
        agent = ModelFreeAgent(env)
        mc = MonteCarlo(0.5)
        with timer('Timer Monte Carlo Iter'):
            mc.iteration(agent, env)
        print('return_pi={}'.format(SnakeClient.run_episodes(env, agent.pi)))
        print(agent.pi)
        env.render_policy(agent.pi, render=False,
                          filename="../../Doc/Env.SnakeEnv/SnakeEnv_ladders8_MonteCarlo_Policy.png")

    @staticmethod
    @time_wrapper
    def sarsa_demo():
        env = SnakeEnv()
        agent = ModelFreeAgent(env)
        mc = SARSA(0.5)
        with timer('Timer sarsa Iter'):
            mc.iteration(agent, env)
        print('return_pi={}'.format(SnakeClient.run_episodes(env, agent.pi)))
        print(agent.pi)
        env.render_policy(agent.pi, render=False)

    @staticmethod
    @time_wrapper
    def qlearning_demo():
        env = SnakeEnv()
        agent = ModelFreeAgent(env)
        mc = QLearning(0.5)
        with timer('Timer q learning Iter'):
            mc.iteration(agent, env)
        print('return_pi={}'.format(SnakeClient.run_episodes(env, agent.pi)))
        print(agent.pi)
        env.render_policy(agent.pi, render=True)


if __name__ == '__main__':

    SnakeRun.monte_carlo_demo()
    # SnakeRun.sarsa_demo()
    # SnakeRun.qlearning_demo()
