import numpy as np
from CommonUtils.decorator import time_wrapper
from CommonUtils.decorator import timer
from Env.FrozenLakeEnv.frozen_lake_client import FrozenLakeClient

from AgentAlgorithm.TemporalDifference.model_free_agent import ModelFreeAgent
from AgentAlgorithm.TemporalDifference.monte_carlo import MonteCarlo
from AgentAlgorithm.TemporalDifference.sarsa import SARSA
from AgentAlgorithm.TemporalDifference.qlearning import QLearning


class SnakeRun(object):

    @staticmethod
    @time_wrapper
    def monte_carlo_demo():
        env = FrozenLakeClient.create_preload_env_obj(order=4)

        agent = ModelFreeAgent(env)
        mc = MonteCarlo(0.5)
        with timer('Timer Monte Carlo Iter'):
            policy = mc.iteration(agent, env)

        # policy analysis and show
        FrozenLakeClient.show(env, policy, file_suffix="_MonteCarloIteration")


if __name__ == '__main__':

    SnakeRun.monte_carlo_demo()
    # SnakeRun.sarsa_demo()
    # SnakeRun.qlearning_demo()
