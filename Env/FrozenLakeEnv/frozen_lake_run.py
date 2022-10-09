from CommonUtils.decorator import time_wrapper
from Env.FrozenLakeEnv.frozen_lake_client import FrozenLakeClient
from AgentAlgorithm.PolicyValueIteration.policy_iteration_client import PolicyIterationClient
from AgentAlgorithm.PolicyValueIteration.value_iteration_client import ValueIterationClient
from AgentAlgorithm.PolicyValueIteration.general_iteration_client import GeneralIterationClient


class FrozenLakeRun:

    @staticmethod
    def preload_policy_run():
        env = FrozenLakeClient.create_preload_env_obj(order=4)

        # given policy
        policy = [0, 3, 3, 3, 0, 0, 0, 0, 3, 1, 0, 0, 0, 2, 1, 0]

        # calculate arriving possibility
        res = FrozenLakeClient.calculate_possibility(env, policy)
        print(res)

        # iteration verification
        print(FrozenLakeClient.run_episodes(env, policy, episodes=10000, render=False))

    @staticmethod
    @time_wrapper
    def policy_iteration_run():
        env = FrozenLakeClient.create_preload_env_obj(order=8)

        # get policy
        policy_client = PolicyIterationClient(env)
        policy = policy_client.policy_iteration()

        # policy analysis and show
        FrozenLakeClient.show(env, policy, file_suffix="_PolicyIteration")

    @staticmethod
    @time_wrapper
    def value_iteration_run():
        env = FrozenLakeClient.create_preload_env_obj(order=8)

        # get policy
        value_client = ValueIterationClient(env)
        policy = value_client.value_iteration()

        # policy analysis and show
        FrozenLakeClient.show(env, policy, file_suffix="_ValueIteration")

    @staticmethod
    @time_wrapper
    def general_iteration_run():
        env = FrozenLakeClient.create_preload_env_obj(order=8)

        # get policy
        algo_client = GeneralIterationClient(env)
        policy = algo_client.general_iteration(value_max_iter=500)

        # policy analysis and show
        FrozenLakeClient.show(env, policy, file_suffix="_GeneralIteration")


if __name__ == '__main__':
    FrozenLakeRun.policy_iteration_run()
    FrozenLakeRun.value_iteration_run()
    FrozenLakeRun.general_iteration_run()
