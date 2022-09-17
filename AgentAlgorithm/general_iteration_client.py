from AgentAlgorithm.policy_iteration_client import PolicyIterationClient
from AgentAlgorithm.value_iteration_client import ValueIterationClient


class GeneralIterationClient:

    def __init__(self, env, gamma=1):
        self.pi_algo = PolicyIterationClient(env, gamma)
        self.vi_algo = ValueIterationClient(env, gamma)

    def general_iteration(self):
        """
        泛化迭代
        :return:
        """
        policy = self.vi_algo.value_iteration(max_iter=100)
        new_policy = self.pi_algo.policy_iteration(initial_policy=policy, max_iter=100)

        return new_policy
