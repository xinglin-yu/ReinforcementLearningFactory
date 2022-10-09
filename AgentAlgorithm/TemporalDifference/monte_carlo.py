

class MonteCarlo(object):
    def __init__(self, epsilon=0.0):
        self.epsilon = epsilon

    def monte_carlo_eval(self, agent, env):
        state = env.reset()
        # state = env.reset()[0]  # Frozen Lake

        # 采样1条轨迹
        episode = []
        while True:
            action = agent.epsilon_greedy_policy(state, self.epsilon)

            next_state, reward, terminate, _ = env.step(action)
            # next_state, reward, terminate, truncated, info = env.step(action)  # Frozen Lake

            episode.append((state, action, reward))
            state = next_state
            if terminate:
                break

        # 计算1条轨迹的状态-动作值
        value = []
        return_val = 0
        for item in reversed(episode):
            return_val = return_val * agent.gamma + item[2]
            value.append((item[0], item[1], return_val))

        # 计算每条轨迹的估计状态-动作值
        for item in reversed(value):
            state, action, return_val = item[0], item[1], item[2]
            agent.value_n[state][action] += 1
            agent.value_q[state][action] += (return_val - agent.value_q[state][action]) / agent.value_n[state][action]

    def iteration(self, agent, env):
        for i in range(10):  # 迭代10轮
            for j in range(100):  # 每轮采样100条轨迹
                self.monte_carlo_eval(agent, env)

            # 每一轮只改进1次策略
            agent.policy_improve()

        return agent.pi
