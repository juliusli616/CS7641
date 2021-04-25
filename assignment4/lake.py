import numpy as np
import gym
from gym.envs.toy_text.frozen_lake import generate_random_map
import matplotlib.pyplot as plt
import time
import mdptoolbox
import mdptoolbox, mdptoolbox.example
import seaborn as sns
import math
import random


def run():
    def draw_policy(policy, V, env, size, file_name, plt_title):
        import matplotlib.pyplot as plt
        print(policy)
        print(env.desc)
        desc = env.desc.astype('U')

        policy_flat = np.argmax(policy, axis=1)
        policy_grid = policy_flat.reshape((size,size))
        V           = V.reshape((size,size))
        sns.set()

        print(size)
        policy_list = np.chararray((size,size), unicode=True)

        policy_list[np.where(policy_grid == 1)] = 'v'
        policy_list[np.where(policy_grid == 2)] = '>'
        policy_list[np.where(policy_grid == 0)] = '<'
        policy_list[np.where(policy_grid == 3)] = '^'

        policy_list[np.where(desc == 'H')]  = '0'
        policy_list[np.where(desc == 'G')] = 'G'
        policy_list[np.where(desc == 'S')] = 'S'
        a4_dims = (12, 12)


        fig, ax = plt.subplots(figsize = a4_dims)
        sns.heatmap(V, annot=policy_list, fmt='', ax=ax)
        #sns_plot.figure.savefig("plots/maze_solution.png")
        plt.title(plt_title)
        plt.savefig(file_name)
        plt.tight_layout()
        plt.close()
        return True


    # Value iteration
    def value_iteration(env, theta=10e-8, discount_factor=1.0):

        def one_step_lookahead(state, V):
            A = np.zeros(env.nA)
            for a in range(env.nA):
                for prob, next_state, reward, done in env.P[state][a]:
                    A[a] += prob * (reward + discount_factor * V[next_state])
            return A
        V = np.zeros(env.nS)
        DELTA_ARR = []
        V_ARR = []
        while True:
            delta = 0
            for s in range(env.nS):
                # Do a one-step lookahead to find the best action
                A = one_step_lookahead(s, V)
                best_action_value = np.max(A)
                # Calculate delta across all states seen so far
                delta = max(delta, np.abs(best_action_value - V[s]))
                # Update the value function. Ref: Sutton book eq. 4.10.
                V[s] = best_action_value
                # Check if we can stop
            DELTA_ARR.append(delta)
            V_ARR.append(V)
            if delta < theta:
                break
        # Create a deterministic policy using the optimal value function
        policy = np.zeros([env.nS, env.nA])
        for s in range(env.nS):
            # One step lookahead to find the best action for this state
            A = one_step_lookahead(s, V)
            best_action = np.argmax(A)
            # Always take the best action
            policy[s, best_action] = 1.0
        return DELTA_ARR, V_ARR, policy


    # Policy iteration
    def policy_eval(policy, env, v_prev,  discount_factor=1.0, theta=0.0001 ):

        """
        Evaluate a policy given an environment and a full description of the environment's dynamics.
        Args:
            policy: [S, A] shaped matrix representing the policy.
            env: OpenAI env. env.P represents the transition probabilities of the environment.
                env.P[s][a] is a list of transition tuples (prob, next_state, reward, done).
                env.nS is a number of states in the environment.
                env.nA is a number of actions in the environment.
            theta: We stop evaluation once our value function change is less than theta for all states.
            discount_factor: Gamma discount factor.

        Returns:
            Vector of length env.nS representing the value function.
        """
        # Start with a random (all 0) value function

        V = v_prev
        num_iters = 0
        while True:
            num_iters = num_iters + 1
            delta = 0
            # For each state, perform a "full backup"
            for s in range(env.nS):
                v = 0
                # Look at the possible next actions
                for a, action_prob in enumerate(policy[s]):
                    # For each action, look at the possible next states...
                    for prob, next_state, reward, done in env.P[s][a]:
                        # Calculate the expected value
                        v += action_prob * prob * (reward + discount_factor * V[next_state])
                # How much our value function changed (across any states)
                delta = max(delta, np.abs(v - V[s]))
                V[s] = v
            # Stop evaluating once our value function change is below a threshold
            if delta < theta:
                break
        return np.array(V)

    def policy_improvement(env, policy_eval_fn=policy_eval, discount_factor=1.0):
        """
        Policy Improvement Algorithm. Iteratively evaluates and improves a policy
        until an optimal policy is found.

        Args:
            env: The OpenAI environment.
            policy_eval_fn: Policy Evaluation function that takes 3 arguments:
                policy, env, discount_factor.
            discount_factor: gamma discount factor.

        Returns:
            A tuple (policy, V).
            policy is the optimal policy, a matrix of shape [S, A] where each state s
            contains a valid probability distribution over actions.
            V is the value function for the optimal policy.

        """

        def one_step_lookahead(state, V):
            """
            Helper function to calculate the value for all action in a given state.

            Args:
                state: The state to consider (int)
                V: The value to use as an estimator, Vector of length env.nS

            Returns:
                A vector of length env.nA containing the expected value of each action.
            """
            A = np.zeros(env.nA)
            for a in range(env.nA):
                for prob, next_state, reward, done in env.P[state][a]:
                    A[a] += prob * (reward + discount_factor * V[next_state])
            return A

        # Start with a random policy
        policy = np.ones([env.nS, env.nA]) / env.nA
        V_ARR = []
        V_SUM = []
        V = np.zeros(env.nS)

        while True:
            # print("iter)")
            # Evaluate the current policy
            #print("A", time.time())
            V = policy_eval_fn(policy, env, V, discount_factor = discount_factor)

            # Will be set to false if we make any changes to the policy
            policy_stable = True
            #print("B", time.time())
            # For each state...
            for s in range(env.nS):
                # The best action we would take under the current policy
                chosen_a = np.argmax(policy[s])
                # Find the best action by one-step lookahead
                # Ties are resolved arbitarily
                action_values = one_step_lookahead(s, V)
                best_a = np.argmax(action_values)
                # Greedily update the policy
                if chosen_a != best_a:
                    policy_stable = False
                policy[s] = np.eye(env.nA)[best_a]
            #print("C)", time.time())
            # If the policy is stable we've found an optimal policy. Return it
            V_ARR.append(V)
            V_SUM.append(V.sum())
            if policy_stable:
                return policy, V_ARR, V_SUM




    ##################################
    # VI
    ##################################

    np.random.seed(5)

    print("Different size of the problem")
    N_ITERS_vi  = []
    SIZE_vi     = np.arange(10,41,2)
    TIME_ARR_vi = []
    for size in SIZE_vi:
        print(size)
        random_map = generate_random_map(size=size, p=0.8)
        env = gym.make("FrozenLake-v0", desc=random_map, is_slippery = False)
        env.reset()
        time0 = time.time()
        _, V_ARR, _ = value_iteration(env, 1e-10, 0.99)
        time1 = time.time()
        N_ITERS_vi.append(len(V_ARR))
        TIME_ARR_vi.append(time1 - time0)


    fig, ax = plt.subplots()
    ax.plot(SIZE_vi, N_ITERS_vi , color='red', label="Number of iterations", linewidth=2.0, linestyle='-')
    ax.plot(SIZE_vi, N_ITERS_vi , color='red',  marker='o', markersize = 4)

    ax.legend(loc='best', frameon=True)
    plt.grid(False, linestyle='--')
    plt.title('Number of iterations to converge \n (Value iteration)')
    plt.ylabel('Number of iterations')
    plt.xlabel('Size of the problem.')
    plt.tight_layout()
    plt.savefig('plots/lake_vi_iters.png')

    fig, ax = plt.subplots()
    ax.plot(SIZE_vi, TIME_ARR_vi , color='red', label="Clock time", linewidth=2.0, linestyle='-')
    ax.plot(SIZE_vi, TIME_ARR_vi , color='red',  marker='o', markersize = 4)

    ax.legend(loc='best', frameon=True)
    plt.grid(False, linestyle='--')

    plt.title('Clock time necessary to converge \n (Value iteration)')
    plt.ylabel('Clock time')
    plt.xlabel('Size of the problem.')
    plt.tight_layout()
    plt.savefig('plots/lake_vi_time.png')


    #
    # Specific size of the problem
    #

    np.random.seed(5)
    size15 = 15
    random_map = generate_random_map(size=size15, p=0.8)
    print(random_map)
    env1 = gym.make("FrozenLake-v0", desc=random_map, is_slippery = False)
    env1.reset()
    env1.render()

    DELTA_ARR_vi_15, V_ARR_vi_15, policy_vi_15 = value_iteration(env1,  1e-10,  0.99)

    np.random.seed(5)
    size30 = 30

    random_map = generate_random_map(size=size30, p=0.8)
    print(random_map)
    env = gym.make("FrozenLake-v0", desc=random_map, is_slippery = False)
    env.reset()
    env.render()

    DELTA_ARR_vi_30, V_ARR_vi_30, policy_vi_30 = value_iteration(env,  1e-10,  0.99)


    X = np.arange(1,len(DELTA_ARR_vi_30)+1,1)
    fig, ax = plt.subplots()
    ax.plot(X, DELTA_ARR_vi_30 , color='steelblue', label="Delta", linewidth=2.0, linestyle='-')
    ax.plot(X, DELTA_ARR_vi_30 , color='steelblue',  marker='o', markersize = 4)

    ax.legend(loc='best', frameon=True)
    plt.grid(False, linestyle='--')
    plt.title('Delta vs Number of iterations, size = 30 (Value iteration). ')
    plt.ylabel('Delta')
    plt.xlabel('Iterations')
    plt.tight_layout()
    plt.savefig('plots/lake_vi_delta.png')
    print(policy_vi_30.shape)

    print(V_ARR_vi_30[len(V_ARR_vi_30)-1])

    draw_policy(policy_vi_30, V_ARR_vi_30[len(V_ARR_vi_30)-1], env, size30, "plots/lake_vi_policy30.png", "Policy visualization ( Value iteration), size =30.")
    plt.rcParams.update(plt.rcParamsDefault)
    draw_policy(policy_vi_15, V_ARR_vi_15[len(V_ARR_vi_15)-1], env1, size15, "plots/lake_vi_policy15.png", "Policy visualization ( Value iteration), size = 15.")
    plt.rcParams.update(plt.rcParamsDefault)


    ##################################
    # PI
    ##################################


    np.random.seed(5)
    N_ITERS_pi  = []
    SIZE_pi     = np.arange(10,41,2)
    TIME_ARR_pi = []
    for size in SIZE_pi:
        print(size)
        np.random.seed(1111)
        random_map = generate_random_map(size=size, p=0.8)
        env = gym.make("FrozenLake-v0", desc=random_map, is_slippery = False)
        env.reset()
        time0 = time.time()
        policy, V_ARR, V_SUM  = policy_improvement(env, discount_factor = 0.99)
        time1 = time.time()
        N_ITERS_pi.append(len(V_SUM))
        TIME_ARR_pi.append(time1 - time0)

    fig, ax = plt.subplots()
    ax.plot(SIZE_pi, N_ITERS_pi , color='red', label="Number of iterations", linewidth=2.0, linestyle='-')
    ax.plot(SIZE_pi, N_ITERS_pi , color='red',  marker='o', markersize = 2)

    ax.legend(loc='best', frameon=True)
    plt.grid(False, linestyle='--')
    plt.title('Number of iteration to converge for different problem size \n (Policy iteration)')
    plt.ylabel('Number of iterations')
    plt.xlabel('Size of the problem')
    plt.tight_layout()
    plt.savefig('plots/lake_pi_size.png')

    fig, ax = plt.subplots()
    ax.plot(SIZE_pi, TIME_ARR_pi , color='red', label="Clock time", linewidth=2.0, linestyle='-')
    ax.plot(SIZE_pi, TIME_ARR_pi , color='red',  marker='o', markersize = 2)

    ax.legend(loc='best', frameon=True)
    plt.grid(False, linestyle='--')
    plt.title('Time necessary to converge \n (policy iteration)')
    plt.ylabel('Clock time')
    plt.xlabel('Size of the problem')
    plt.tight_layout()
    plt.savefig('plots/lake_pi_time.png')


    np.random.seed(5)
    size15 = 15
    random_map = generate_random_map(size=size15, p=0.8)
    print(random_map)
    env = gym.make("FrozenLake-v0", desc=random_map, is_slippery = False)
    env.reset()
    env.render()
    policy_pi_15, V_ARR_pi_15, V_SUM_pi_15 = policy_improvement(env,  discount_factor = 0.99)


    np.random.seed(5)
    size30 = 30
    random_map = generate_random_map(size=size30, p=0.8)
    print(random_map)
    env1 = gym.make("FrozenLake-v0", desc=random_map, is_slippery = False)
    env1.reset()
    env1.render()
    policy_pi_30, V_ARR_pi_30,V_SUM_pi_30 = policy_improvement(env1,  discount_factor = 0.99)
    print(V_SUM_pi_30)


    plt.tight_layout()
    X = np.arange(1, len(V_SUM_pi_30)+1,1)
    fig, ax = plt.subplots()
    ax.plot(X, V_SUM_pi_30 , color='steelblue', label="Sum of V Values", linewidth=2.0, linestyle='-')
    ax.plot(X, V_SUM_pi_30 , color='steelblue',  marker='o', markersize = 4)

    ax.legend(loc='best', frameon=True)
    plt.grid(False, linestyle='--')
    plt.title('Sum of V values vs Number of iterations, \n( policy-iteration algorithm ). ')
    plt.ylabel('Sum of V Values')
    plt.xlabel('Iterations')
    plt.tight_layout()
    plt.savefig('plots/lake_pi_value.png')

    draw_policy(policy_pi_30, V_ARR_pi_30[len(V_ARR_pi_30)-1], env1, size30, "plots/lake_pi_policy30.png", "Policy visualization (policy iteration), size = 30")
    plt.rcParams.update(plt.rcParamsDefault)
    draw_policy(policy_pi_15, V_ARR_pi_15[len(V_ARR_pi_15)-1], env, size15, "plots/lake_pi_policy15.png", "Policy visualization (policy iteration), size = 15")
    plt.rcParams.update(plt.rcParamsDefault)




    def smooth(in_data, window=10):
        import pandas as pd
        mean = pd.Series(in_data).rolling(window, min_periods=window).mean().to_list()
        return mean


    class Policy(object):
        def __init__(self,
                     policy,
                     q_table,
                     name,
                     iterations_to_converge,
                     time_to_converge=.0):
            self.policy = policy
            self.q_table = q_table
            self.name = name
            self.iterations_to_converge = iterations_to_converge
            self.time_to_converge = time_to_converge

    class Metric(object):
        def __init__(self):
            self.max = -99999
            self.episode = 99999
            self.steps = 0
            self.eps = 0.0

        def track(self, reward, episode=0, step=0, epsilon=0.0):
            if reward == self.max:
                pass
            if reward >= self.max:
                self.max = reward
                self.episode = episode
                self.steps = step
                self.eps = epsilon

        def __str__(self):
            return f"Reward: {self.max}; Episode: {self.episode}; Step: {self.steps}; Epsilon: {self.eps}"


    def Q_learning_train(env, alpha, gamma, episodes, dataset_name, epsilons=0.1, plot=True):
        """Q Learning Algorithm with epsilon greedy
        Args:
            env: Environment
            alpha: Learning Rate --> Extent to which our Q-values are being updated in every iteration.
            gamma: Discount Rate --> How much importance we want to give to future rewards
            epsilon: Probability of selecting random action instead of the 'optimal' action
            episodes: No. of episodes to train on
        Returns:
            Q-learning Trained policy
        """

        """Training the agent"""

        # For plotting metrics
        metrics = {}
        epsilon_decay = 0.99
        epsilons = [x / 10 for x in range(1, 11, 2)] if not epsilons else [epsilons]
        policy = None
        q_table = None
        max_rewards = Metric()


        for epsilon in epsilons:
            print(f"Epsilon {epsilon}")
            metrics[epsilon] = {}
            all_epochs = []
            all_penalties = []
            all_rewards = []
            all_times = []
            experiment_epsilon = epsilon

            # Initialize Q table of 500 x 6 size (500 states and 6 actions) with all zeroes
            q_table = np.zeros([env.observation_space.n, env.action_space.n])

            for i in range(1, episodes + 1):
                st = time.time()
                experiment_epsilon *= epsilon_decay
                # if i % 100 == 0:
                #     print(f"Episode {i}")

                state = env.reset()

                epochs, penalties, reward, total_reward = 0, 0, 0, 0

                while reward != 20 and reward != 5 and epochs <= 10000:
                # while epochs <= 10000:
                    # if epochs % 1000 == 0:
                    #     print(f"Epoch #{epochs}")

                    if random.uniform(0, 1) < experiment_epsilon:
                        action = env.action_space.sample()  # Explore action space randomly
                    else:
                        action = np.argmax(q_table[state])  # Exploit learned values by choosing optimal values

                    next_state, reward, done, info = env.step(action)
                    if done:
                        env.reset()
                        if reward == 1:
                            reward = 5

                        if reward == 0:
                            penalties += 1
                            reward = -5
                    else:
                        if reward == 0:
                            reward = -1

                    old_value = q_table[state, action]
                    next_max = np.max(q_table[next_state])

                    new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
                    q_table[state, action] = new_value

                    if reward == -10:
                        penalties += 1

                    state = next_state
                    epochs += 1
                    total_reward += reward

                max_rewards.track(reward=total_reward, episode=i, epsilon=epsilon, step=epochs)
                all_rewards.append(total_reward)
                all_epochs.append(epochs)
                all_penalties.append(penalties)
                all_times.append(time.time() - st)

            metrics[epsilon]['epochs'] = all_epochs
            metrics[epsilon]['penalties'] = all_penalties
            metrics[epsilon]['rewards'] = all_rewards
            metrics[epsilon]['times'] = all_times

            # Start with a random policy
            policy = np.ones([env.observation_space.n, env.action_space.n]) / env.action_space.n

            for state in range(env.observation_space.n):  # for each states
                best_act = np.argmax(q_table[state])  # find best action
                policy[state] = np.eye(env.action_space.n)[best_act]  # update

        print(max_rewards)

        if plot:
            for metric in ['rewards', 'epochs', 'penalties', 'times']:
                plt.figure()
                title = metric + ' analysis'
                plt.title(title)
                plt.xlabel('Episodes')
                plt.ylabel(metric)
                for k, v in metrics.items():
                    plt.plot(smooth(v[metric], window=100), label=f"epsilon={k}")
                plt.legend()
                filename = 'lake_ql_epsilon_%s' % metric
                chart_path = 'plots/%s.png' % filename
                print(chart_path)
                plt.savefig(chart_path)
                plt.close()
                # processor = Processor()
                # processor.latex_subgraph(dataset=env, fig=filename, caption=metric, filename=filename)

        return metrics, Policy(policy=policy, q_table=q_table, name="Q Learning", iterations_to_converge=0)


    def Q_learning_train_gamma(env, alpha, gamma, episodes, dataset_name, epsilons=0.1, plot=True):
        """Q Learning Algorithm with epsilon greedy
        Args:
            env: Environment
            alpha: Learning Rate --> Extent to which our Q-values are being updated in every iteration.
            gamma: Discount Rate --> How much importance we want to give to future rewards
            epsilon: Probability of selecting random action instead of the 'optimal' action
            episodes: No. of episodes to train on
        Returns:
            Q-learning Trained policy
        """

        """Training the agent"""

        # For plotting metrics
        metrics = {}
        epsilon_decay = 0.99
        # epsilons = [x / 10 for x in range(1, 11, 2)] if not epsilons else [epsilons]
        epsilon = epsilons
        policy = None
        q_table = None
        max_rewards = Metric()

        gammas = [0.9, 0.95, 0.99, 1.0]

        for gamma in gammas:
            print(f"gamma {gamma}")
            metrics[gamma] = {}
            all_epochs = []
            all_penalties = []
            all_rewards = []
            all_times = []
            experiment_epsilon = epsilon

            # Initialize Q table of 500 x 6 size (500 states and 6 actions) with all zeroes
            q_table = np.zeros([env.observation_space.n, env.action_space.n])

            for i in range(1, episodes + 1):
                st = time.time()
                experiment_epsilon *= epsilon_decay
                # if i % 100 == 0:
                #     print(f"Episode {i}")

                state = env.reset()

                epochs, penalties, reward, total_reward = 0, 0, 0, 0

                while reward != 20 and reward != 5 and epochs <= 10000:
                # while epochs <= 10000:
                    # if epochs % 1000 == 0:
                    #     print(f"Epoch #{epochs}")

                    if random.uniform(0, 1) < experiment_epsilon:
                        action = env.action_space.sample()  # Explore action space randomly
                    else:
                        action = np.argmax(q_table[state])  # Exploit learned values by choosing optimal values

                    next_state, reward, done, info = env.step(action)
                    if done:
                        env.reset()
                        if reward == 1:
                            reward = 5

                        if reward == 0:
                            penalties += 1
                            reward = -5
                    else:
                        if reward == 0:
                            reward = -1

                    old_value = q_table[state, action]
                    next_max = np.max(q_table[next_state])

                    new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
                    q_table[state, action] = new_value

                    if reward == -10:
                        penalties += 1

                    state = next_state
                    epochs += 1
                    total_reward += reward

                max_rewards.track(reward=total_reward, episode=i, epsilon=epsilon, step=epochs)
                all_rewards.append(total_reward)
                all_epochs.append(epochs)
                all_penalties.append(penalties)
                all_times.append(time.time() - st)

            metrics[gamma]['epochs'] = all_epochs
            metrics[gamma]['penalties'] = all_penalties
            metrics[gamma]['rewards'] = all_rewards
            metrics[gamma]['times'] = all_times

            # Start with a random policy
            policy = np.ones([env.observation_space.n, env.action_space.n]) / env.action_space.n

            for state in range(env.observation_space.n):  # for each states
                best_act = np.argmax(q_table[state])  # find best action
                policy[state] = np.eye(env.action_space.n)[best_act]  # update

        print(max_rewards)

        if plot:
            for metric in ['rewards', 'epochs', 'penalties', 'times']:
                plt.figure()
                title = metric + ' analysis'
                plt.title(title)
                plt.xlabel('Episodes')
                plt.ylabel(metric)
                for k, v in metrics.items():
                    plt.plot(smooth(v[metric], window=100), label=f"gamma={k}")
                plt.legend()
                filename = 'lake_ql_gamma_%s' % metric
                chart_path = 'plots/%s.png' % filename
                print(chart_path)
                plt.savefig(chart_path)
                plt.close()
                # processor = Processor()
                # processor.latex_subgraph(dataset=env, fig=filename, caption=metric, filename=filename)

        return metrics, Policy(policy=policy, q_table=q_table, name="Q Learning", iterations_to_converge=0)




    def Q_learning_train_decay(env, alpha, gamma, episodes, dataset_name, epsilons=0.1, plot=True):
        """Q Learning Algorithm with epsilon greedy
        Args:
            env: Environment
            alpha: Learning Rate --> Extent to which our Q-values are being updated in every iteration.
            gamma: Discount Rate --> How much importance we want to give to future rewards
            epsilon: Probability of selecting random action instead of the 'optimal' action
            episodes: No. of episodes to train on
        Returns:
            Q-learning Trained policy
        """

        """Training the agent"""

        # For plotting metrics
        metrics = {}
        epsilon_decays = [0.1, 0.3, 0.5, 0.7, 0.9]
        # epsilons = [x / 10 for x in range(1, 11, 2)] if not epsilons else [epsilons]
        epsilon = epsilons
        policy = None
        q_table = None
        max_rewards = Metric()

        gammas = [0.1, 0.3, 0.5, 0.7, 0.9]

        for epsilon_decay in epsilon_decays:
            print(f"epsilon_decay {epsilon_decay}")
            metrics[epsilon_decay] = {}
            all_epochs = []
            all_penalties = []
            all_rewards = []
            all_times = []
            experiment_epsilon = epsilon

            # Initialize Q table of 500 x 6 size (500 states and 6 actions) with all zeroes
            q_table = np.zeros([env.observation_space.n, env.action_space.n])

            for i in range(1, episodes + 1):
                st = time.time()
                experiment_epsilon *= epsilon_decay
                # if i % 100 == 0:
                #     print(f"Episode {i}")

                state = env.reset()

                epochs, penalties, reward, total_reward = 0, 0, 0, 0

                while reward != 20 and reward != 5 and epochs <= 10000:
                # while epochs <= 10000:
                    # if epochs % 1000 == 0:
                    #     print(f"Epoch #{epochs}")

                    if random.uniform(0, 1) < experiment_epsilon:
                        action = env.action_space.sample()  # Explore action space randomly
                    else:
                        action = np.argmax(q_table[state])  # Exploit learned values by choosing optimal values

                    next_state, reward, done, info = env.step(action)
                    if done:
                        env.reset()
                        if reward == 1:
                            reward = 5

                        if reward == 0:
                            penalties += 1
                            reward = -5
                    else:
                        if reward == 0:
                            reward = -1

                    old_value = q_table[state, action]
                    next_max = np.max(q_table[next_state])

                    new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
                    q_table[state, action] = new_value

                    if reward == -10:
                        penalties += 1

                    state = next_state
                    epochs += 1
                    total_reward += reward

                max_rewards.track(reward=total_reward, episode=i, epsilon=epsilon, step=epochs)
                all_rewards.append(total_reward)
                all_epochs.append(epochs)
                all_penalties.append(penalties)
                all_times.append(time.time() - st)

            metrics[epsilon_decay]['epochs'] = all_epochs
            metrics[epsilon_decay]['penalties'] = all_penalties
            metrics[epsilon_decay]['rewards'] = all_rewards
            metrics[epsilon_decay]['times'] = all_times

            # Start with a random policy
            policy = np.ones([env.observation_space.n, env.action_space.n]) / env.action_space.n

            for state in range(env.observation_space.n):  # for each states
                best_act = np.argmax(q_table[state])  # find best action
                policy[state] = np.eye(env.action_space.n)[best_act]  # update

        print(max_rewards)

        if plot:
            for metric in ['rewards', 'epochs', 'penalties', 'times']:
                plt.figure()
                title = metric + ' analysis'
                plt.title(title)
                plt.xlabel('Episodes')
                plt.ylabel(metric)
                for k, v in metrics.items():
                    plt.plot(smooth(v[metric], window=100), label=f"epsilon_decay={k}")
                plt.legend()
                filename = 'lake_ql_epsilon_decay_%s' % metric
                chart_path = 'plots/%s.png' % filename
                print(chart_path)
                plt.savefig(chart_path)
                plt.close()
                # processor = Processor()
                # processor.latex_subgraph(dataset=env, fig=filename, caption=metric, filename=filename)

        return metrics, Policy(policy=policy, q_table=q_table, name="Q Learning", iterations_to_converge=0)





    dataset_name='FrozenLake-v0'


    SIZE     = np.arange(10,41,2)
    rewards_arr = []
    np.random.seed(5)
    for size in SIZE:
        print(size)
        random_map = generate_random_map(size=size, p=0.8)
        env = gym.make("FrozenLake-v0", desc=random_map, is_slippery = False)
        env.reset()
        policy, V_ARR, V_SUM  = policy_improvement(env, discount_factor = 0.99)
        metrics, Q_learn_pol = Q_learning_train(env=env, alpha=0.2, gamma=0.95, episodes=5000,
                                                          epsilons=0.1, dataset_name=dataset_name, plot=False)
        qtable = Q_learn_pol.q_table
        all_rewards = metrics[0.1]['rewards']
        rewards_arr.append(all_rewards)

    stop_ep = []
    for r in rewards_arr:
        r_final = r[-1]
        r_mod = [sum(r[i:i + 80]) / 80 for i in range(len(r) - 80)]
        for idx in range(len(r_mod)):
            if r_final - 0.01 < r_mod[idx] < r_final + 0.01:
                stop_ep.append(idx)
                break

    N_ITERS_ql  = []
    SIZE_ql     = np.arange(10,41,2)
    TIME_ARR_ql = []
    np.random.seed(5)
    for idx, size in enumerate(SIZE_ql):
        print(size)
        random_map = generate_random_map(size=size, p=0.8)
        env = gym.make("FrozenLake-v0", desc=random_map, is_slippery = False)
        env.reset()
        time0 = time.time()
        policy, V_ARR, V_SUM  = policy_improvement(env, discount_factor = 0.99)
        metrics, Q_learn_pol = Q_learning_train(env=env, alpha=0.2, gamma=0.95, episodes=stop_ep[idx],
                                                          epsilons=0.1, dataset_name=dataset_name, plot=False)
        qtable = Q_learn_pol.q_table
        all_rewards = metrics[0.1]['rewards']

        time1 = time.time()
        N_ITERS_ql.append(len(all_rewards))
        rewards_arr.append(all_rewards)
        TIME_ARR_ql.append(time1 - time0)


    fig, ax = plt.subplots()
    ax.plot(SIZE_ql, N_ITERS_ql, color='red', label="Number of iterations", linewidth=2.0, linestyle='-')
    ax.plot(SIZE_ql, N_ITERS_ql , color='red',  marker='o', markersize = 2)

    ax.legend(loc='best', frameon=True)
    plt.grid(False, linestyle='--')
    plt.title('Number of iteration to converge for different problem size \n (Q Learning)')
    plt.ylabel('Number of iterations')
    plt.xlabel('Size of the problem')
    plt.tight_layout()
    plt.savefig('plots/lake_ql_size.png')

    fig, ax = plt.subplots()
    ax.plot(SIZE_ql, TIME_ARR_ql , color='red', label="Clock time", linewidth=2.0, linestyle='-')
    ax.plot(SIZE_ql, TIME_ARR_ql , color='red',  marker='o', markersize = 2)

    ax.legend(loc='best', frameon=True)
    plt.grid(False, linestyle='--')
    plt.title('Time necessary to converge \n (Q Learning)')
    plt.ylabel('Clock time')
    plt.xlabel('Size of the problem')
    plt.tight_layout()
    plt.savefig('plots/lake_ql_time.png')

    ##################################
    episodes=600
    np.random.seed(5)
    size30 = 30
    random_map = generate_random_map(size=size30, p=0.8)
    env = gym.make("FrozenLake-v0", desc=random_map, is_slippery = False)

    metrics30, Q_learn_pol30 = Q_learning_train(env=env, alpha=0.2, gamma=0.95, episodes=episodes,
                                            epsilons=None, dataset_name=dataset_name, plot=True)

    qtable30 = Q_learn_pol30.q_table
    all_rewards30 = metrics30[0.1]['rewards']

    qbest30 = np.empty(env.nS)
    #print(qtable)

    for state in range(env.nS):

        qbest30[state] = np.argmax(qtable30[state,:])

    def moving_average(a, n=3) :
        ret = np.cumsum(a, dtype=float)
        ret[n:] = ret[n:] - ret[:-n]
        return ret[n - 1:] / n

    draw_policy(qtable30, qbest30, env, size30, "plots/lake_ql_policy30.png",
                   "Policy visualization (Q learning), size = 30 ")
    plt.rcParams.update(plt.rcParamsDefault)
    env.close

    ##################################
    episodes=600
    np.random.seed(5)
    size15 = 15
    random_map = generate_random_map(size=size15, p=0.8)
    env = gym.make("FrozenLake-v0", desc=random_map, is_slippery = False)

    metrics15, Q_learn_pol15 = Q_learning_train(env=env, alpha=0.2, gamma=0.95, episodes=episodes,
                                            epsilons=None, dataset_name=dataset_name, plot=False)

    qtable15 = Q_learn_pol15.q_table
    all_rewards15 = metrics15[0.1]['rewards']

    qbest15 = np.empty(env.nS)
    #print(qtable)

    for state in range(env.nS):

        qbest15[state] = np.argmax(qtable15[state,:])

    def moving_average(a, n=3) :
        ret = np.cumsum(a, dtype=float)
        ret[n:] = ret[n:] - ret[:-n]
        return ret[n - 1:] / n

    draw_policy(qtable15, qbest15, env, size15, "plots/lake_ql_policy15.png",
                   "Policy visualization (Q learning), size = 15 ")
    plt.rcParams.update(plt.rcParamsDefault)
    env.close




    fig, ax = plt.subplots()
    ax.plot(SIZE_vi, N_ITERS_vi , color='red', label="Value Iteration", linewidth=2.0, linestyle='-')
    ax.plot(SIZE_vi, N_ITERS_vi , color='red',  marker='o', markersize = 4)
    ax.plot(SIZE_pi, N_ITERS_pi , color='blue', label="Policy Iteration", linewidth=2.0, linestyle='-')
    ax.plot(SIZE_pi, N_ITERS_pi , color='blue',  marker='o', markersize = 4)
    ax.plot(SIZE_ql, N_ITERS_ql , color='green', label="Q Learning", linewidth=2.0, linestyle='-')
    ax.plot(SIZE_ql, N_ITERS_ql , color='green',  marker='o', markersize = 4)
    ax.set_yscale('log')
    ax.legend(loc='best', frameon=True)
    plt.grid(False, linestyle='--')
    plt.title('Number of iterations to converge \n (Value iteration)')
    plt.ylabel('Number of iterations')
    plt.xlabel('Size of the problem.')
    plt.tight_layout()
    plt.savefig('plots/lake_all_iters.png')





    fig, ax = plt.subplots()
    ax.plot(SIZE_vi, TIME_ARR_vi , color='red', label="Value Iteration", linewidth=2.0, linestyle='-')
    ax.plot(SIZE_vi, TIME_ARR_vi , color='red',  marker='o', markersize = 4)
    ax.plot(SIZE_pi, TIME_ARR_pi , color='blue', label="Policy IterationPolicy Iteration", linewidth=2.0, linestyle='-')
    ax.plot(SIZE_pi, TIME_ARR_pi , color='blue',  marker='o', markersize = 4)
    ax.plot(SIZE_ql, TIME_ARR_ql , color='green', label="Q Learning", linewidth=2.0, linestyle='-')
    ax.plot(SIZE_ql, TIME_ARR_ql , color='green',  marker='o', markersize = 4)

    ax.legend(loc='best', frameon=True)
    plt.grid(False, linestyle='--')

    plt.title('Clock time necessary to converge \n (Value iteration)')
    plt.ylabel('Clock time')
    plt.xlabel('Size of the problem.')
    plt.tight_layout()
    plt.savefig('plots/lake_all_time.png')









    ##################################
    episodes=600
    np.random.seed(5)
    size30 = 30
    random_map = generate_random_map(size=size30, p=0.8)
    env = gym.make("FrozenLake-v0", desc=random_map, is_slippery = False)

    metrics30, Q_learn_pol30 = Q_learning_train_gamma(env=env, alpha=0.2, gamma=0.95, episodes=episodes,
                                            epsilons=0.1, dataset_name=dataset_name, plot=True)





    ##################################
    episodes=600
    np.random.seed(5)
    size30 = 30
    random_map = generate_random_map(size=size30, p=0.8)
    env = gym.make("FrozenLake-v0", desc=random_map, is_slippery = False)

    metrics30, Q_learn_pol30 = Q_learning_train_decay(env=env, alpha=0.2, gamma=0.95, episodes=episodes,
                                            epsilons=0.5, dataset_name=dataset_name, plot=True)








