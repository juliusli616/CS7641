import numpy as np
import gym
from gym.envs.toy_text.frozen_lake import generate_random_map
import matplotlib.pyplot as plt
import time
import mdptoolbox
import mdptoolbox, mdptoolbox.example
import seaborn as sns
import math


def run():

    def draw_policy(policy, V, env, size, file_name, plt_title):
        import matplotlib.pyplot as plt


        #desc = env.desc.astype('U')

        policy_flat = np.argmax(policy, axis=1)
        V           = policy_flat
        policy_grid = np.copy(policy_flat)
        sns.set()

        policy_list = np.chararray((size), unicode=True)

        policy_list[np.where(policy_grid == 0)] = 'Wait'
        policy_list[np.where(policy_grid == 1)] = 'Cut'

        a4_dims = (3, 9)


        fig, ax = plt.subplots(figsize = a4_dims)

        V= V.reshape((size,1))
        policy_list = policy_list.reshape((size,1))
        print(V.shape,policy_list.shape)


        sns.heatmap(V, annot=policy_list, fmt='', ax=ax)
        plt.title(plt_title)
        plt.tight_layout()
        plt.savefig(file_name)
        plt.close()

        return True

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
        V_SUM = []
        while True:
            delta = 0
            for s in range(env.nS):
                # Do a one-step lookahead to find the best action
                A = one_step_lookahead(s, V)
                best_action_value = np.max(A)
                # Calculate delta across all states seen so farg
                delta = max(delta, np.abs(best_action_value - V[s]))
                # Update the value function. Ref: Sutton book eq. 4.10.
                V[s] = best_action_value
                # Check if we can stop
            DELTA_ARR.append(delta)
            V_ARR.append(V)
            V_SUM.append(V.sum())
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
        return DELTA_ARR, V_ARR,V_SUM, policy


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

    class my_env:
        def __init__(self, n_states, n_actions):
            self.P =  [[[] for x in range(n_actions)] for y in range(n_states)]
            self.nS = n_states
            self.nA = n_actions

    def my_forest(size):
        np.random.seed(5)
        n_states  = size
        n_actions = 2
        P, R = mdptoolbox.example.forest(S=n_states, r1=4,r2=50, p=0.6)

        env = my_env(n_states, n_actions)
        for action in range(0, n_actions):
            for state in range(0, n_states):
                for state_slash in range(0,n_states):
                    reward = R[state][action]
                    env.P[state][action].append([P[action][state][state_slash], state_slash, reward, False])
        return env


    ########################
    # Vi
    ########################

    N_ITERS_vi  = []
    SIZE_vi     = np.arange(3,15,1)
    TIME_ARR_vi = []
    for size in SIZE_vi:
        print(size)
        np.random.seed(5)
        env = my_forest(size)
        # print("nStates  ", env.nS)
        time0 = time.time()
        DELTA_ARR, V_ARR, V_SUM, policy = value_iteration(env, 1e-15,  0.999)
        time1 = time.time()
        N_ITERS_vi.append(len(V_SUM))
        TIME_ARR_vi.append(time1 - time0)
    # """
    fig, ax = plt.subplots()
    ax.plot(SIZE_vi, N_ITERS_vi , color='red', label="Number of iterations", linewidth=2.0, linestyle='-')
    ax.plot(SIZE_vi, N_ITERS_vi , color='red',  marker='o', markersize = 4)

    ax.legend(loc='best', frameon=True)
    plt.grid(False, linestyle='--')
    plt.title('Number of iterations to converge \n (Value iteration)')
    plt.ylabel('Number of iterations')
    plt.xlabel('Size of the problem.')
    plt.tight_layout()
    plt.savefig('plots/forest_vi_iters.png')
    # """


    fig, ax = plt.subplots()
    ax.plot(SIZE_vi, TIME_ARR_vi , color='red', label="Clock time", linewidth=2.0, linestyle='-')
    ax.plot(SIZE_vi, TIME_ARR_vi , color='red',  marker='o', markersize = 4)

    ax.legend(loc='best', frameon=True)
    plt.grid(False, linestyle='--')

    plt.title('Clock time necessary to converge \n (Value iteration)')
    plt.ylabel('Clock time')
    plt.xlabel('Size of the problem.')
    plt.tight_layout()
    plt.savefig('plots/forest_vi_time.png')



    ########################
    np.random.seed(5)
    size = 10
    env = my_forest(size)

    DELTA_ARR_vi99, V_ARR_vi99, V_SUM_vi99, policy_vi99 = value_iteration(env,  1e-15,  0.99)
    DELTA_ARR_vi90, V_ARR_vi90, V_SUM_vi90, policy_vi90 = value_iteration(env,  1e-15,  0.90)
    DELTA_ARR_vi70, V_ARR_vi70, V_SUM_vi70, policy_vi70 = value_iteration(env,  1e-15,  0.70)

    X99 = np.arange(1,len(V_SUM_vi99)+1,1)
    X90 = np.arange(1,len(V_SUM_vi90)+1,1)
    X70 = np.arange(1,len(V_SUM_vi70)+1,1)
    fig, ax = plt.subplots()
    ax.plot(X99, V_SUM_vi99 , color='steelblue', label="gamma = 0.99", linewidth=2.0, linestyle='-')
    ax.plot(X90, V_SUM_vi90 , color='red', label="gamma = 0.90", linewidth=2.0, linestyle='-')
    ax.plot(X70, V_SUM_vi70 , color='purple', label="gamma = 0.70", linewidth=2.0, linestyle='-')
    ax.legend(loc='best', frameon=True)
    plt.grid(False, linestyle='--')
    plt.title('Sum of V vlaues vs Number of iterations, size = 10 (Value iteration). ')
    plt.ylabel('Sum of V Values')
    plt.xlabel('Iterations')
    plt.xlim((0,500))
    plt.tight_layout()
    plt.savefig('plots/forest_vi_V.png')

    draw_policy(policy_vi99, V_ARR_vi99[len(V_ARR_vi99)-1], env, size, "plots/forest_vi_policy99.png", "Policy visualization \n(value iteration), size = 10")
    plt.rcParams.update(plt.rcParamsDefault)





    ########################
    # PI
    ########################

    np.random.seed(5)

    #
    # Different sizes
    #
    N_ITERS_pi  = []
    SIZE_pi     = np.arange(3,15,1)
    TIME_ARR_pi = []
    for size in SIZE_pi:
        print(size)
        np.random.seed(5)
        env = my_forest(size)
        # print("nStates  ", env.nS)
        time0 = time.time()
        policy, V_ARR, V_SUM = policy_improvement(env, discount_factor= 0.99)
        time1 = time.time()
        N_ITERS_pi.append(len(V_ARR))
        TIME_ARR_pi.append(time1 - time0)
    # """
    fig, ax = plt.subplots()
    ax.plot(SIZE_pi, N_ITERS_pi , color='red', label="Number of iterations", linewidth=2.0, linestyle='-')
    ax.plot(SIZE_pi, N_ITERS_pi , color='red',  marker='o', markersize = 4)

    ax.legend(loc='best', frameon=True)
    plt.grid(False, linestyle='--')
    plt.title('Number of iteration to converge for different problem size \n (Policy iteration)')
    plt.ylabel('Number of iterations')
    plt.xlabel('Size of the problem')
    plt.tight_layout()
    plt.savefig('plots/forest_pi_size.png')
    # """

    fig, ax = plt.subplots()
    ax.plot(SIZE_pi, TIME_ARR_pi , color='red', label="Clock time", linewidth=2.0, linestyle='-')
    ax.plot(SIZE_pi, TIME_ARR_pi , color='red',  marker='o', markersize = 4)

    ax.legend(loc='best', frameon=True)
    plt.grid(False, linestyle='--')
    plt.title('Time necessary to converge \n (policy iteration)')
    plt.ylabel('Clock time')
    plt.xlabel('Size of the problem')
    plt.tight_layout()
    plt.savefig('plots/forest_pi_time.png')








    np.random.seed(5)

    size = 10
    env = my_forest(size)


    policy_pi99,V_ARR_pi99, V_SUM_pi99 = policy_improvement(env,  discount_factor = 0.99)
    policy_pi90,V_ARR_pi90, V_SUM_pi90 = policy_improvement(env,  discount_factor = 0.90)
    policy_pi70,V_ARR_pi70, V_SUM_pi70 = policy_improvement(env,  discount_factor = 0.70)



    X99 = np.arange(1,len(V_SUM_pi99)+1,1)
    X90 = np.arange(1,len(V_SUM_pi90)+1,1)
    X70 = np.arange(1,len(V_SUM_pi70)+1,1)
    fig, ax = plt.subplots()
    ax.plot(X99, V_SUM_pi99 , color='steelblue', label="gamma = 0.99", linewidth=2.0, linestyle='-')
    ax.plot(X90, V_SUM_pi90 , color='red', label="gamma = 0.9 ", linewidth=2.0, linestyle='-')
    ax.plot(X70, V_SUM_pi70 , color='purple', label="gamma = 0.7", linewidth=2.0, linestyle='-')
    ax.legend(loc='best', frameon=True)
    plt.grid(False, linestyle='--')
    plt.title('Sum of V values vs Number of iterations, size = 10 (Policy iteration). ')
    plt.ylabel('Sum of V values')
    plt.xlabel('Iterations')
    plt.tight_layout()
    plt.savefig('plots/forest_pi_plot.png')

    draw_policy(policy_pi99, V_ARR_pi99[len(V_ARR_pi99)-1], env, size, "plots/forest_pi_policy99.png", "Policy visualization \n(policy iteration), size = 10")
    plt.rcParams.update(plt.rcParamsDefault)





    def Q_learning(env, total_episodes, learning_rate, max_steps, gamma, epsilon,
               max_epsilon, min_epsilon, decay_rate, verbose= True):
        qtable = np.zeros((env.nS, env.nA))
        time0         = time.time()
        clean_episode = True
        episode_length = total_episodes
        time_length    = 10e6
        for episode in range(total_episodes):
            # Reset the environment
            state = np.random.randint(env.nS, size=1)[0]

            step = 0
            done = False
            total_rewards = 0
            REWARD_ARR = []
            for step in range(max_steps):
                # 3. Choose an action a in the current world state (s)
                ## First we randomize a number
                exp_exp_tradeoff = random.uniform(0, 1)
                ## If this number > greater than epsilon --> exploitation (taking the biggest Q value for this state)
                if exp_exp_tradeoff > epsilon:
                    action = np.argmax(qtable[state, :])
                # Else doing a random choice --> exploration
                else:
                    action = env.random_action()
                # Take the action (a) and observe the outcome state(s') and reward (r)
                new_state, reward, done, info = env.new_state(state, action)
                if reward > 0:
                    total_rewards += reward
                # Update Q(s,a):= Q(s,a) + lr [R(s,a) + gamma * max Q(s',a') - Q(s,a)]
                # qtable[new_state,:] : all the actions we can take from new state
                qtable[state, action] = qtable[state, action] + learning_rate * (
                        reward + gamma * np.max(qtable[new_state, :]) - qtable[state, action])
                # Our new state is state
                state = new_state


            # Reduce epsilon (because we need less and less exploration)
            epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay_rate * episode)

            if verbose:
                if math.fmod(episode,100)==0:
                    print(episode, total_rewards, epsilon, decay_rate)
            rewards.append(total_rewards)

            if np.array(rewards)[-100:].mean() > 995 and clean_episode==True:
                episode_length = episode
                time_length = time.time() - time0
                clean_episode = False
                break

        return time_length, episode_length, qtable, rewards


    class my_env:
        def __init__(self, n_states, n_actions):
            self.P =  [[[] for x in range(n_actions)] for y in range(n_states)]
            self.nS = n_states
            self.nA = n_actions

        def new_state(self,state,action):
            listy = env.P[state][action]
            p = []
            for item in listy:
                p.append(item[0])
            p = np.array(p)
            #print(p,state)
            chosen_index = np.random.choice(env.nS, 1, p=p)[0]
            chosen_item = listy[chosen_index]
            return chosen_item[1],chosen_item[0], chosen_item[2],chosen_item[3]

        def random_action(self):
            action = np.random.randint(2, size=1)[0]
            return action

    def my_forest(size):
        np.random.seed(1111)
        n_states  = size
        n_actions = 2
        P, R = mdptoolbox.example.forest(S=n_states, r1=4,r2=50, p=0.6)

        env = my_env(n_states, n_actions)
        for action in range(0, n_actions):
            for state in range(0, n_states):
                for state_slash in range(0,n_states):
                    reward = R[state][action]
                    env.P[state][action].append([P[action][state][state_slash], state_slash, reward, False])
        return env


    size = 10
    env = my_forest(size)


    import random
    action_size = env.nA
    state_size = env.nS
    qtable = np.zeros((state_size, action_size))

    total_episodes = 1000
    learning_rate = 0.1         # Learning rate
    max_steps = 1000           # Max steps per episode
    gamma = 0.9          # Discounting rate

    # Exploration parameters

    epsilon = 0.1                 # Exploration rate
    max_epsilon = 1.0             # Exploration probability at start
    min_epsilon = 10e-9           # Minimum exploration probability
    decay_rate = 0.1       # Exponential decay rate for exploration prob

    # List of rewards
    rewards = []

    def moving_average(a, n=3) :
        ret = np.cumsum(a, dtype=float)
        ret[n:] = ret[n:] - ret[:-n]
        return ret[n - 1:] / n

    time00 = time.time()
    time_length, episode_length, qtable,rewards = Q_learning(env, total_episodes, learning_rate, max_steps, gamma, epsilon,
                                                      max_epsilon, min_epsilon, decay_rate, verbose = True)
    time01 = time.time()

    qbest = np.empty(env.nS)
    #print(qtable)
    for state in range(env.nS):
        qbest[state] = np.argmax(qtable[state,:])
    print("Score over time: " + str(sum(rewards) / total_episodes))
    #print(qtable)
    fig, ax = plt.subplots()
    rewards_moving1 = moving_average(rewards,10)
    X1 = np.arange(1,len(rewards_moving1)+1,1)



    action_size = env.nA
    state_size = env.nS
    qtable = np.zeros((state_size, action_size))

    total_episodes = 1000
    learning_rate = 0.1         # Learning rate
    max_steps = 1000           # Max steps per episode
    gamma = 0.9          # Discounting rate

    # Exploration parameters

    epsilon = 0.1                 # Exploration rate
    max_epsilon = 1.0             # Exploration probability at start
    min_epsilon = 10e-9           # Minimum exploration probability
    decay_rate = 0.3       # Exponential decay rate for exploration prob

    # List of rewards
    rewards = []

    def moving_average(a, n=3) :
        ret = np.cumsum(a, dtype=float)
        ret[n:] = ret[n:] - ret[:-n]
        return ret[n - 1:] / n

    time10 = time.time()
    time_length, episode_length, qtable,rewards = Q_learning(env, total_episodes, learning_rate, max_steps, gamma, epsilon,
                                                      max_epsilon, min_epsilon, decay_rate, verbose = True)
    time11 = time.time()
    qbest = np.empty(env.nS)
    #print(qtable)
    for state in range(env.nS):
        qbest[state] = np.argmax(qtable[state,:])
    print("Score over time: " + str(sum(rewards) / total_episodes))
    #print(qtable)
    fig, ax = plt.subplots()
    rewards_moving2 = moving_average(rewards,10)
    X2 = np.arange(1,len(rewards_moving2)+1,1)





    action_size = env.nA
    state_size = env.nS
    qtable = np.zeros((state_size, action_size))

    total_episodes = 1000
    learning_rate = 0.1         # Learning rate
    max_steps = 1000           # Max steps per episode
    gamma = 0.9          # Discounting rate

    # Exploration parameters

    epsilon = 0.1                 # Exploration rate
    max_epsilon = 1.0             # Exploration probability at start
    min_epsilon = 10e-9           # Minimum exploration probability
    decay_rate = 0.5       # Exponential decay rate for exploration prob

    # List of rewards
    rewards = []

    def moving_average(a, n=3) :
        ret = np.cumsum(a, dtype=float)
        ret[n:] = ret[n:] - ret[:-n]
        return ret[n - 1:] / n

    time20 = time.time()
    time_length, episode_length, qtable,rewards = Q_learning(env, total_episodes, learning_rate, max_steps, gamma, epsilon,
                                                      max_epsilon, min_epsilon, decay_rate, verbose = True)
    time21 = time.time()
    qbest = np.empty(env.nS)
    #print(qtable)
    for state in range(env.nS):
        qbest[state] = np.argmax(qtable[state,:])
    print("Score over time: " + str(sum(rewards) / total_episodes))
    #print(qtable)
    fig, ax = plt.subplots()
    rewards_moving3 = moving_average(rewards,10)
    X3 = np.arange(1,len(rewards_moving3)+1,1)




    action_size = env.nA
    state_size = env.nS
    qtable = np.zeros((state_size, action_size))

    total_episodes = 1000
    learning_rate = 0.1         # Learning rate
    max_steps = 1000           # Max steps per episode
    gamma = 0.9          # Discounting rate

    # Exploration parameters

    epsilon = 0.1                 # Exploration rate
    max_epsilon = 1.0             # Exploration probability at start
    min_epsilon = 10e-9           # Minimum exploration probability
    decay_rate = 0.7       # Exponential decay rate for exploration prob

    # List of rewards
    rewards = []

    def moving_average(a, n=3) :
        ret = np.cumsum(a, dtype=float)
        ret[n:] = ret[n:] - ret[:-n]
        return ret[n - 1:] / n

    time30 = time.time()
    time_length, episode_length, qtable,rewards = Q_learning(env, total_episodes, learning_rate, max_steps, gamma, epsilon,
                                                      max_epsilon, min_epsilon, decay_rate, verbose = True)
    time31 = time.time()
    qbest = np.empty(env.nS)
    #print(qtable)
    for state in range(env.nS):
        qbest[state] = np.argmax(qtable[state,:])
    print("Score over time: " + str(sum(rewards) / total_episodes))
    #print(qtable)
    fig, ax = plt.subplots()
    rewards_moving4 = moving_average(rewards,10)
    X4 = np.arange(1,len(rewards_moving4)+1,1)



    action_size = env.nA
    state_size = env.nS
    qtable = np.zeros((state_size, action_size))

    total_episodes = 1000
    learning_rate = 0.1         # Learning rate
    max_steps = 1000           # Max steps per episode
    gamma = 0.9          # Discounting rate

    # Exploration parameters

    epsilon = 0.1                 # Exploration rate
    max_epsilon = 1.0             # Exploration probability at start
    min_epsilon = 10e-9           # Minimum exploration probability
    decay_rate = 0.9       # Exponential decay rate for exploration prob

    # List of rewards
    rewards = []

    def moving_average(a, n=3) :
        ret = np.cumsum(a, dtype=float)
        ret[n:] = ret[n:] - ret[:-n]
        return ret[n - 1:] / n

    time40 = time.time()
    time_length, episode_length, qtable,rewards = Q_learning(env, total_episodes, learning_rate, max_steps, gamma, epsilon,
                                                      max_epsilon, min_epsilon, decay_rate, verbose = True)
    time41 = time.time()
    qbest = np.empty(env.nS)
    #print(qtable)
    for state in range(env.nS):
        qbest[state] = np.argmax(qtable[state,:])
    print("Score over time: " + str(sum(rewards) / total_episodes))
    #print(qtable)
    fig, ax = plt.subplots()
    rewards_moving5 = moving_average(rewards,10)
    X5 = np.arange(1,len(rewards_moving5)+1,1)






    ax.plot(X1, rewards_moving1, label="Mean Reward (decay rate = 0.1)", linewidth=2.0, linestyle='-')
    ax.plot(X2, rewards_moving2, label="Mean Reward (decay rate = 0.3)", linewidth=2.0, linestyle='-')
    ax.plot(X3, rewards_moving3, label="Mean Reward (decay rate = 0.5)", linewidth=2.0, linestyle='-')
    ax.plot(X4, rewards_moving4, label="Mean Reward (decay rate = 0.7)", linewidth=2.0, linestyle='-')
    ax.plot(X5, rewards_moving5, label="Mean Reward (decay rate = 0.9)", linewidth=2.0, linestyle='-')
    ax.legend(loc='best', frameon=True)
    plt.grid(False, linestyle='--')
    plt.title('Mean Reward vs number of episodes, Q learning (grid size = 10).')
    plt.ylabel('Mean Reward (over 10 episodes')
    plt.xlabel(' Number of episodes')
    plt.savefig('plots/forest_ql_decay_rate.png')

    print("eps = 0.1, num_episodes: ", time01-time00, len(X1))
    print("eps = 0.3, num_episodes: ", time11-time10, len(X2))
    print("eps = 0.5, num_episodes: ", time21-time20, len(X3))
    print("eps = 0.7, num_episodes: ", time31-time30, len(X4))
    print("eps = 0.9, num_episodes: ", time41-time40, len(X5))












    action_size = env.nA
    state_size = env.nS
    qtable = np.zeros((state_size, action_size))

    total_episodes = 15
    learning_rate = 0.1         # Learning rate
    max_steps = 1000           # Max steps per episode
    gamma = 0.9          # Discounting rate

    # Exploration parameters

    epsilon = 0.1                   # Exploration rate
    max_epsilon = 1.0             # Exploration probability at start
    min_epsilon = 10e-9           # Minimum exploration probability
    decay_rate = 0.95               # Exponential decay rate for exploration prob

    # List of rewards
    rewards = []

    def moving_average(a, n=3) :
        ret = np.cumsum(a, dtype=float)
        ret[n:] = ret[n:] - ret[:-n]
        return ret[n - 1:] / n

    time_length, episode_length, qtable,rewards = Q_learning(env, total_episodes, learning_rate, max_steps, gamma, epsilon,
                                                      max_epsilon, min_epsilon, decay_rate, verbose = True)

    qbest = np.empty(env.nS)
    #print(qtable)
    for state in range(env.nS):
        qbest[state] = np.argmax(qtable[state,:])
    print("Score over time: " + str(sum(rewards) / total_episodes))
    #print(qtable)
    fig, ax = plt.subplots()
    rewards_moving1 = moving_average(rewards,5)
    X1 = np.arange(1,len(rewards_moving1)+1,1)


    # print(qtable)





    action_size = env.nA
    state_size = env.nS
    qtable = np.zeros((state_size, action_size))

    total_episodes = 15
    learning_rate = 0.1         # Learning rate
    max_steps = 1000           # Max steps per episode
    gamma = 0.9          # Discounting rate

    # Exploration parameters

    epsilon = 0.3                  # Exploration rate
    max_epsilon = 1.0             # Exploration probability at start
    min_epsilon = 10e-9           # Minimum exploration probability
    decay_rate = 0.95               # Exponential decay rate for exploration prob

    # List of rewards
    rewards = []

    def moving_average(a, n=3) :
        ret = np.cumsum(a, dtype=float)
        ret[n:] = ret[n:] - ret[:-n]
        return ret[n - 1:] / n

    time_length, episode_length, qtable,rewards = Q_learning(env, total_episodes, learning_rate, max_steps, gamma, epsilon,
                                                      max_epsilon, min_epsilon, decay_rate, verbose = True)

    qbest = np.empty(env.nS)
    #print(qtable)
    for state in range(env.nS):
        qbest[state] = np.argmax(qtable[state,:])
    print("Score over time: " + str(sum(rewards) / total_episodes))
    #print(qtable)
    fig, ax = plt.subplots()
    rewards_moving2 = moving_average(rewards,5)
    X2 = np.arange(1,len(rewards_moving2)+1,1)


    # print(qtable)






    action_size = env.nA
    state_size = env.nS
    qtable = np.zeros((state_size, action_size))

    total_episodes = 15
    learning_rate = 0.1         # Learning rate
    max_steps = 1000           # Max steps per episode
    gamma = 0.9          # Discounting rate

    # Exploration parameters

    epsilon = 0.5                  # Exploration rate
    max_epsilon = 1.0             # Exploration probability at start
    min_epsilon = 10e-9           # Minimum exploration probability
    decay_rate = 0.95               # Exponential decay rate for exploration prob

    # List of rewards
    rewards = []

    def moving_average(a, n=3) :
        ret = np.cumsum(a, dtype=float)
        ret[n:] = ret[n:] - ret[:-n]
        return ret[n - 1:] / n

    time_length, episode_length, qtable,rewards = Q_learning(env, total_episodes, learning_rate, max_steps, gamma, epsilon,
                                                      max_epsilon, min_epsilon, decay_rate, verbose = True)

    qbest = np.empty(env.nS)
    #print(qtable)
    for state in range(env.nS):
        qbest[state] = np.argmax(qtable[state,:])
    print("Score over time: " + str(sum(rewards) / total_episodes))
    #print(qtable)
    fig, ax = plt.subplots()
    rewards_moving3 = moving_average(rewards,5)
    X3 = np.arange(1,len(rewards_moving3)+1,1)


    # print(qtable)




    action_size = env.nA
    state_size = env.nS
    qtable = np.zeros((state_size, action_size))

    total_episodes = 15
    learning_rate = 0.1         # Learning rate
    max_steps = 1000           # Max steps per episode
    gamma = 0.9          # Discounting rate

    # Exploration parameters

    epsilon = 0.7                  # Exploration rate
    max_epsilon = 1.0             # Exploration probability at start
    min_epsilon = 10e-9           # Minimum exploration probability
    decay_rate = 0.95               # Exponential decay rate for exploration prob

    # List of rewards
    rewards = []

    def moving_average(a, n=3) :
        ret = np.cumsum(a, dtype=float)
        ret[n:] = ret[n:] - ret[:-n]
        return ret[n - 1:] / n

    time_length, episode_length, qtable,rewards = Q_learning(env, total_episodes, learning_rate, max_steps, gamma, epsilon,
                                                      max_epsilon, min_epsilon, decay_rate, verbose = True)

    qbest = np.empty(env.nS)
    #print(qtable)
    for state in range(env.nS):
        qbest[state] = np.argmax(qtable[state,:])
    print("Score over time: " + str(sum(rewards) / total_episodes))
    #print(qtable)
    fig, ax = plt.subplots()
    rewards_moving4 = moving_average(rewards,5)
    X4 = np.arange(1,len(rewards_moving4)+1,1)


    # print(qtable)





    action_size = env.nA
    state_size = env.nS
    qtable = np.zeros((state_size, action_size))

    total_episodes = 15
    learning_rate = 0.1         # Learning rate
    max_steps = 1000           # Max steps per episode
    gamma = 0.9          # Discounting rate

    # Exploration parameters

    epsilon = 0.9                  # Exploration rate
    max_epsilon = 1.0             # Exploration probability at start
    min_epsilon = 10e-9           # Minimum exploration probability
    decay_rate = 0.95               # Exponential decay rate for exploration prob

    # List of rewards
    rewards = []

    def moving_average(a, n=3) :
        ret = np.cumsum(a, dtype=float)
        ret[n:] = ret[n:] - ret[:-n]
        return ret[n - 1:] / n

    time_length, episode_length, qtable,rewards = Q_learning(env, total_episodes, learning_rate, max_steps, gamma, epsilon,
                                                      max_epsilon, min_epsilon, decay_rate, verbose = True)

    qbest = np.empty(env.nS)
    #print(qtable)
    for state in range(env.nS):
        qbest[state] = np.argmax(qtable[state,:])
    print("Score over time: " + str(sum(rewards) / total_episodes))
    #print(qtable)
    fig, ax = plt.subplots()
    rewards_moving5 = moving_average(rewards,5)
    X5 = np.arange(1,len(rewards_moving5)+1,1)


    # print(qtable)






    #
    # action_size = env.nA
    # state_size = env.nS
    # qtable = np.zeros((state_size, action_size))
    #
    # total_episodes = 1000
    # learning_rate = 0.1         # Learning rate
    # max_steps = 1000           # Max steps per episode
    # gamma = 0.9          # Discounting rate
    #
    # # Exploration parameters
    #
    # epsilon = 0.02                  # Exploration rate
    # max_epsilon = 0.02              # Exploration probability at start
    # min_epsilon = 0.02              # Minimum exploration probability
    # decay_rate = 1.0               # Exponential decay rate for exploration prob
    #
    # # List of rewards
    # rewards = []
    #
    # def moving_average(a, n=3) :
    #     ret = np.cumsum(a, dtype=float)
    #     ret[n:] = ret[n:] - ret[:-n]
    #     return ret[n - 1:] / n
    #
    # time_length, episode_length, qtable,rewards = Q_learning(env, total_episodes, learning_rate, max_steps, gamma, epsilon,
    #                                                   max_epsilon, min_epsilon, decay_rate, verbose = True)
    #
    # qbest = np.empty(env.nS)
    # #print(qtable)
    # for state in range(env.nS):
    #     qbest[state] = np.argmax(qtable[state,:])
    # print("Score over time: " + str(sum(rewards) / total_episodes))
    # #print(qtable)
    # fig, ax = plt.subplots()
    # rewards_moving2 = moving_average(rewards,10)
    # X2 = np.arange(1,len(rewards_moving2)+1,1)








    ax.plot(X1, rewards_moving1, label="Mean Reward (epsilon = 0.1)", linewidth=2.0, linestyle='-')
    ax.plot(X2, rewards_moving2, label="Mean Reward (epsilon = 0.3)", linewidth=2.0, linestyle='-')
    ax.plot(X3, rewards_moving3, label="Mean Reward (epsilon = 0.5)", linewidth=2.0, linestyle='-')
    ax.plot(X4, rewards_moving4, label="Mean Reward (epsilon = 0.7)", linewidth=2.0, linestyle='-')
    ax.plot(X5, rewards_moving5, label="Mean Reward (epsilon = 0.9)", linewidth=2.0, linestyle='-')
    ax.legend(loc='best', frameon=True)
    plt.grid(False, linestyle='--')
    plt.title('Mean Reward vs number of episodes, Q learning (grid size = 10).')
    plt.ylabel('Mean Reward (over 10 episodes')
    plt.xlabel(' Number of episodes')
    plt.savefig('plots/forest_ql_decay_epsilon.png')


    # print(qtable)



    np.random.seed(5)
    N_ITERS_ql  = []
    SIZE_ql     = np.arange(3,15,1)
    TIME_ARR_ql = []

    for size in SIZE_ql:
        print(size)
        action_size = env.nA
        state_size = env.nS
        qtable = np.zeros((state_size, action_size))

        total_episodes = 1000
        learning_rate = 0.1  # Learning rate
        max_steps = 1000  # Max steps per episode
        gamma = 0.9  # Discounting rate

        # Exploration parameters

        epsilon = 0.5  # Exploration rate
        max_epsilon = 1.0  # Exploration probability at start
        min_epsilon = 10e-9  # Minimum exploration probability
        decay_rate = 0.95  # Exponential decay rate for exploration prob

        # List of rewards
        rewards = []


        def moving_average(a, n=3):
            ret = np.cumsum(a, dtype=float)
            ret[n:] = ret[n:] - ret[:-n]
            return ret[n - 1:] / n


        time0 = time.time()
        time_length, episode_length, qtable, rewards = Q_learning(env, total_episodes, learning_rate, max_steps, gamma,
                                                                  epsilon,
                                                                  max_epsilon, min_epsilon, decay_rate, verbose=True)
        time1 = time.time()

        N_ITERS_ql.append(len(rewards))
        TIME_ARR_ql.append(time1 - time0)

    fig, ax = plt.subplots()
    ax.plot(SIZE_ql, N_ITERS_ql , color='red', label="Number of iterations", linewidth=2.0, linestyle='-')
    ax.plot(SIZE_ql, N_ITERS_ql , color='red',  marker='o', markersize = 2)

    ax.legend(loc='best', frameon=True)
    plt.grid(False, linestyle='--')
    plt.title('Number of iteration to converge for different problem size \n (Policy iteration)')
    plt.ylabel('Number of iterations')
    plt.xlabel('Size of the problem')
    plt.tight_layout()
    plt.savefig('plots/forest_ql_size.png')

    fig, ax = plt.subplots()
    ax.plot(SIZE_ql, TIME_ARR_ql , color='red', label="Clock time", linewidth=2.0, linestyle='-')
    ax.plot(SIZE_ql, TIME_ARR_ql , color='red',  marker='o', markersize = 2)

    ax.legend(loc='best', frameon=True)
    plt.grid(False, linestyle='--')
    plt.title('Time necessary to converge \n (policy iteration)')
    plt.ylabel('Clock time')
    plt.xlabel('Size of the problem')
    plt.tight_layout()
    plt.savefig('plots/forest_ql_time.png')





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
    plt.savefig('plots/forest_all_iters.png')





    fig, ax = plt.subplots()
    ax.plot(SIZE_vi, TIME_ARR_vi , color='red', label="Value Iteration", linewidth=2.0, linestyle='-')
    ax.plot(SIZE_vi, TIME_ARR_vi , color='red',  marker='o', markersize = 4)
    ax.plot(SIZE_pi, TIME_ARR_pi , color='blue', label="Policy IterationPolicy Iteration", linewidth=2.0, linestyle='-')
    ax.plot(SIZE_pi, TIME_ARR_pi , color='blue',  marker='o', markersize = 4)
    ax.plot(SIZE_ql, TIME_ARR_ql , color='green', label="Q Learning", linewidth=2.0, linestyle='-')
    ax.plot(SIZE_ql, TIME_ARR_ql , color='green',  marker='o', markersize = 4)
    ax.set_yscale('log')

    ax.legend(loc='best', frameon=True)
    plt.grid(False, linestyle='--')

    plt.title('Clock time necessary to converge \n (Value iteration)')
    plt.ylabel('Clock time')
    plt.xlabel('Size of the problem.')
    plt.tight_layout()
    plt.savefig('plots/forest_all_time.png')



    for state in range(env.nS):
        qbest[state] = np.argmax(qtable[state,:])

    draw_policy(policy_pi99, V_ARR_pi99[len(V_ARR_pi99)-1], env, size, "plots/forest_pi_policy99.png", "Policy visualization \n(policy iteration), size = 10")
    plt.rcParams.update(plt.rcParamsDefault)
