import os
import six
import sys
sys.modules['sklearn.externals.six'] = six

# import neural_networks as nn
import numpy as np
import matplotlib.pyplot as plt
# import randomized_optimization as ro
import sys
import time
import pandas as pd
from mlrose.opt_probs import TSPOpt, DiscreteOpt
from mlrose.fitness import TravellingSales, FlipFlop, FourPeaks, Queens, MaxKColor, Knapsack, SixPeaks

# import numpy as np
# import matplotlib.pyplot as plt
import utils
# Neural Networks

import numpy as np
import matplotlib.pyplot as plt
import time
import utils

from mlrose.decay import ExpDecay
from mlrose.neural import NeuralNetwork

from sklearn.metrics import log_loss, classification_report, precision_recall_curve
from sklearn.model_selection import train_test_split

from mlrose.algorithms import random_hill_climb, simulated_annealing, genetic_alg, mimic
from mlrose.decay import ExpDecay

IMAGE_DIR = 'images/'

from sklearn.datasets import load_breast_cancer, load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold


# Problem 1 - Four Peaks
length = 30
four_fitness = FourPeaks(t_pct=0.2)
problem = DiscreteOpt(length=length, fitness_fn=four_fitness, maximize=True, max_val=2)
problem.mimic_speed = True
random_states = [1234 + 1 * i for i in range(5)]  # random seeds for get performances over multiple random runs

kwargs = {"rhc_max_iters": 1000,

          "sa_max_iters": 1000,
          "sa_init_temp": 100,
          "sa_exp_decay_rate": 0.02,
          "sa_min_temp": 0.001,

          "ga_max_iters": 300,
          "ga_pop_size": 900,
          "ga_keep_pct": 0.5,

          "mimic_max_iters": 200,
          "mimic_pop_size": 900,
          "mimic_keep_pct": 0.5,

          "plot_name": 'Four_Peaks',
          "plot_ylabel": 'Fitness'}

# Initialize lists of fitness curves and time curves
rhc_fitness, sa_fitness, ga_fitness, mimic_fitness = [], [], [], []
rhc_times, sa_times, ga_times, mimic_times = [], [], [], []

# Set an exponential decay schedule for SA
exp_decay = ExpDecay(init_temp=kwargs['sa_init_temp'],
                     exp_const=kwargs['sa_exp_decay_rate'],
                     min_temp=kwargs['sa_min_temp'])

# For multiple random runs
for random_state in random_states:
    # Run RHC and get best state and objective found
    start_time = time.time()
    _, best_fitness, fitness_curve = random_hill_climb(problem,
                                                       max_attempts=kwargs['rhc_max_iters'],
                                                       max_iters=kwargs['rhc_max_iters'],
                                                       curve=True, random_state=random_state)

    rhc_fitness.append(fitness_curve)
    rhc_times.append(time.time() - start_time)
    print('\nRHC: best_objective = {:.3f}'.format(best_fitness))

    # Run SA and get best state and objective found
    start_time = time.time()
    _, best_fitness, fitness_curve = simulated_annealing(problem,
                                                         schedule=exp_decay,
                                                         max_attempts=kwargs['sa_max_iters'],
                                                         max_iters=kwargs['sa_max_iters'],
                                                         curve=True, random_state=random_state)

    sa_fitness.append(fitness_curve)
    sa_times.append(time.time() - start_time)
    print('SA: best_objective = {:.3f}'.format(best_fitness))

    # Run GA and get best state and objective found
    start_time = time.time()
    _, best_fitness, fitness_curve = genetic_alg(problem,
                                                 pop_size=kwargs['ga_pop_size'],
                                                 mutation_prob=1.0 - kwargs['ga_keep_pct'],
                                                 max_attempts=kwargs['ga_max_iters'],
                                                 max_iters=kwargs['ga_max_iters'],
                                                 curve=True, random_state=random_state)

    ga_fitness.append(fitness_curve)
    ga_times.append(time.time() - start_time)
    print('GA: best_objective = {:.3f}'.format(best_fitness))

    # Run MIMIC and get best state and objective found
    start_time = time.time()
    _, best_fitness, fitness_curve = mimic(problem,
                                           pop_size=kwargs['mimic_pop_size'],
                                           keep_pct=kwargs['mimic_keep_pct'],
                                           max_attempts=kwargs['mimic_max_iters'],
                                           max_iters=kwargs['mimic_max_iters'],
                                           curve=True, random_state=random_state)

    mimic_fitness.append(fitness_curve)
    mimic_times.append(time.time() - start_time)
    print('MIMIC: best_objective = {:.3f}'.format(best_fitness))


print('RHC: fitting time = {:.3f}'.format(sum(rhc_times)/len(random_states)))
print('SA: fitting time = {:.3f}'.format(sum(sa_times)/len(random_states)))
print('GA: fitting time = {:.3f}'.format(sum(ga_times)/len(random_states)))
print('MIMIC: fitting time = {:.3f}'.format(sum(mimic_times)/len(random_states)))


# Array of iterations to plot fitness vs. for RHC, SA, GA and MIMIC
rhc_iterations = np.arange(1, kwargs['rhc_max_iters'] + 1)
sa_iterations = np.arange(1, kwargs['sa_max_iters'] + 1)
ga_iterations = np.arange(1, kwargs['ga_max_iters'] + 1)
mimic_iterations = np.arange(1, kwargs['mimic_max_iters'] + 1)

# Plot objective curves, set title and labels
plt.figure()
utils.plot_helper(x_axis=rhc_iterations, y_axis=np.array(rhc_fitness), label='RHC')
utils.plot_helper(x_axis=sa_iterations, y_axis=np.array(sa_fitness), label='SA')
utils.plot_helper(x_axis=ga_iterations, y_axis=np.array(ga_fitness), label='GA')
utils.plot_helper(x_axis=mimic_iterations, y_axis=np.array(mimic_fitness), label='MIMIC')
utils.set_plot_title_labels(title='{} - Fitness versus iterations'.format(kwargs['plot_name']),
                            x_label='Iterations',
                            y_label=kwargs['plot_ylabel'])

# Save figure
plt.savefig(IMAGE_DIR + '{}_fitness_vs_iterations'.format(kwargs['plot_name']))
print('\n')

pop_size_list = [100, 300, 500, 700, 900]
ga_pop_fitness_list = []
mimic_pop_fitness_list = []

for random_state in random_states:
    ga_pop_fitness = []
    mimic_pop_fitness = []
    for i in range(len(pop_size_list)):
        # Run GA and get best state and objective found
        _, best_fitness, _ = genetic_alg(problem,
                                         pop_size=pop_size_list[i],
                                         mutation_prob=1.0 - kwargs['ga_keep_pct'],
                                         max_attempts=kwargs['ga_max_iters'],
                                         max_iters=kwargs['ga_max_iters'],
                                         curve=True, random_state=random_state)

        ga_pop_fitness.append(best_fitness)
        print('GA: best_objective = {:.3f}'.format(best_fitness))

        # Run MIMIC and get best state and objective found
        _, best_fitness, _ = mimic(problem,
                                   pop_size=pop_size_list[i],
                                   keep_pct=kwargs['mimic_keep_pct'],
                                   max_attempts=kwargs['mimic_max_iters'],
                                   max_iters=kwargs['mimic_max_iters'],
                                   curve=True, random_state=random_state)

        mimic_pop_fitness.append(best_fitness)
        print('MIMIC: best_objective = {:.3f}'.format(best_fitness))

    ga_pop_fitness_list.append(ga_pop_fitness)
    mimic_pop_fitness_list.append(mimic_pop_fitness)

# Plot objective curves, set title and labels
plt.figure()
# plot = plt.plot(pop_size_list, y_mean, label='GA')
# plot = plt.plot(x_axis, y_mean, label='MIMIC')
utils.plot_helper(x_axis=pop_size_list, y_axis=np.array(ga_pop_fitness_list), label='GA')
utils.plot_helper(x_axis=pop_size_list, y_axis=np.array(mimic_pop_fitness_list), label='MIMIC')
utils.set_plot_title_labels(title='{} - fitness versus population size'.format(kwargs['plot_name']),
                            x_label='population size',
                            y_label=kwargs['plot_ylabel'])

# Save figure
plt.savefig(IMAGE_DIR + '{}_fitness_vs_population_size'.format(kwargs['plot_name']))
print('\n')


keep_pct_list = [0.1, 0.3, 0.5, 0.7, 0.9]
ga_keep_fitness_list = []
mimic_keep_fitness_list = []

for random_state in random_states:
    ga_keep_fitness = []
    mimic_keep_fitness = []
    for i in range(len(keep_pct_list)):
        # Run GA and get best state and objective found
        _, best_fitness, _ = genetic_alg(problem,
                                         pop_size=kwargs['ga_pop_size'],
                                         mutation_prob=1.0 - keep_pct_list[i],
                                         max_attempts=kwargs['ga_max_iters'],
                                         max_iters=kwargs['ga_max_iters'],
                                         curve=True, random_state=random_state)

        ga_keep_fitness.append(best_fitness)
        print('GA: best_objective = {:.3f}'.format(best_fitness))

        # Run MIMIC and get best state and objective found
        _, best_fitness, _ = mimic(problem,
                                   pop_size=kwargs['mimic_pop_size'],
                                   keep_pct=keep_pct_list[i],
                                   max_attempts=kwargs['mimic_max_iters'],
                                   max_iters=kwargs['mimic_max_iters'],
                                   curve=True, random_state=random_state)

        mimic_keep_fitness.append(best_fitness)
        print('MIMIC: best_objective = {:.3f}'.format(best_fitness))

    ga_keep_fitness_list.append(ga_keep_fitness)
    mimic_keep_fitness_list.append(mimic_keep_fitness)
print('\n')


# Plot objective curves, set title and labels
plt.figure()
utils.plot_helper(x_axis=keep_pct_list, y_axis=np.array(ga_keep_fitness_list), label='GA')
utils.plot_helper(x_axis=keep_pct_list, y_axis=np.array(mimic_keep_fitness_list), label='MIMIC')
utils.set_plot_title_labels(title='{} - fitness versus keep pct'.format(kwargs['plot_name']),
                            x_label='keep_pct',
                            y_label=kwargs['plot_ylabel'])

# Save figure
plt.savefig(IMAGE_DIR + '{}_fitness_vs_keep_pct'.format(kwargs['plot_name']))

temp_decay_list = [0.02, 0.04, 0.06, 0.08, 0.1]
sa_decay_fitness_list = []

for random_state in random_states:
    sa_decay_fitness = []
    for i in range(len(temp_decay_list)):
        # Set an exponential decay schedule for SA
        exp_decay = ExpDecay(init_temp=kwargs['sa_init_temp'],
                             exp_const=temp_decay_list[i],
                             min_temp=kwargs['sa_min_temp'])

        _, best_fitness, _ = simulated_annealing(problem,
                                                 schedule=exp_decay,
                                                 # max_attempts=kwargs['sa_max_iters'],
                                                 # max_iters=kwargs['sa_max_iters'],
                                                 max_attempts=1000,
                                                 max_iters=1000,
                                                 curve=True, random_state=random_state)
        sa_decay_fitness.append(best_fitness)
        print('MIMIC: best_objective = {:.3f}'.format(best_fitness))
    sa_decay_fitness_list.append(sa_decay_fitness)

# Plot objective curves, set title and labels
plt.figure()
utils.plot_helper(x_axis=temp_decay_list, y_axis=np.array(sa_decay_fitness_list), label='SA')
utils.set_plot_title_labels(title='{} - fitness versus temperature decay rate'.format(kwargs['plot_name']),
                            x_label='temp_decay_rate',
                            y_label=kwargs['plot_ylabel'])

# Save figure
plt.savefig(IMAGE_DIR + '{}_fitness_vs_temp_decay_rate'.format(kwargs['plot_name']))

print('\n')







# Problem 2 - N-Queens




# Define Queens objective function and problem
length = 100
queen = Queens()
problem = DiscreteOpt(length=length, fitness_fn=queen, maximize=True, max_val=2)
problem.mimic_speed = True  # set fast MIMIC


random_states = [1234 + 1 * i for i in range(5)]  # random seeds for get performances over multiple random runs

kwargs = {"rhc_max_iters": 600,

          "sa_max_iters": 600,
          "sa_init_temp": 100,
          "sa_exp_decay_rate": 0.02,
          "sa_min_temp": 0.001,

          "ga_max_iters": 600,
          "ga_pop_size": 900,
          "ga_keep_pct": 0.5,

          "mimic_max_iters": 50,
          "mimic_pop_size": 900,
          "mimic_keep_pct": 0.5,

          "plot_name": 'Queen',
          "plot_ylabel": 'Fitness'}

# Initialize lists of fitness curves and time curves
rhc_fitness, sa_fitness, ga_fitness, mimic_fitness = [], [], [], []
rhc_times, sa_times, ga_times, mimic_times = [], [], [], []

# Set an exponential decay schedule for SA
exp_decay = ExpDecay(init_temp=kwargs['sa_init_temp'],
                     exp_const=kwargs['sa_exp_decay_rate'],
                     min_temp=kwargs['sa_min_temp'])

# For multiple random runs
for random_state in random_states:
    # Run RHC and get best state and objective found
    start_time = time.time()
    _, best_fitness, fitness_curve = random_hill_climb(problem,
                                                       max_attempts=kwargs['rhc_max_iters'],
                                                       max_iters=kwargs['rhc_max_iters'],
                                                       curve=True, random_state=random_state)

    rhc_fitness.append(fitness_curve)
    rhc_times.append(time.time() - start_time)
    print('\nRHC: best_objective = {:.3f}'.format(best_fitness))

    # Run SA and get best state and objective found
    start_time = time.time()
    _, best_fitness, fitness_curve = simulated_annealing(problem,
                                                         schedule=exp_decay,
                                                         max_attempts=kwargs['sa_max_iters'],
                                                         max_iters=kwargs['sa_max_iters'],
                                                         curve=True, random_state=random_state)

    sa_fitness.append(fitness_curve)
    sa_times.append(time.time() - start_time)
    print('SA: best_objective = {:.3f}'.format(best_fitness))

    # Run GA and get best state and objective found
    start_time = time.time()
    _, best_fitness, fitness_curve = genetic_alg(problem,
                                                 pop_size=kwargs['ga_pop_size'],
                                                 mutation_prob=1.0 - kwargs['ga_keep_pct'],
                                                 max_attempts=kwargs['ga_max_iters'],
                                                 max_iters=kwargs['ga_max_iters'],
                                                 curve=True, random_state=random_state)

    ga_fitness.append(fitness_curve)
    ga_times.append(time.time() - start_time)
    print('GA: best_objective = {:.3f}'.format(best_fitness))

    # Run MIMIC and get best state and objective found
    start_time = time.time()
    _, best_fitness, fitness_curve = mimic(problem,
                                           pop_size=kwargs['mimic_pop_size'],
                                           keep_pct=kwargs['mimic_keep_pct'],
                                           max_attempts=kwargs['mimic_max_iters'],
                                           max_iters=kwargs['mimic_max_iters'],
                                           curve=True, random_state=random_state)

    mimic_fitness.append(fitness_curve)
    mimic_times.append(time.time() - start_time)
    print('MIMIC: best_objective = {:.3f}'.format(best_fitness))


print('RHC: fitting time = {:.3f}'.format(sum(rhc_times)/len(random_states)))
print('SA: fitting time = {:.3f}'.format(sum(sa_times)/len(random_states)))
print('GA: fitting time = {:.3f}'.format(sum(ga_times)/len(random_states)))
print('MIMIC: fitting time = {:.3f}'.format(sum(mimic_times)/len(random_states)))


# Array of iterations to plot fitness vs. for RHC, SA, GA and MIMIC
rhc_iterations = np.arange(1, kwargs['rhc_max_iters'] + 1)
sa_iterations = np.arange(1, kwargs['sa_max_iters'] + 1)
ga_iterations = np.arange(1, kwargs['ga_max_iters'] + 1)
mimic_iterations = np.arange(1, kwargs['mimic_max_iters'] + 1)

# Plot objective curves, set title and labels
plt.figure()
utils.plot_helper(x_axis=rhc_iterations, y_axis=np.array(rhc_fitness), label='RHC')
utils.plot_helper(x_axis=sa_iterations, y_axis=np.array(sa_fitness), label='SA')
utils.plot_helper(x_axis=ga_iterations, y_axis=np.array(ga_fitness), label='GA')
utils.plot_helper(x_axis=mimic_iterations, y_axis=np.array(mimic_fitness), label='MIMIC')
utils.set_plot_title_labels(title='{} - Fitness versus iterations'.format(kwargs['plot_name']),
                            x_label='Iterations',
                            y_label=kwargs['plot_ylabel'])

# Save figure
plt.savefig(IMAGE_DIR + '{}_fitness_vs_iterations'.format(kwargs['plot_name']))
print('\n')

pop_size_list = [100, 300, 500, 700, 900]
ga_pop_fitness_list = []
mimic_pop_fitness_list = []

for random_state in random_states:
    ga_pop_fitness = []
    mimic_pop_fitness = []
    for i in range(len(pop_size_list)):
        # Run GA and get best state and objective found
        _, best_fitness, _ = genetic_alg(problem,
                                         pop_size=pop_size_list[i],
                                         mutation_prob=1.0 - kwargs['ga_keep_pct'],
                                         max_attempts=kwargs['ga_max_iters'],
                                         max_iters=kwargs['ga_max_iters'],
                                         curve=True, random_state=random_state)

        ga_pop_fitness.append(best_fitness)
        print('GA: best_objective = {:.3f}'.format(best_fitness))

        # Run MIMIC and get best state and objective found
        _, best_fitness, _ = mimic(problem,
                                   pop_size=pop_size_list[i],
                                   keep_pct=kwargs['mimic_keep_pct'],
                                   max_attempts=kwargs['mimic_max_iters'],
                                   max_iters=kwargs['mimic_max_iters'],
                                   curve=True, random_state=random_state)

        mimic_pop_fitness.append(best_fitness)
        print('MIMIC: best_objective = {:.3f}'.format(best_fitness))

    ga_pop_fitness_list.append(ga_pop_fitness)
    mimic_pop_fitness_list.append(mimic_pop_fitness)

# Plot objective curves, set title and labels
plt.figure()
# plot = plt.plot(pop_size_list, y_mean, label='GA')
# plot = plt.plot(x_axis, y_mean, label='MIMIC')
utils.plot_helper(x_axis=pop_size_list, y_axis=np.array(ga_pop_fitness_list), label='GA')
utils.plot_helper(x_axis=pop_size_list, y_axis=np.array(mimic_pop_fitness_list), label='MIMIC')
utils.set_plot_title_labels(title='{} - fitness versus population size'.format(kwargs['plot_name']),
                            x_label='population size',
                            y_label=kwargs['plot_ylabel'])

# Save figure
plt.savefig(IMAGE_DIR + '{}_fitness_vs_population_size'.format(kwargs['plot_name']))
print('\n')


keep_pct_list = [0.1, 0.3, 0.5, 0.7, 0.9]
ga_keep_fitness_list = []
mimic_keep_fitness_list = []

for random_state in random_states:
    ga_keep_fitness = []
    mimic_keep_fitness = []
    for i in range(len(keep_pct_list)):
        # Run GA and get best state and objective found
        _, best_fitness, _ = genetic_alg(problem,
                                         pop_size=kwargs['ga_pop_size'],
                                         mutation_prob=1.0 - keep_pct_list[i],
                                         max_attempts=kwargs['ga_max_iters'],
                                         max_iters=kwargs['ga_max_iters'],
                                         curve=True, random_state=random_state)

        ga_keep_fitness.append(best_fitness)
        print('GA: best_objective = {:.3f}'.format(best_fitness))

        # Run MIMIC and get best state and objective found
        _, best_fitness, _ = mimic(problem,
                                   pop_size=kwargs['mimic_pop_size'],
                                   keep_pct=keep_pct_list[i],
                                   max_attempts=kwargs['mimic_max_iters'],
                                   max_iters=kwargs['mimic_max_iters'],
                                   curve=True, random_state=random_state)

        mimic_keep_fitness.append(best_fitness)
        print('MIMIC: best_objective = {:.3f}'.format(best_fitness))

    ga_keep_fitness_list.append(ga_keep_fitness)
    mimic_keep_fitness_list.append(mimic_keep_fitness)
print('\n')


# Plot objective curves, set title and labels
plt.figure()
utils.plot_helper(x_axis=keep_pct_list, y_axis=np.array(ga_keep_fitness_list), label='GA')
utils.plot_helper(x_axis=keep_pct_list, y_axis=np.array(mimic_keep_fitness_list), label='MIMIC')
utils.set_plot_title_labels(title='{} - fitness versus keep pct'.format(kwargs['plot_name']),
                            x_label='keep_pct',
                            y_label=kwargs['plot_ylabel'])

# Save figure
plt.savefig(IMAGE_DIR + '{}_fitness_vs_keep_pct'.format(kwargs['plot_name']))

temp_decay_list = [0.02, 0.04, 0.06, 0.08, 0.1]
sa_decay_fitness_list = []

for random_state in random_states:
    sa_decay_fitness = []
    for i in range(len(temp_decay_list)):
        # Set an exponential decay schedule for SA
        exp_decay = ExpDecay(init_temp=kwargs['sa_init_temp'],
                             exp_const=temp_decay_list[i],
                             min_temp=kwargs['sa_min_temp'])

        _, best_fitness, _ = simulated_annealing(problem,
                                                 schedule=exp_decay,
                                                 # max_attempts=kwargs['sa_max_iters'],
                                                 # max_iters=kwargs['sa_max_iters'],
                                                 max_attempts=1000,
                                                 max_iters=1000,
                                                 curve=True, random_state=random_state)
        sa_decay_fitness.append(best_fitness)
        print('MIMIC: best_objective = {:.3f}'.format(best_fitness))
    sa_decay_fitness_list.append(sa_decay_fitness)

# Plot objective curves, set title and labels
plt.figure()
utils.plot_helper(x_axis=temp_decay_list, y_axis=np.array(sa_decay_fitness_list), label='SA')
utils.set_plot_title_labels(title='{} - fitness versus temperature decay rate'.format(kwargs['plot_name']),
                            x_label='temp_decay_rate',
                            y_label=kwargs['plot_ylabel'])

# Save figure
plt.savefig(IMAGE_DIR + '{}_fitness_vs_temp_decay_rate'.format(kwargs['plot_name']))

print('\n')




# Problem 3 - Knapsack

weights = [10, 5, 2, 8, 15, 20, 5, 2, 1, 20, 8, 6, 14, 22, 50, 5, 10, 12, 12, 18, 26, 32, 4, 8, 10, 5, 22,
           10, 5, 2, 8, 15, 20, 5, 2, 1, 20, 8, 6, 14, 22, 50, 5, 10, 12, 12, 18, 26, 32, 4, 8, 10, 5, 22,
           10, 5, 2, 8, 15, 20, 5, 2, 1, 20, 8, 6, 14, 22, 50, 5, 10, 12, 12, 18, 26, 32, 4, 8, 10, 5, 22]
values = [1, 2, 3, 4, 5, 2, 5, 10, 1, 4, 10, 2, 2, 8, 100, 5, 15, 24, 8, 14, 36, 10, 5, 2, 120, 4, 8,
          1, 2, 3, 4, 5, 2, 5, 10, 1, 4, 10, 2, 2, 8, 100, 5, 15, 24, 8, 14, 36, 10, 5, 2, 120, 4, 8,
          1, 2, 3, 4, 5, 2, 5, 10, 1, 4, 10, 2, 2, 8, 100, 5, 15, 24, 8, 14, 36, 10, 5, 2, 120, 4, 8]
max_weight_pct = 0.6
n = len(weights)
# Initialize fitness function object using coords_list
fitness = Knapsack(weights, values, max_weight_pct)
# Define optimization problem object
problem = DiscreteOpt(length=n, fitness_fn=fitness, maximize=True)
problem.mimic_speed = True  # set fast MIMIC



random_states = [1234 + 1 * i for i in range(5)]  # random seeds for get performances over multiple random runs

kwargs = {"rhc_max_iters": 600,

          "sa_max_iters": 600,
          "sa_init_temp": 100,
          "sa_exp_decay_rate": 0.02,
          "sa_min_temp": 0.001,

          "ga_max_iters": 600,
          "ga_pop_size": 900,
          "ga_keep_pct": 0.5,

          "mimic_max_iters": 100,
          "mimic_pop_size": 900,
          "mimic_keep_pct": 0.5,

          "plot_name": 'Knapsack',
          "plot_ylabel": 'Fitness'}

# Initialize lists of fitness curves and time curves
rhc_fitness, sa_fitness, ga_fitness, mimic_fitness = [], [], [], []
rhc_times, sa_times, ga_times, mimic_times = [], [], [], []

# Set an exponential decay schedule for SA
exp_decay = ExpDecay(init_temp=kwargs['sa_init_temp'],
                     exp_const=kwargs['sa_exp_decay_rate'],
                     min_temp=kwargs['sa_min_temp'])

# For multiple random runs
for random_state in random_states:
    # Run RHC and get best state and objective found
    start_time = time.time()
    _, best_fitness, fitness_curve = random_hill_climb(problem,
                                                       max_attempts=kwargs['rhc_max_iters'],
                                                       max_iters=kwargs['rhc_max_iters'],
                                                       curve=True, random_state=random_state)

    rhc_fitness.append(fitness_curve)
    rhc_times.append(time.time() - start_time)
    print('\nRHC: best_objective = {:.3f}'.format(best_fitness))

    # Run SA and get best state and objective found
    start_time = time.time()
    _, best_fitness, fitness_curve = simulated_annealing(problem,
                                                         schedule=exp_decay,
                                                         max_attempts=kwargs['sa_max_iters'],
                                                         max_iters=kwargs['sa_max_iters'],
                                                         curve=True, random_state=random_state)

    sa_fitness.append(fitness_curve)
    sa_times.append(time.time() - start_time)
    print('SA: best_objective = {:.3f}'.format(best_fitness))

    # Run GA and get best state and objective found
    start_time = time.time()
    _, best_fitness, fitness_curve = genetic_alg(problem,
                                                 pop_size=kwargs['ga_pop_size'],
                                                 mutation_prob=1.0 - kwargs['ga_keep_pct'],
                                                 max_attempts=kwargs['ga_max_iters'],
                                                 max_iters=kwargs['ga_max_iters'],
                                                 curve=True, random_state=random_state)

    ga_fitness.append(fitness_curve)
    ga_times.append(time.time() - start_time)
    print('GA: best_objective = {:.3f}'.format(best_fitness))

    # Run MIMIC and get best state and objective found
    start_time = time.time()
    _, best_fitness, fitness_curve = mimic(problem,
                                           pop_size=kwargs['mimic_pop_size'],
                                           keep_pct=kwargs['mimic_keep_pct'],
                                           max_attempts=kwargs['mimic_max_iters'],
                                           max_iters=kwargs['mimic_max_iters'],
                                           curve=True, random_state=random_state)

    mimic_fitness.append(fitness_curve)
    mimic_times.append(time.time() - start_time)
    print('MIMIC: best_objective = {:.3f}'.format(best_fitness))


print('RHC: fitting time = {:.3f}'.format(sum(rhc_times)/len(random_states)))
print('SA: fitting time = {:.3f}'.format(sum(sa_times)/len(random_states)))
print('GA: fitting time = {:.3f}'.format(sum(ga_times)/len(random_states)))
print('MIMIC: fitting time = {:.3f}'.format(sum(mimic_times)/len(random_states)))


# Array of iterations to plot fitness vs. for RHC, SA, GA and MIMIC
rhc_iterations = np.arange(1, kwargs['rhc_max_iters'] + 1)
sa_iterations = np.arange(1, kwargs['sa_max_iters'] + 1)
ga_iterations = np.arange(1, kwargs['ga_max_iters'] + 1)
mimic_iterations = np.arange(1, kwargs['mimic_max_iters'] + 1)

# Plot objective curves, set title and labels
plt.figure()
utils.plot_helper(x_axis=rhc_iterations, y_axis=np.array(rhc_fitness), label='RHC')
utils.plot_helper(x_axis=sa_iterations, y_axis=np.array(sa_fitness), label='SA')
utils.plot_helper(x_axis=ga_iterations, y_axis=np.array(ga_fitness), label='GA')
utils.plot_helper(x_axis=mimic_iterations, y_axis=np.array(mimic_fitness), label='MIMIC')
utils.set_plot_title_labels(title='{} - Fitness versus iterations'.format(kwargs['plot_name']),
                            x_label='Iterations',
                            y_label=kwargs['plot_ylabel'])

# Save figure
plt.savefig(IMAGE_DIR + '{}_fitness_vs_iterations'.format(kwargs['plot_name']))
print('\n')

pop_size_list = [100, 300, 500, 700, 900]
ga_pop_fitness_list = []
mimic_pop_fitness_list = []

for random_state in random_states:
    ga_pop_fitness = []
    mimic_pop_fitness = []
    for i in range(len(pop_size_list)):
        # Run GA and get best state and objective found
        _, best_fitness, _ = genetic_alg(problem,
                                         pop_size=pop_size_list[i],
                                         mutation_prob=1.0 - kwargs['ga_keep_pct'],
                                         max_attempts=kwargs['ga_max_iters'],
                                         max_iters=kwargs['ga_max_iters'],
                                         curve=True, random_state=random_state)

        ga_pop_fitness.append(best_fitness)
        print('GA: best_objective = {:.3f}'.format(best_fitness))

        # Run MIMIC and get best state and objective found
        _, best_fitness, _ = mimic(problem,
                                   pop_size=pop_size_list[i],
                                   keep_pct=kwargs['mimic_keep_pct'],
                                   max_attempts=kwargs['mimic_max_iters'],
                                   max_iters=kwargs['mimic_max_iters'],
                                   curve=True, random_state=random_state)

        mimic_pop_fitness.append(best_fitness)
        print('MIMIC: best_objective = {:.3f}'.format(best_fitness))

    ga_pop_fitness_list.append(ga_pop_fitness)
    mimic_pop_fitness_list.append(mimic_pop_fitness)

# Plot objective curves, set title and labels
plt.figure()
# plot = plt.plot(pop_size_list, y_mean, label='GA')
# plot = plt.plot(x_axis, y_mean, label='MIMIC')
utils.plot_helper(x_axis=pop_size_list, y_axis=np.array(ga_pop_fitness_list), label='GA')
utils.plot_helper(x_axis=pop_size_list, y_axis=np.array(mimic_pop_fitness_list), label='MIMIC')
utils.set_plot_title_labels(title='{} - fitness versus population size'.format(kwargs['plot_name']),
                            x_label='population size',
                            y_label=kwargs['plot_ylabel'])

# Save figure
plt.savefig(IMAGE_DIR + '{}_fitness_vs_population_size'.format(kwargs['plot_name']))
print('\n')


keep_pct_list = [0.1, 0.3, 0.5, 0.7, 0.9]
ga_keep_fitness_list = []
mimic_keep_fitness_list = []

for random_state in random_states:
    ga_keep_fitness = []
    mimic_keep_fitness = []
    for i in range(len(keep_pct_list)):
        # Run GA and get best state and objective found
        _, best_fitness, _ = genetic_alg(problem,
                                         pop_size=kwargs['ga_pop_size'],
                                         mutation_prob=1.0 - keep_pct_list[i],
                                         max_attempts=kwargs['ga_max_iters'],
                                         max_iters=kwargs['ga_max_iters'],
                                         curve=True, random_state=random_state)

        ga_keep_fitness.append(best_fitness)
        print('GA: best_objective = {:.3f}'.format(best_fitness))

        # Run MIMIC and get best state and objective found
        _, best_fitness, _ = mimic(problem,
                                   pop_size=kwargs['mimic_pop_size'],
                                   keep_pct=keep_pct_list[i],
                                   max_attempts=kwargs['mimic_max_iters'],
                                   max_iters=kwargs['mimic_max_iters'],
                                   curve=True, random_state=random_state)

        mimic_keep_fitness.append(best_fitness)
        print('MIMIC: best_objective = {:.3f}'.format(best_fitness))

    ga_keep_fitness_list.append(ga_keep_fitness)
    mimic_keep_fitness_list.append(mimic_keep_fitness)
print('\n')


# Plot objective curves, set title and labels
plt.figure()
utils.plot_helper(x_axis=keep_pct_list, y_axis=np.array(ga_keep_fitness_list), label='GA')
utils.plot_helper(x_axis=keep_pct_list, y_axis=np.array(mimic_keep_fitness_list), label='MIMIC')
utils.set_plot_title_labels(title='{} - fitness versus keep pct'.format(kwargs['plot_name']),
                            x_label='keep_pct',
                            y_label=kwargs['plot_ylabel'])

# Save figure
plt.savefig(IMAGE_DIR + '{}_fitness_vs_keep_pct'.format(kwargs['plot_name']))

temp_decay_list = [0.02, 0.04, 0.06, 0.08, 0.1]
sa_decay_fitness_list = []

for random_state in random_states:
    sa_decay_fitness = []
    for i in range(len(temp_decay_list)):
        # Set an exponential decay schedule for SA
        exp_decay = ExpDecay(init_temp=kwargs['sa_init_temp'],
                             exp_const=temp_decay_list[i],
                             min_temp=kwargs['sa_min_temp'])

        _, best_fitness, _ = simulated_annealing(problem,
                                                 schedule=exp_decay,
                                                 # max_attempts=kwargs['sa_max_iters'],
                                                 # max_iters=kwargs['sa_max_iters'],
                                                 max_attempts=1000,
                                                 max_iters=1000,
                                                 curve=True, random_state=random_state)
        sa_decay_fitness.append(best_fitness)
        print('MIMIC: best_objective = {:.3f}'.format(best_fitness))
    sa_decay_fitness_list.append(sa_decay_fitness)

# Plot objective curves, set title and labels
plt.figure()
utils.plot_helper(x_axis=temp_decay_list, y_axis=np.array(sa_decay_fitness_list), label='SA')
utils.set_plot_title_labels(title='{} - fitness versus temperature decay rate'.format(kwargs['plot_name']),
                            x_label='temp_decay_rate',
                            y_label=kwargs['plot_ylabel'])

# Save figure
plt.savefig(IMAGE_DIR + '{}_fitness_vs_temp_decay_rate'.format(kwargs['plot_name']))

print('\n')






# NN

def process_dataset(dataset_name):
    if dataset_name == "obesity":

        data = pd.read_csv(os.path.join("datasets", "ObesityDataSet_raw_and_data_sinthetic.csv"))

        data['Gender'] = data['Gender'].map({'Female': 0, 'Male': 1})
        data['family_history_with_overweight'] = data['family_history_with_overweight'].map({'no': 0, 'yes': 1})
        data['FAVC'] = data['FAVC'].map({'no': 0, 'yes': 1})
        data['CAEC'] = data['CAEC'].map({'no': 0, 'Sometimes': 1, 'Frequently': 2, 'Always': 3})
        data['SMOKE'] = data['SMOKE'].map({'no': 0, 'yes': 1})
        data['SCC'] = data['SCC'].map({'no': 0, 'yes': 1})
        data['CALC'] = data['CALC'].map({'no': 0, 'Sometimes': 1, 'Frequently': 2, 'Always': 3})
        data['MTRANS'] = data['MTRANS'].map({'Walking': 0, 'Bike': 1, 'Motorbike': 2, 'Public_Transportation': 3,
                                             'Automobile': 4})
        data['NObeyesdad'] = data['NObeyesdad'].map({'Insufficient_Weight': 0,
                                                     'Normal_Weight': 1,
                                                     'Overweight_Level_I': 2,
                                                     'Overweight_Level_II': 3,
                                                     'Obesity_Type_I': 4,
                                                     'Obesity_Type_II': 5,
                                                     'Obesity_Type_III': 6})

    elif dataset_name == "online_shopping":

        data = pd.read_csv(os.path.join("datasets", "online_shoppers_intention.csv"))

        data['Month'] = data['Month'].map({'Feb': 2,
                                           'Mar': 3,
                                           'May': 5,
                                           'June': 6,
                                           'Jul': 7,
                                           'Aug': 8,
                                           'Sep': 9,
                                           'Oct': 10,
                                           'Nov': 11,
                                           'Dec': 12})
        data['VisitorType'] = data['VisitorType'].map({'Returning_Visitor': 0,
                                                       'New_Visitor': 1,
                                                       'Other': 2})
        data['Weekend'] = data['Weekend'].astype(int)
        data['Revenue'] = data['Revenue'].astype(int)

    else:
        data = []

    return data


def split_data(data, testing_raio=0.2, norm=False):
    data_matrix = data.values

    def scale(col, min, max):
        range = col.max() - col.min()
        a = (col - col.min()) / range
        return a * (max - min) + min

    if norm and data_matrix.shape[1] == 17:
        data_matrix[:, 1] = scale(data_matrix[:, 1], 0, 5)
        data_matrix[:, 2] = scale(data_matrix[:, 2], 0, 5)
        data_matrix[:, 3] = scale(data_matrix[:, 3], 0, 5)

    x = data_matrix[:, :-1]
    y = data_matrix[:, -1]

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=testing_raio, shuffle=True,
                                                        random_state=42, stratify=y)

    return x_train, x_test, y_train, y_test


# import and process datasets
dataset_name_list = ["online_shopping"]
datas = []
for dataset_name in dataset_name_list:
    datas.append(process_dataset(dataset_name))

kfold = KFold(n_splits=10, shuffle=True, random_state=1)

for data_num, data in enumerate(datas):
    dataset_name = dataset_name_list[data_num]
    # Split data
    if dataset_name == "obesity":
        x_train, x_test, y_train, y_test = split_data(data, norm=True)
    else:
        x_train, x_test, y_train, y_test = split_data(data)

    print("Training Set Shape: {}".format(x_train.shape))
    print("Testing Set Shape: {}".format(x_test.shape))




random_seeds = [1234 + 1 * i for i in range(1)]

# iterations = np.array([i for i in range(1, 10)] + [10 * i for i in range(1, 20, 2)])
iterations = np.array([5, 10, 25, 50, 100, 200, 300])
# iterations = np.array([5, 25])


kwargs = {"random_seeds": random_seeds,
          "rhc_max_iters": iterations,
          "sa_max_iters": iterations,
          "ga_max_iters": iterations,
          "init_temp": 100,
          "exp_decay_rate": 0.1,
          "min_temp": 0.001,
          "pop_size": 100,
          "mutation_prob": 0.2,
          }

# Initialize algorithms, corresponding acronyms and max number of iterations
algorithms = ['random_hill_climb', 'simulated_annealing', 'genetic_alg']
acronyms = ['RHC', 'SA', 'GA']
max_iters = ['rhc_max_iters', 'sa_max_iters', 'ga_max_iters']

# Initialize lists of training curves, validation curves and training times curves
train_curves, val_curves, train_time_curves = [], [], []

# Define SA exponential decay schedule
exp_decay = ExpDecay(init_temp=kwargs['init_temp'],
                     exp_const=kwargs['exp_decay_rate'],
                     min_temp=kwargs['min_temp'])

# Create one figure for training and validation losses, the second for training time
plt.figure()
train_val_figure = plt.gcf().number
plt.figure()
train_times_figure = plt.gcf().number
marker = ['+', 'x', 'o']
# For each of the optimization algorithms to test the Neural Network with
for i, algorithm in enumerate(algorithms):
    print('\nAlgorithm = {}'.format(algorithm))

    # For multiple random runs
    for random_seed in random_seeds:

        # Initialize training losses, validation losses and training time lists for current random run
        train_losses, val_losses, train_times = [], [], []

        # Compute stratified k-fold
        x_train_fold, x_val_fold, y_train_fold, y_val_fold = train_test_split(x_train, y_train,
                                                                              test_size=0.2, shuffle=True,
                                                                              random_state=random_seed,
                                                                              stratify=y_train)
        # For each max iterations to run for
        for max_iter in kwargs[max_iters[i]]:
            # Define Neural Network using current algorithm
            nn = NeuralNetwork(hidden_nodes=[50, 30], activation='relu',
                               algorithm=algorithm, max_iters=int(max_iter),
                               bias=True, is_classifier=True, learning_rate=0.001,
                               early_stopping=False, clip_max=1e10, schedule=exp_decay,
                               pop_size=kwargs['pop_size'], mutation_prob=kwargs['mutation_prob'],
                               max_attempts=int(max_iter), random_state=random_seed, curve=False)

            # Train on current training fold and append training time
            start_time = time.time()
            nn.fit(x_train_fold, y_train_fold)
            train_times.append(time.time() - start_time)

            # Compute and append training and validation log losses
            train_loss = log_loss(y_train_fold, nn.predict(x_train_fold))
            val_loss = log_loss(y_val_fold, nn.predict(x_val_fold))
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            print('{} - train loss = {:.3f}, val loss = {:.3f}'.format(max_iter, train_loss, val_loss))

        # Append curves for current random seed to corresponding lists of curves
        train_curves.append(train_losses)
        val_curves.append(val_losses)
        train_time_curves.append(train_times)

    # Plot training and validation figure for current algorithm
    plt.figure(train_val_figure)
    # utils.plot_helper(x_axis=kwargs[max_iters[i]], y_axis=np.array(train_curves), label='{} train'.format(acronyms[i]))
    # utils.plot_helper(x_axis=kwargs[max_iters[i]], y_axis=np.array(val_curves), label='{} val'.format(acronyms[i]))
    plt.plot(kwargs[max_iters[i]], np.mean(np.array(train_curves), axis=0), marker=marker[i], label='{} train'.format(acronyms[i]))
    plt.plot(kwargs[max_iters[i]], np.mean(np.array(val_curves), axis=0), marker=marker[i], label='{} test'.format(acronyms[i]))

    # Plot training time figure for current algorithm
    plt.figure(train_times_figure)
    # utils.plot_helper(x_axis=kwargs[max_iters[i]], y_axis=np.array(train_time_curves), label=acronyms[i])
    plt.plot(kwargs[max_iters[i]], np.mean(np.array(train_time_curves), axis=0), marker=marker[i], label='{} test'.format(acronyms[i]))

# Set title and labels to training and validation figure
plt.figure(train_val_figure)
utils.set_plot_title_labels(title='Neural Network - Loss vs. iterations',
                            x_label='Iterations',
                            y_label='Loss')

# Save figure
plt.savefig(IMAGE_DIR + 'nn_objective_vs_iterations')

# Set title and labels to training time figure
plt.figure(train_times_figure)
utils.set_plot_title_labels(title='Neural Network - Time vs. iterations',
                            x_label='Iterations',
                            y_label='Time (seconds)')

# Save figure
plt.savefig(IMAGE_DIR + 'nn_time_vs_iterations')









kwargs = {"random_seeds": random_seeds[0],
          "max_iters": 200,
          "init_temp": 100,
          "exp_decay_rate": 0.1,
          "min_temp": 0.001,
          "pop_size": 100,
          "mutation_prob": 0.2,
          }

# Define SA exponential decay schedule
exp_decay = ExpDecay(init_temp=kwargs['init_temp'],
                     exp_const=kwargs['exp_decay_rate'],
                     min_temp=kwargs['min_temp'])

# Define Neural Network using RHC for weights optimization
rhc_nn = NeuralNetwork(hidden_nodes=[50, 30], activation='relu',
                       algorithm='random_hill_climb', max_iters=kwargs['max_iters'],
                       bias=True, is_classifier=True, learning_rate=0.001,
                       early_stopping=False, clip_max=1e10,
                       max_attempts=kwargs['max_iters'], random_state=random_seed, curve=False)

# Define Neural Network using SA for weights optimization
sa_nn = NeuralNetwork(hidden_nodes=[50, 30], activation='relu',
                      algorithm='simulated_annealing', max_iters=kwargs['max_iters'],
                      bias=True, is_classifier=True, learning_rate=0.001,
                      early_stopping=False, clip_max=1e10, schedule=exp_decay,
                      max_attempts=kwargs['max_iters'], random_state=random_seed, curve=False)

# Define Neural Network using GA for weights optimization
ga_nn = NeuralNetwork(hidden_nodes=[50, 30], activation='relu',
                      algorithm='genetic_alg', max_iters=kwargs['max_iters'],
                      bias=True, is_classifier=True, learning_rate=0.001,
                      early_stopping=False, clip_max=1e10,
                      pop_size=kwargs['pop_size'], mutation_prob=kwargs['mutation_prob'],
                      max_attempts=kwargs['max_iters'], random_state=random_seed, curve=False)

# Fit each of the Neural Networks using the different optimization algorithms
# mimic_nn.fit(x_train, y_train)
rhc_nn.fit(x_train, y_train)
sa_nn.fit(x_train, y_train)
ga_nn.fit(x_train, y_train)

# https: // towardsdatascience.com / metrics - to - evaluate - your - machine - learning - algorithm - f10ba6e38234
# right now outputs F1 score --> 2*precision*recall/(precision+recall)

# Print classification reports for all of the optimization algorithms
# print('MIMIC test classification report = \n {}'.format(classification_report(y_test, mimic_nn.predict(x_test))))
print('RHC test classification report = \n {}'.format(classification_report(y_test, rhc_nn.predict(x_test))))
print('SA test classification report = \n {}'.format(classification_report(y_test, sa_nn.predict(x_test))))
print('GA test classification report = \n {}'.format(classification_report(y_test, ga_nn.predict(x_test))))







