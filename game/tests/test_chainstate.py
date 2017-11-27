# test for game.py

import numpy as np
import random
from game import ChainState, Config
import time

# Configuration
config = Config()
config.num_nodes = 25
np.set_printoptions(formatter={'float': '{: 0.3f}'.format})

histogram_options = np.zeros(20, dtype=np.int)
histogram_distance = np.zeros(30, dtype=np.int)
duration_paretos = 0
duration_minimax = 0
distance_paretos = 0
distance_minimax = 0
matches = 0

for i in range(1000):
    # run the experiment
    config.num_nodes = 20
    state = ChainState(config)
    state.generate_graph()
    state.calculate_results()

    # do the pareto calculation
    start_time = time.time()
    paretos_efficient = state.pareto_defense_actions()
    duration_paretos += time.time() - start_time

    # do the minimax calculation
    start_time = time.time()
    minimax_efficient = state.minimax()
    duration_minimax += time.time() - start_time

    # calculate the minimum distances
    paretos_distance = state.calculate_average_distance(paretos_efficient)
    minimax_distance = state.calculate_average_distance(minimax_efficient)
    p_index = int((paretos_distance - minimax_distance) * 10 ) + 15 
    if p_index >= 30:
        p_index = 29
        print ("error p = {}".format(p_index))
    if p_index < 0:
        p_index = 0
        print ("error p = {}".format(p_index))
        
    # add the options to the histogram
    histogram_options[np.count_nonzero(paretos_efficient) - 1] += 1
    histogram_distance[p_index] += 1


    # add the total matches
    if np.any(paretos_efficient & minimax_efficient):
        matches += 1

print ("---------")
print (matches)
print (histogram_options)
print (histogram_distance)
print (duration_paretos)
print (duration_minimax)

