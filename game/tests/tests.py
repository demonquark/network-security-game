# tests for the game module


import time
import numpy as np
from game import Config, State, StateReader
from test_state import TestState
from test_reader import TestStateReader

config = Config()
config.num_service = 4
config.num_viruses = 1
config.num_datadir = 1
config.num_nodes = 5
config.sparcity = 0.1
config.att_points = 100
config.def_points = 100

config.ratios = np.array([3, 3, 2], dtype=np.int)


# Generate a graph
test_case1 = TestState(config)

# Write the state to file
test_case2 = TestStateReader()
# test_case2.write_state(test_case1.state)

# Read the graph
# test_case1.state = test_case2.read_state()
test_case1.print_graph()
print (test_case1.state.graph_edges)

start_time = time.time()

# make a move
print ("Possible attack actions: {}".format(np.sum(test_case1.state.actions_att)))
print ("Possible defence actions: {}".format(np.sum(test_case1.state.actions_def)))
# test_case1.test_make_move()
# print ("Possible attack actions: {}".format(np.sum(test_case1.state.actions_att)))
# print ("Possible defence actions: {}".format(np.sum(test_case1.state.actions_def)))

print ("--- Generate: %s seconds ---" % (time.time() - start_time))
# test_case1.print_graph()



























#--------------State and Reader-------------#



# # print the graph
# test_case1.print_graph()

# # Write the state to file
# start_time = time.time()

# test_case2 = TestStateReader()
# test_case2.write_state(test_case1.state)

# print ("--- Write: %s seconds ---" % (time.time() - start_time))


# test_case1.generate_graph()
# test_case1.generate_graph()
# test_case1.generate_graph()


# # Read from file
# start_time = time.time()


# print ("--- Read: %s seconds ---" % (time.time() - start_time))
# # print the graph
# test_case1.print_graph()
