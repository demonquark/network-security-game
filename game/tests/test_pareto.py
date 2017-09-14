# test for pareto.py

from game.state import State, Config
from game.reader import StateReader
from game.pareto import is_pareto_efficient_k0, is_pareto_sets_efficient
import numpy as np
import time
import random

# read and write existing state
reader = StateReader()
state = State(Config())
reader.write_state(state)
state.print_graph()
action_att = 0

def calc_score(state_x, action_a, action_d):
    """Read the data file and reset the score"""
    state_x = reader.read_state()
    score = state_x.make_move(action_a, action_d)
    return score

# choose an attack action
start_time = time.time()
actions = []
for i in range(6):
    for j in range(100):
        action_att = np.random.randint(0, state.size_graph)
        if state.actions_att[action_att] == 1:
            break
    actions.append(action_att)
# print ("---choose attack %s seconds ---" % (time.time() - start_time))
scores = np.array([([ calc_score(state, y, x) for x in range(state.size_graph + 1) if state.actions_def[x]]) for y in actions])
print ("---calculate scores %s seconds ---" % (time.time() - start_time))
print (scores)
print ("------------------")

# manually create scores
test_value1 = np.array([90, 120, 10])
test_value2 = np.array([80, 122, 12])
test_value3 = np.array([80, 120, 10])
test_value4 = np.array([80, 122, 10])
test_value5 = np.array([80, 120, 12])
test_matrix1 = np.array([test_value1, test_value2, test_value3, test_value4, test_value5])

test_value1 = np.array([90, 100, 10])
test_value2 = np.array([81, 122, 12])
test_value3 = np.array([81, 120, 10])
test_value4 = np.array([81, 122, 10])
test_value5 = np.array([81, 120, 12])
test_matrix2 = np.array([test_value1, test_value2, test_value3, test_value4, test_value5])

test_value1 = np.array([90, 120, 10])
test_value2 = np.array([80, 72, 12])
test_value3 = np.array([65, 120, 10])
test_value4 = np.array([80, 122, 10])
test_value5 = np.array([80, 120, 6])
test_matrix3 = np.array([test_value1, test_value2, test_value3, test_value4, test_value5])

test_defenses = np.array([test_matrix1, test_matrix2, test_matrix3])

# start_time = time.time()
# result2 = is_pareto_efficient_k0(test_defenses[0])
# print ("---modified pareto %s seconds ---" % (time.time() - start_time))
# print (result2)
# print ("------------------")
# result2 = is_pareto_efficient_k0(test_defenses[1])
# print ("---modified pareto %s seconds ---" % (time.time() - start_time))
# print (result2)
# print ("------------------")
# result2 = is_pareto_efficient_k0(test_defenses[2])
# print ("---modified pareto %s seconds ---" % (time.time() - start_time))
# print (result2)
# print ("------------------")

start_time = time.time()
pareto_fronts = np.array([score[is_pareto_efficient_k0(score)] for score in scores])
# print ("---modified pareto %s seconds ---" % (time.time() - start_time))
# print (pareto_fronts)
# print ("------------------")

# start_time = time.time()
is_efficient = is_pareto_sets_efficient(pareto_fronts)
print ("---modified pareto %s seconds ---" % (time.time() - start_time))
# print (is_efficient)
# print ("------------------")


# print out the valid actions
# outputstring = ""
# for i in range(state.size_graph + 1):
#     if state.actions_def[i]:
#         outputstring += "{}|".format(i)

# outputstring += "\n"
# print (outputstring)
# print ("------------------")


# test_value1 = np.array([90, 120, 10])
# test_value2 = np.array([80, 122, 12])
# test_value3 = np.array([80, 120, 10])
# test_value4 = np.array([80, 122, 10])
# test_value5 = np.array([80, 120, 12])
# test_matrix = np.array([test_value1, test_value2, test_value3, test_value4, test_value5])

# print (test_value1)
# print (test_value2)
# print (test_value3)
# print (test_value4)
# print (test_value5)

# print (test_matrix[np.ones(test_matrix.shape[0], dtype = bool)] >= test_matrix[0])
# print (np.any(test_matrix[np.ones(test_matrix.shape[0], dtype = bool)] >= test_matrix[0], axis=1))
# print (np.all(test_matrix[np.ones(test_matrix.shape[0], dtype = bool)] >= test_matrix[0], axis=1))

# print ("test1: {} {} {} {}".format(test_value1 >= test_value2, np.all(test_value1 >= test_value2), test_value1 < test_value2, not np.any(test_value1 < test_value2)))
# print ("test2: {} {} {} {}".format(test_value1 >= test_value3, np.all(test_value1 >= test_value3), test_value1 < test_value3, not np.any(test_value1 < test_value3)))
# print ("test3: {} {} {} {}".format(test_value1 >= test_value4, np.all(test_value1 >= test_value4), test_value1 < test_value4, not np.any(test_value1 < test_value4)))
# print ("test4: {} {} {} {}".format(test_value1 >= test_value5, np.all(test_value1 >= test_value5), test_value1 < test_value5, not np.any(test_value1 < test_value5)))
