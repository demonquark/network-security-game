# test for game.py

from game import State, Config
import numpy as np
import time
import random

# create a new state
state = State(Config())

# print (state.get_actions(True).astype(np.int))
# print (state.get_actions(False).astype(np.int))
print (state.actions_att.astype(np.int))
print (state.actions_def.astype(np.int))


# start_time = time.time()
# cross_product = np.multiply(state.actions_att.astype(np.int), state.actions_def.astype(np.int))
# print ("---calculate scores %s seconds ---" % (time.time() - start_time))

a = (state.size_graph + 1) * (state.size_graph + 1)
b = np.sum(state.actions_att.astype(np.int)) * np.sum(state.actions_def.astype(np.int))

print ("{} {} {} ".format(a, b, b/a))

print ("Testings World")
