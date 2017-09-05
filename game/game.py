# main.py

import random
import time
import numpy as np
from state import Config, State
from reader import StateReader

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import RMSprop

# create a new state
config = Config()
config.num_service = 3
config.num_viruses = 1
config.num_datadir = 1
config.num_nodes = 250
config.sparcity = 0.1
config.att_points = 100
config.def_points = 100

state = State(config)
reader = StateReader()
# reader.write_state(state)
state = reader.read_state()


# create the DQN
model = Sequential()
model.add(Dense(units=400, input_dim=state.nn_input.size))
model.add(Activation('relu'))

model.add(Dense(units=350))
model.add(Activation('relu'))

model.add(Dense(units=(state.size_graph+1)))
model.add(Activation('linear'))

model.compile(optimizer='rmsprop', loss='mse')

q_table = model.predict(state.nn_input.reshape(1, state.nn_input.size), batch_size=1)
model.fit(state.nn_input.reshape(1, state.nn_input.size), q_table, batch_size=1, epochs=1, verbose=0)

print state.get_graph()
print state.get_weight()
print "Nodes: {}, Services: {}, Viruses: {}, Datadir: {}, Edges: {}".format(state.config.num_nodes, state.config.num_service, 
                                                                            state.config.num_viruses, state.config.num_datadir, state.size_graph_edges)
print "Cost Attack: {}, Cost Defense: {}".format(state.att_cost, state.def_cost)
print "Game points: {}, {} | State score: {} -> {} | Maintenance: {}".format(state.get_points(False), state.get_points(True), 
                                                                        state.score_old, state.get_score(), state.maintenance_cost)

print "---------"

output_string = "\n"

start_time = time.time()

failsafe = 0
reward_sum = 0
avg_sum = 0
act1 = 0
act2 = 0

epochs = 5000
gamma = 0.1 # since it may take several moves to goal, making gamma high
epsilon = 0
for i in range(epochs):

    # epsilon = 0.1 if i < 500 else 0.8
    epsilon = (1 - (i * 1. / epochs)) if i < (epochs * 3 / 5)  else 0.2
    state = reader.read_state()
    nn_input_old = np.zeros(state.size_graph + 2, dtype=np.int) # +2 for the game points
    y = np.zeros((1, state.size_graph+1))

    # q_table = model.predict(state.nn_input.reshape(1, state.nn_input.size), batch_size=1)
    # output_string += "{0}) {1} | {2}: {3!r}\n".format(i,
    #                                         int(reward_sum),
    #                                         np.argmax(q_table[0] - min(q_table[0])),
    #                                         np.array(q_table[0], dtype=np.int).tolist())

    if i % 50 == 0:
        output_string += "{0}\n".format(int(avg_sum / 50))
        avg_sum = 0
    else:
        avg_sum += reward_sum
    
    # while game still in progress
    failsafe = 0
    reward_sum = 0

    while state.get_points(True) > 0 and state.get_points(False) > 0 and failsafe < 200:
        # find the Q-values for the state-action pairs
        q_table = model.predict(state.nn_input.reshape(1, state.nn_input.size), batch_size=1)

        # choose a defense action
        action_def = np.random.randint(0, state.size_graph)
        if random.random() < epsilon: # random action
            j = 0
            while state.actions_def[action_def] == 0 and j < 100:
                action_def = np.random.randint(0, state.size_graph)
                j += 1
        else: # from Q(s,a) values
            action_def = np.argmax(np.multiply(state.actions_def, q_table[0] - min(q_table[0])))

        # choose an attack action
        action_att = np.random.randint(0, state.size_graph)
        j = 0
        while state.actions_att[action_att] == 0 and j < 100:
            action_att = np.random.randint(0, state.size_graph)
            j += 1

        # Take actions, observe new state
        np.copyto(nn_input_old, state.nn_input)
        state.make_move(action_att, action_def)

        if failsafe == 0:
            act1 = action_att
            act2 = action_def

        # Observe reward
        score_now = state.get_score(True)
        score_old = state.get_score(False)
        score = np.subtract(state.get_score(), [0, 0, score_old[2]])
        reward = np.dot(state.config.scalarization, score) / np.sum(state.config.scalarization)

        # Get max_Q(S',a)
        q_table_new_state = model.predict(state.nn_input.reshape(1, state.nn_input.size), batch_size=1)
        maxQ = np.max(q_table_new_state)

        # # update the q_table
        y[:] = q_table[:]
        update = (reward + (gamma * maxQ))
        y[0][action_def] = update
        model.fit(nn_input_old.reshape(1, state.nn_input.size), y, batch_size=1, epochs=1, verbose=0)

        reward_sum += reward
        # output_string += "{0}) {1!r} {2} | {3}\n".format(failsafe%10, state.nn_input.tolist(), action_def, action_att)

        # update the state
        failsafe += 1

    # output_string += "{0}) {1!r}\n".format(failsafe%10, state.nn_input.tolist())
    # output_string += "({},{}), ".format(reward_sum, failsafe)

    # print output_string

print ("--- %s seconds ---" % (time.time() - start_time))
print output_string