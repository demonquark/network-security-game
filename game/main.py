# main.py

import random
import time
import numpy as np

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import RMSprop

from state import State, Config

# create a new state
config = Config()

# state = State(config, [1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, config.att_points, config.def_points])
state = State(config)

# create the DQN
model = Sequential()
model.add(Dense(units=500, input_dim=state.nn_input.size))
model.add(Activation('relu'))

model.add(Dense(units=200))
model.add(Activation('relu'))

model.add(Dense(units=state.size_graph))
model.add(Activation('linear'))

model.compile(optimizer='rmsprop', loss='mse')

q_table = model.predict(state.nn_input.reshape(1, state.nn_input.size), batch_size=1)
model.fit(state.nn_input.reshape(1, state.nn_input.size), q_table, batch_size=1, epochs=1, verbose=0)

# print state.get_graph()
# print state.get_points(True), " | ", state.get_points(False)
# print config.def_cost, " | ", config.att_cost
# print "---------"

output_string = "\n"

start_time = time.time()

epochs = 1
gamma = 0.9 # since it may take several moves to goal, making gamma high
epsilon = 0
for i in range(epochs):

    epsilon = 0.1 if epochs < 400 else 0.8
    # state = State(config, [1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, config.att_points, config.def_points])
    state = State(config)

    # while game still in progress
    failsafe = 0
    prev_action = -1

    while state.get_points(True) > 0 and state.get_points(False) > 0 and failsafe < 1:
        # find the Q-values for the state-action pairs
        q_table = model.predict(state.nn_input.reshape(1, state.nn_input.size), batch_size=1)

        # # choose a defense action
        # action_def = np.random.randint(0, state.size_graph)
        # if random.random() < epsilon: # random action
        #     j = 0
        #     while state.actions_def[action_def] == 0 and j < 100:
        #         action_def = np.random.randint(0, state.size_graph)
        #         j += 1
        # else: # from Q(s,a) values
        #     action_def = np.argmax(np.multiply(state.actions_def, q_table[0] - min(q_table[0])))

        # # choose an attack action
        # action_att = np.random.randint(0, state.size_graph)
        # j = 0
        # while state.actions_att[action_att] == 0 and j < 100:
        #     action_att = np.random.randint(0, state.size_graph)
        #     j += 1

        # # Take actions, observe new state
        # new_state = State(config, state.nn_input)
        # new_state.make_move(True, action_def)
        # new_state.make_move(False, action_att)

        # # Observe reward
        # score = state.get_score()
        # score = np.subtract(new_state.get_score(), [0, 0, score[2]])
        # reward = np.dot(state.config.weights, score) / np.sum(state.config.weights)

        # Get max_Q(S',a)
        # q_table_new_state = model.predict(
        #     new_state.nn_input.reshape(1, new_state.nn_input.size), batch_size=1)
        q_table_new_state = model.predict(
            state.nn_input.reshape(1, state.nn_input.size), batch_size=1)
        maxQ = np.max(q_table_new_state)

        # update the q_table
        y = np.zeros((1, state.size_graph))
        y[:] = q_table[:]
        # update = (reward + (gamma * maxQ))
        # y[0][action_def] = update
        update = (10 + (gamma * maxQ))
        y[0][10] = update
        model.fit(state.nn_input.reshape(1, state.nn_input.size), y,
                  batch_size=1, epochs=1, verbose=0)

        # output_string += "{0}) {1!r} {2} | {3}\n".format(failsafe%10, state.nn_input.tolist(), action_def, action_att)

        # update the state
        # state = new_state
        # state.reset_actions()
        failsafe += 1
        # prev_action = action_def

    # output_string += "{0}) {1!r}\n".format(failsafe%10, state.nn_input.tolist())
    output_string += "{}, ".format(failsafe)

    # print output_string

print ("--- %s seconds ---" % (time.time() - start_time))
print output_string