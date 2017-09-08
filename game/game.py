# main.py

import random
import time
import numpy as np
from state import Config, State
from reader import StateReader

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import RMSprop

def create_model(input_state, layer1=300, layer2=250):
    # create the DQN
    model = Sequential()
    model.add(Dense(units=layer1, input_dim=input_state.nn_input.size))
    model.add(Activation('relu'))

    model.add(Dense(units=layer2))
    model.add(Activation('relu'))

    model.add(Dense(units=(input_state.size_graph+1)))
    model.add(Activation('linear'))

    model.compile(optimizer='rmsprop', loss='mse')

    model.predict(input_state.nn_input.reshape(1, input_state.nn_input.size), batch_size=1)

    return model

def calculate_offset(file_name=None):

    # initial variables
    offset = np.zeros(3, dtype=int)
    offset_sum = np.zeros(3, dtype=int)
    min_score_sum = np.zeros(3, dtype=int)
    max_score_sum = np.zeros(3, dtype=int)
    range_max = np.zeros(3, dtype=int)
    times_to_run = 50

    # calculate the offset
    for i in range(times_to_run):
        input_state = reader.read_state(file_name)
        min_score = np.zeros(3, dtype=int)
        max_score = np.zeros(3, dtype=int)
        steps = 0

        while input_state.get_points(True) > 0 and input_state.get_points(False) > 0 and steps < 200:

            # choose a defense action
            for j in range(100):
                action_def = np.random.randint(0, input_state.size_graph)
                if input_state.actions_def[action_def] == 1:
                    break

            # choose an attack action
            for j in range(100):
                action_att = np.random.randint(0, input_state.size_graph)
                if input_state.actions_att[action_att] == 1:
                    break

            # Take actions, observe new state
            input_state.make_move(action_att, action_def)

            # Observe reward
            score_old = input_state.get_score(False)
            score = np.subtract(input_state.get_score(), [0, 0, score_old[2]])
            min_score = np.minimum(min_score, score) if steps > 0 else score
            max_score = np.maximum(max_score, score) if steps > 0 else score
            range_max = np.maximum(range_max, score) if i > 0 else score

            # print ("{3}) {0} {1} {2} {4} {5}".format(min_score, max_score, range_max, steps, score_old, score))

            # next step
            steps += 1

        # calculate this game's offset
        min_score_sum += min_score
        offset += max_score - min_score

    # calculate the average offset
    offset = np.divide(min_score_sum, times_to_run) - np.divide(offset, times_to_run)
    offset = np.maximum(offset, np.zeros(3, dtype=np.int)) 
    # print ("--------")
    # print ("normalizer: {0} ".format(range_max))
    # print ("offset: {0} ".format(offset))
    range_max = np.maximum(range_max - offset, np.zeros(3, dtype=np.int))

    # return the offset
    return offset, range_max

def print_graph(s_out):
    print (s_out.get_graph())
    print (s_out.get_weight())
    print ("Nodes: {}, Services: {}, Viruses: {}, Datadir: {}, Edges: {}".format(s_out.config.num_nodes, s_out.config.num_service, 
                                                                                s_out.config.num_viruses, s_out.config.num_datadir, s_out.size_graph_edges))
    print ("Cost Attack: {}, Cost Defense: {}".format(s_out.att_cost, s_out.def_cost))
    print ("Game points: {}, {} | State score: {} -> {} | Maintenance: {}".format(s_out.get_points(False), s_out.get_points(True), 
                                                                        s_out.score_old, s_out.get_score(), s_out.maintenance_cost))
    print ("---------")

# variables
epochs = 50
gamma = 0.1 # since immediate rewards are more important keep gamma low
epsilon = 0
steps = 0
reward_sum = 0
avg_sum = 0
output_string = "\n"
output_string2 = "\n"
output_string3 = "\n"


# Configuration
config = Config()
config.num_service = 3
config.num_viruses = 1
config.num_datadir = 1
config.num_nodes = 6
config.sparcity = 0.1
config.att_points = 200
config.def_points = 200
config.offset = np.zeros(3, dtype=np.int)
check_one = np.zeros(3, dtype=float)
check_two = np.zeros(3, dtype=float)
check_three = np.zeros(3, dtype=float)
config.scalarization = np.array([0, 0, 10], dtype=np.int)


reader = StateReader()
state = State(config)
reader.write_state(state)

# read an existing state
state = reader.read_state()

final_offset, normalizer = calculate_offset()

print (final_offset)
print (normalizer)

# create the DQN
model = create_model(state)

# run the experiment
start_time = time.time()

for i in range(epochs):

    # reset game for the next epoch
    epsilon = (1 - (i / epochs)) if i < (epochs * 3 / 5)  else 0.2
    steps = 0
    reward_sum = 0
    score_float = np.zeros(3, dtype=float)
    vector_reward_sum = np.zeros(3, dtype=np.int)
    state = reader.read_state()
    nn_input_old = np.zeros(state.size_graph + 2, dtype=np.int) # +2 for the game points
    y = np.zeros((1, state.size_graph+1))

    # run the game
    while state.get_points(True) > 0 and state.get_points(False) > 0 and steps < 200:
        # find the Q-values for the state-action pairs
        q_table = model.predict(state.nn_input.reshape(1, state.nn_input.size), batch_size=1)

        # choose a defense action
        if random.random() < epsilon: # random action
            for j in range(100):
                action_def = np.random.randint(0, state.size_graph)
                if state.actions_def[action_def] == 1:
                    break
        else: # from Q(s,a) values
            action_def = np.argmax(np.multiply(state.actions_def, q_table[0] - min(q_table[0])))

        # choose an attack action
        for j in range(100):
            action_att = np.random.randint(0, state.size_graph)
            if state.actions_att[action_att] == 1:
                break

        action_def = state.size_graph

        # Take actions, observe new state
        np.copyto(nn_input_old, state.nn_input)
        state.make_move(action_att, action_def)

        # Observe reward
        score_now = state.get_score(True)
        score_old = state.get_score(False)
        score = np.subtract(state.get_score(), [0, 0, score_old[2]])
        np.copyto(check_one, score)
        vector_reward_sum += score
        score = np.subtract(score, final_offset)
        np.copyto(check_two, score)
        np.copyto(score_float, score)
        score_float = np.divide(score_float, normalizer)
        score_float = np.multiply(score_float, 100)
        np.copyto(check_three, score_float)
        reward = np.dot(state.config.scalarization, score_float) / np.sum(state.config.scalarization)

        # output_string3 += ("{0}) {1} : {2} {3} {4} ".format(steps, action_def, check_one.astype(int), check_two.astype(int), check_three.astype(int)))
        # output_string3 += ("{0} {1} {2} \n".format(score_float.astype(int), state.config.scalarization, reward))

        # Get max_Q(S',a)
        q_table_new_state = model.predict(state.nn_input.reshape(1, state.nn_input.size), batch_size=1)
        maxQ = np.max(q_table_new_state)
        # output_string3 += ("{0!r} \n".format((q_table_new_state.astype(int)).tolist()))

        # update the q_table
        update = (reward + (gamma * maxQ))
        q_table[0][action_def] =update
        model.fit(nn_input_old.reshape(1, state.nn_input.size), q_table, batch_size=1, epochs=1, verbose=0)

        # move to the next state
        reward_sum += reward
        steps += 1
        output_string3 += ("{} \n".format(reward_sum)

    # output the data
    if (i % (epochs / 100)) == 0:
        output_string += "{0}\n".format(int((avg_sum * 100) / epochs))
        output_string2 += "{0!r}\n".format(np.divide(vector_reward_sum, 10).astype(int).tolist())
        avg_sum = 0
    else:
        avg_sum += reward_sum
    
    # output_string += "{0}) {1} | {2}: {3!r}\n".format(i,
    #                                         int(reward_sum),
    #                                         np.argmax(q_table[0] - min(q_table[0])),
    #                                         np.array(q_table[0], dtype=np.int).tolist())


print ("--- %s seconds ---" % (time.time() - start_time))
print (output_string)
print ("------------------")
print (output_string2)
print ("------------------")
# print (output_string3)



