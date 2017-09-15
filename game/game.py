# main.py

import random
import time
import numpy as np
from state import Config, State
from reader import StateReader

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import RMSprop

class LogObject(object):
    """Save some text files"""
    def __init__(self):
        # output strings
        self.output_string = "\n"
        self.output_string2 = "\n"
        self.output_string3 = "\n"
        self.vector_reward_sum = np.zeros(3, dtype=np.int)
        self.reward_sum = 0
        self.step_count = 0
        self.chosen_action = ""

def create_model(input_state, layer1=450, layer2=350):
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

def calculate_offset(reader):

    # initial variables
    offset = np.zeros(3, dtype=int)
    offset_sum = np.zeros(3, dtype=int)
    min_score_sum = np.zeros(3, dtype=int)
    max_score_sum = np.zeros(3, dtype=int)
    range_max = np.zeros(3, dtype=int)
    times_to_run = 50

    # calculate the offset
    for i in range(times_to_run):
        input_state = reader.read_state()
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

def calculate_max(array_containing_max):
    """Pick a number from the highest 5 values"""
    
    chosen_index = np.argmax(array_containing_max)
    choices = np.zeros(5, dtype=np.int)
    max_values = np.zeros(5, dtype=float)

    for j in range(5):
        choices[j] = np.argmax(array_containing_max)
        max_values[j] = (0 if j == 0 else max_values[j-1]) + array_containing_max[choices[j]]
        array_containing_max[choices[j]] = 0

    random_index = random.random()
    for j in range(5):
        if random_index < (max_values[j] / max_values[4]):
            chosen_index = choices[j]
            break
    return chosen_index

def run_game(epsilon, reader, model, log_object, final_offset, normalizer, run_type=0, pareto_filter=None, single_action=False, default_action=0):
    # reset game for the next epoch
    gamma = 0.1 # since immediate rewards are more important keep gamma low
    steps = 0
    reward_sum = 0
    score_float = np.zeros(3, dtype=float)
    vector_reward_sum = np.zeros(3, dtype=np.int)
    state = reader.read_state()
    nn_input_old = np.zeros(state.size_graph + 2, dtype=np.int) # +2 for the game points
    y = np.zeros((1, state.size_graph+1))
    choices = np.zeros(5, dtype=np.int)
    max_search = np.zeros(state.size_graph+1, dtype=float)
    top_max = 0
    action_def = 0
    log_object.chosen_action = ""
    check_one = np.zeros(3, dtype=float)
    check_two = np.zeros(3, dtype=float)
    check_three = np.zeros(3, dtype=float)

    if run_type == 0:
        epsilon = 1

    # run the game
    while state.get_points(True) > 0 and state.get_points(False) > 0 and steps < 200:
        # find the Q-values for the state-action pairs
        q_table = model.predict(state.nn_input.reshape(1, state.nn_input.size), batch_size=1)

        if run_type == 2:
            if steps == 0 and pareto_filter is not None:
                state.actions_pareto_def = pareto_filter
            else:
                state.pareto_defense_actions()
        else:
            state.actions_pareto_def = state.actions_def

        # choose a defense action
        if single_action:
            action_def = default_action
            steps = 200
        elif random.random() < epsilon: # random action
            for j in range(100):
                action_def = np.random.randint(0, state.size_graph)
                if state.actions_pareto_def[action_def] == 1:
                    break
        else: # from Q(s,a) values
            action_def = np.argmax(np.multiply(state.actions_pareto_def, q_table[0] - min(q_table[0])))

        # choose an attack action
        for j in range(100):
            action_att = np.random.randint(0, state.size_graph)
            if state.actions_att[action_att] == 1:
                break

        # Take actions, observe new state
        np.copyto(nn_input_old, state.nn_input)
        state.make_move(action_att, action_def)

        # Update the score
        score_now = state.get_score(True)
        score_old = state.get_score(False)
        score = np.subtract(score_now, [0, 0, score_old[2]])
        vector_reward_sum += score
        np.copyto(check_one, score)

        # calculate the reward
        score = np.subtract(score, final_offset)
        np.copyto(check_two, score)
        score = np.divide(score * 100, normalizer)
        reward = np.dot(state.config.scalarization, score) / np.sum(state.config.scalarization)
        np.copyto(check_three, score)

        # Get max_Q(S',a)
        if run_type > 0:
            q_table_new_state = model.predict(state.nn_input.reshape(1, state.nn_input.size), batch_size=1)
            maxQ = np.max(q_table_new_state)

            # update the q_table
            update = (reward + (gamma * maxQ))
            q_table[0][action_def] =update
            model.fit(nn_input_old.reshape(1, state.nn_input.size), q_table, batch_size=1, epochs=1, verbose=0)

        # move to the next state
        reward_sum += reward
        steps += 1
        log_object.chosen_action += "{0}|".format(action_def)

        # output some data
        # if not single_action:
        #     type_of_att = -1 if action_att == state.size_graph else int(action_att % state.size_graph_cols < state.size_graph_col1)
        #     type_of_def = -1 if action_def == state.size_graph else int(action_def % state.size_graph_cols < state.size_graph_col1)
        #     log_object.output_string2 += ("{0}) {1} {2} : {3} {4} : {5} {6} {7} \n".format(steps,
        #                                     action_att, action_def, type_of_att, type_of_def,
        #                                     check_one.astype(int),
        #                                     check_three.astype(int), int(reward)))
            # log_object.output_string2 += ("{0}) {1} : {2} {3} {4} {5} \n".format(steps, action_def,
            #                                 check_one.astype(int),
            #                                 check_two.astype(int),
            #                                 check_three.astype(int), int(reward)))
            # log_object.output_string2 += ("{0}) {1} : {2!r} \n".format(steps, action_def, q_table.astype(int).tolist()))
            # q_table = model.predict(nn_input_old.reshape(1, state.nn_input.size), batch_size=1)
            # log_object.output_string2 += ("{0}) {1} : {2!r} \n".format(steps, action_def, q_table.astype(int).tolist()))

    log_object.reward_sum = reward_sum
    log_object.vector_reward_sum = vector_reward_sum
    log_object.step_count = steps
    return log_object

def run_epochs(reader, epochs, run_type=0, pareto_filter=None):

    # read and write existing state
    state = reader.read_state()
    log_object = LogObject()
    final_offset, normalizer = calculate_offset(reader)

    avg_sum = 0
    avg_vector_sum = np.zeros(3, dtype=np.int)

    # create the DQN
    model = create_model(state)

    print ("------------------ configuration {} ({} : {}) {} ------------------".format(reader.default_file_name, state.config.num_nodes, state.config.sparcity, run_type))

    # run the experiment
    start_time = time.time()
    for i in range(state.size_graph + 1):

        # go through every action
        log_object = run_game(0, reader, model, log_object, final_offset, normalizer, run_type, pareto_filter, True, state.size_graph - i)
        # save the output data
        # log_object.output_string += "{0}) {1} {2:03d}, {3!r}\n".format(log_object.step_count, log_object.chosen_action,
        #                             int(log_object.reward_sum), log_object.vector_reward_sum.astype(int).tolist())

    print ("--- run 0 %s seconds ---" % (time.time() - start_time))
    start_time = time.time()

    for i in range(epochs):

        # reset game for the next epoch
        epsilon = (1 - (i / epochs)) if i < (epochs * 3 / 5)  else 0.2
        log_object = run_game(epsilon, reader, model, log_object, final_offset, normalizer, run_type, pareto_filter)

        # save the output data
        avg_sum += log_object.reward_sum
        avg_vector_sum += log_object.vector_reward_sum
        if (i % (epochs / 100)) == 0:
            log_object.output_string += "{0}\n".format(int((avg_sum * 100) / epochs))
            log_object.output_string2 += "{0!r}\n".format(np.divide(log_object.vector_reward_sum * 100, epochs).astype(int).tolist())
            log_object.output_string3 += "({0}) {1} \n".format(log_object.step_count, log_object.chosen_action)
            avg_vector_sum = np.zeros(3, dtype=np.int)
            avg_sum = 0

    # output the data
    print ("--- run 1 %s seconds ---" % (time.time() - start_time))
    print (log_object.output_string)
    print ("------------------")
    print (log_object.output_string2)
    print ("------------------")
    # print (log_object.output_string3)
    # print ("------------------")


#------- START MAIN CODE --------

# Configuration
config = Config()
config.num_service = 3
config.num_viruses = 1
config.num_datadir = 1
config.offset = np.zeros(3, dtype=np.int)

node_options = [50, 100, 250, 500]
# node_options = [10, 30]
points_options = [100, 400, 1000, 2000]
sparse_options = [0.001, 0.005, 0.01, 0.05]
# sparse_options = [0.001, 0.005]
epochs_options = [200, 200, 200, 200]

reader_files = []

# create the states
for i in enumerate(node_options):
    config.num_nodes = i[1]
    config.att_points = points_options[i[0]]
    config.def_points = points_options[i[0]]
    for j in enumerate(sparse_options):
        config.sparcity = j[1]
        reader = StateReader("state_{}_{}.csv".format(i[0], j[0]))
        reader.write_state(State(config))
        reader_files.append(reader.default_file_name)

# run the game with the states
for i in enumerate(node_options):
    for j in enumerate(sparse_options):
        # read the state and generate the starting pareto front
        reader = StateReader(reader_files[(i[0] * len(sparse_options)) + j[0]])
        starting_pareto_filter = reader.read_state().pareto_defense_actions()
        for k in range(3):
            run_epochs(reader, epochs_options[i[0]], k, starting_pareto_filter)


# config.scalarization = np.array([0, 0, 10], dtype=np.int)

