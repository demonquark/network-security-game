# main.py

import random
import time
import numpy as np
from state import Config, State
from chaosstate import ChaosState
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
    """Create a ML model"""
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
    """Calculate the offset for the reward output"""
    # initial variables
    offset = np.zeros(3, dtype=int)
    min_score_sum = np.zeros(3, dtype=int)
    range_max = np.zeros(3, dtype=int)
    times_to_run = 50
    in_state = reader.read_state()
    default_reward_matrix = in_state.reward_matrix if isinstance(in_state, ChaosState) else None

    # calculate the offset
    for i in range(times_to_run):
        in_state = reader.read_state(default_reward_matrix)
        min_score = np.zeros(3, dtype=int)
        max_score = np.zeros(3, dtype=int)
        steps = 0

        while in_state.get_points(True) > 0 and in_state.get_points(False) > 0 and steps < 200:

            # choose a defense action
            for j in range(100):
                action_def = np.random.randint(0, in_state.size_graph)
                if in_state.actions_def[action_def] == 1:
                    break

            # choose an attack action
            for j in range(100):
                action_att = np.random.randint(0, in_state.size_graph)
                if in_state.actions_att[action_att] == 1:
                    break

            # Take actions, observe new state
            in_state.make_move(action_att, action_def)

            # Observe reward
            score_old = in_state.get_score(False)
            score = np.subtract(in_state.get_score(), [0, 0, score_old[2]])
            min_score = np.minimum(min_score, score) if steps > 0 else score
            max_score = np.maximum(max_score, score) if steps > 0 else score
            range_max = np.maximum(range_max, score) if i > 0 else score

            # next step
            steps += 1

        # calculate this game's offset
        min_score_sum += min_score
        offset += max_score - min_score

    # calculate the average offset
    offset = np.divide(min_score_sum, times_to_run) - np.divide(offset, times_to_run)
    offset = np.maximum(offset, np.zeros(3, dtype=np.int))
    range_max = np.maximum(range_max - offset, np.zeros(3, dtype=np.int))

    # return the offset
    return offset, range_max

def run_game(epsilon, state, model, log_object, final_offset, normalizer,
             run_type=0, pareto_filter=None, default_action=-1):
    """"Run a Game"""
    # reset game for the next epoch
    gamma = 0.1 # since immediate rewards are more important keep gamma low
    steps = 0
    action_def = 0
    action_att = 0
    reward_sum = 0
    vector_reward_sum = np.zeros(3, dtype=np.int)
    attacker_scalarization = np.array([7, 3, 0], dtype=np.int)
    nn_input_old = np.zeros(state.size_graph + 2, dtype=np.int) # +2 for the game points

    # run_type == 0 (RANDOM) means random defender actions
    if run_type == 0:
        epsilon = 1

    # loggin variables
    log_object.chosen_action = ""
    check_one = np.zeros(3, dtype=float)
    check_two = np.zeros(3, dtype=float)
    check_three = np.zeros(3, dtype=float)

    # run the game
    while state.get_points(True) > 0 and state.get_points(False) > 0 and steps < 200:
        # find the Q-values for the state-action pairs
        q_table = model.predict(state.nn_input.reshape(1, state.nn_input.size), batch_size=1)

        if isinstance(state, ChaosState):
            # determine the valid moves by using the chaos state method
            state.scalarized_attack_actions(attacker_scalarization, run_type)
        else:
            if run_type == 2:
                # use the precalculated pareto front for the first move
                if steps == 0 and pareto_filter is not None:
                    state.actions_pareto_def = pareto_filter
                else:
                    state.pareto_defense_actions()
            else:
                # use entire set
                state.actions_pareto_def = state.actions_def

        # choose a defense action
        if default_action >= 0:
            # default action assigned means we just want to run a single action once
            action_def = default_action
            steps = 200
        elif random.random() < epsilon: 
            # random action
            for j in range(100):
                action_def = np.random.randint(0, state.size_graph)
                if state.actions_pareto_def[action_def] == 1:
                    break
        else: 
            # from Q(s,a) values
            action_def = np.argmax(np.multiply(state.actions_pareto_def, q_table[0] - min(q_table[0])))

        # choose an attack action
        num_possible_moves = state.size_graph + 1
        indices = np.zeros(num_possible_moves, np.int)
        for i in range(num_possible_moves):
            indices[i] = i
        indices = indices[state.actions_att]
        
        reward_set = np.zeros(num_possible_moves * 3, np.int).reshape(num_possible_moves, 3)
        np.copyto(reward_set, state.reward_matrix[action_def * num_possible_moves:(action_def+1) * num_possible_moves])

        preferred_attacks = np.zeros(len(indices), np.int)
        output_text = ""
        for i in range(len(indices)):
            preferred_attacks[i] = np.dot(attacker_scalarization, reward_set[indices[i]])
        
        action_att = np.argmin(preferred_attacks)
        action_att = indices[action_att]

        # # for j in range(state.size_graph + 1):
        # #     if not state.actions_att[j]:
        # #         reward_set[j] = np.ones(3, np.int)

        # # actions_pareto_att = state._pareto_front_filter(reward_set)
        # for j in range(100):
        #     action_att = np.random.randint(0, state.size_graph)
        #     if state.actions_att[action_att] == 1:
        #         break

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
        # goal_scalarization = np.array([5, 5, 0], dtype=np.int)
        score = np.subtract(score, final_offset)
        np.copyto(check_two, score)
        score = np.divide(score * 100, normalizer)
        reward = np.dot(state.config.scalarization, score) / np.sum(state.config.scalarization)
        # reward = np.dot(goal_scalarization, score) / np.sum(goal_scalarization)
        np.copyto(check_three, score)

        # Get max_Q(S',a)
        if run_type > 0:
            q_table_new_state = model.predict(state.nn_input.reshape(1, state.nn_input.size), batch_size=1)
            maxQ = np.max(q_table_new_state)

            # update the q_table
            update = (reward + (gamma * maxQ))
            q_table[0][action_def] = update
            model.fit(nn_input_old.reshape(1, state.nn_input.size), q_table, batch_size=1, epochs=1, verbose=0)

        # move to the next state
        reward_sum += reward
        steps += 1
        log_object.chosen_action += "{0}-{1}|".format(action_att, action_def)

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

def run_epochs(reader, epochs, run_type=0):
    """Run the game for the given number of epochs"""

    # read state and reset variables
    avg_sum = 0
    avg_vector_sum = np.zeros(3, dtype=np.int)
    log_object = LogObject()
    state = reader.read_state()
    offset, normalizer = (0, 100) if  isinstance(state, ChaosState) else calculate_offset(reader)
    default_reward_matrix = state.reward_matrix if isinstance(state, ChaosState) else None
    pareto_filter = state.pareto_defense_actions()

    # create the DQN
    model = create_model(state)

    # start the runs
    print ("------------------ configuration {} ({} : {}) {} ------------------".format(
        reader.default_file_name, state.config.num_nodes, state.config.sparcity, run_type))

    # # dry run of the experiment to offset first move bias
    # start_time = time.time()
    # for i in range(state.size_graph + 1):
    #     state = reader.read_state(None, default_reward_matrix)
    #     log_object = run_game(0, state, model, log_object, offset, normalizer, run_type, pareto_filter, state.size_graph - i)
    # print ("--- run 0 %s seconds ---" % (time.time() - start_time))

    # actual run of the experiment
    start_time = time.time()
    for i in range(epochs):

        # play the game
        state = reader.read_state(None, default_reward_matrix)
        epsilon = (1 - (i / epochs)) if i < (epochs * 3 / 5)  else 0.2
        log_object = run_game(epsilon, state, model, log_object, offset, normalizer, run_type, pareto_filter)

        # save the output data
        avg_sum += log_object.reward_sum
        avg_vector_sum += log_object.vector_reward_sum
        if (i % (epochs / 100)) == 0:
            log_object.output_string += "{0}\n".format(int((avg_sum * 100) / epochs))
            log_object.output_string2 += "{0!r}\n".format(
                np.divide(log_object.vector_reward_sum * 100, epochs).astype(int).tolist())
            log_object.output_string3 += "({0}) {1} \n".format(
                log_object.step_count, log_object.chosen_action)
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
config.num_viruses = 3
config.num_datadir = 0
config.num_nodes = 3
config.offset = np.zeros(3, dtype=np.int)
config.scalarization = np.array([3, 7, 0], dtype=np.int)

node_options = [50, 100, 250, 500]
points_options = [60, 60, 60, 60]
sparse_options = [0.001, 0.005, 0.01, 0.05]
epochs_options = [200, 200, 200, 200]

node_options = [50]
sparse_options = [0.1]
points_options = [50]
epochs_options = [200]

reader_files = []


# create the states
for node_count in enumerate(node_options):
    config.num_nodes = node_count[1]
    config.att_points = points_options[node_count[0]]
    config.def_points = points_options[node_count[0]]
    for sparsity in enumerate(sparse_options):
        config.sparcity = sparsity[1]
        in_reader = StateReader("state_{}_{}.csv".format(node_count[0], sparsity[0]))
        # in_reader.write_state(State(config))
        in_reader.write_state(ChaosState(config))
        reader_files.append(in_reader.default_file_name)


# run the game with the states
for node_count in enumerate(node_options):
    for sparsity in enumerate(sparse_options):
        # read the state and generate the starting pareto front
        in_reader = StateReader(reader_files[(node_count[0] * len(sparse_options)) + sparsity[0]])
        for k in range(7):
            run_epochs(in_reader, epochs_options[node_count[0]], k)


# config.scalarization = np.array([0, 0, 10], dtype=np.int)
