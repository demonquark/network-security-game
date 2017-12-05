# File: game.py
# Class file for the Game object and LogObject
# - contains the the 
# - child of the State class (see state.py)
# - goal functions determined by a chaos function


import time
import numpy as np

from keras.models import Sequential
from keras.layers.core import Dense, Activation

from . import Config, State
from . import StateReader
from . import ChaosState


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


class GameObject(object):
    """Run the actual game"""
    def __init__(self):
        # output strings
        self.log_object = LogObject()
        self.model = None
        self.reader = None
        self.offset = None
        self.normalizer = None

    def create_model(self, input_state, layer1=450, layer2=350):
        """Create a ML model"""
        # create the DQN
        self.model = Sequential()
        self.model.add(Dense(units=layer1, input_dim=input_state.nn_input.size))
        self.model.add(Activation('relu'))

        self.model.add(Dense(units=layer2))
        self.model.add(Activation('relu'))

        self.model.add(Dense(units=(input_state.size_graph+1)))
        self.model.add(Activation('linear'))

        self.model.compile(optimizer='rmsprop', loss='mse')

        self.model.predict(input_state.nn_input.reshape(1, input_state.nn_input.size), batch_size=1)

    def calculate_offset(self):
        """Calculate the offset for the reward output"""

        # make sure there is a valid reader file
        if self.reader is None:
            return np.zeros(3, dtype=int), np.zeros(3, dtype=int)

        # initial variables
        offset = np.zeros(3, dtype=int)
        min_score_sum = np.zeros(3, dtype=int)
        range_max = np.zeros(3, dtype=int)
        times_to_run = 50
        in_state = self.reader.read_state()
        default_reward_matrix = in_state.reward_matrix if isinstance(in_state, ChaosState) else None

        # calculate the offset
        for i in range(times_to_run):
            in_state = self.reader.read_state(default_reward_matrix)
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

    def run_game(self, epsilon, state, run_type=0, pareto_filter=None, default_action=-1):
        """"Run a Game"""
        # reset game for the next epoch
        gamma = 0.1  # since immediate rewards are more important keep gamma low
        steps = 0
        action_def = 0
        action_att = 0
        reward_sum = 0
        vector_reward_sum = np.zeros(3, dtype=np.int)
        scalar_att = state.config.scalarize_att
        att_mag = np.sqrt(np.einsum('i,i', scalar_att, scalar_att))
        nn_input_old = np.zeros(state.size_graph + 2, dtype=np.int)  # +2 for the game points

        # run_type == 0 (RANDOM) means random defender actions
        if run_type == 0:
            epsilon = 1

        # log variables
        self.log_object.chosen_action = ""
        check_one = np.zeros(3, dtype=float)
        check_two = np.zeros(3, dtype=float)
        check_three = np.zeros(3, dtype=float)

        # make sure that there is a reward matrix
        if isinstance(state, State) and not isinstance(state, ChaosState):
            state.reset_reward_matrix()

        # run the game
        while state.get_points(True) > 0 and state.get_points(False) > 0 and steps < 200:
            # find the Q-values for the state-action pairs
            q_table = self.model.predict(state.nn_input.reshape(1, state.nn_input.size), batch_size=1)

            # guess an attack action
            # determine the valid attack actions
            num_moves = state.size_graph + 1
            indices = np.zeros(num_moves, dtype=np.int)
            for i in range(num_moves):
                indices[i] = i
            indices = indices[state.actions_att]

            # assume a random defense action
            assumed_def = state.size_graph
            for j in range(100):
                assumed_def = np.random.randint(0, state.size_graph)
                if state.actions_pareto_def[assumed_def] == 1:
                    break

            # determine the attacker reward for the assumed defense action
            rew_att = np.zeros(num_moves * 3, np.int).reshape(num_moves, 3)
            if isinstance(state, State):
                if isinstance(state, ChaosState):
                    np.copyto(rew_att, state.reward_matrix[assumed_def * num_moves:(assumed_def+1) * num_moves])
                else:
                    np.copyto(rew_att, state.reward_matrix[assumed_def, :, :])

            # calculate the cosine of the angle between the attacker scalarization and the rewards
            cosine = np.zeros(len(indices), dtype=float)
            for i in range(len(indices)):
                cosine[i] = np.absolute(np.dot(scalar_att, rew_att[indices[i]]) / (att_mag * np.sqrt(
                    np.einsum('i,i', rew_att[indices[i]], rew_att[indices[i]]))))

            # choose the strategy closest to the attacker scalarization
            action_att = indices[np.argmax(cosine)]

            # determine the valid defense actions based on the model and algorithm
            if isinstance(state, ChaosState):

                # determine the valid moves by using the chaos state method
                state.scalarized_attack_actions(action_att, run_type)
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

            # choose an actual attack action
            if np.random.rand() < (0.45 if run_type == 4 else 0.3):
                action_att = indices[np.random.randint(0, len(indices))]

            # choose a defense action
            if default_action >= 0:
                # default action assigned means we just want to run a single action once
                action_def = default_action
                steps = 200
            elif np.random.rand() < epsilon:
                # random action
                for j in range(100):
                    action_def = np.random.randint(0, state.size_graph)
                    if state.actions_pareto_def[action_def] == 1:
                        break
            else:
                # from Q(s,a) values
                action_def = np.argmax(np.multiply(state.actions_pareto_def, q_table[0] - min(q_table[0])))

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
            score = np.subtract(score, self.offset)
            np.copyto(check_two, score)
            score = np.divide(score * 100, self.normalizer)
            reward = np.dot(state.config.scalarization, score) / np.sum(state.config.scalarization)
            # reward = np.dot(goal_scalarization, score) / np.sum(goal_scalarization)
            np.copyto(check_three, score)

            # Get max_Q(S',a)
            if run_type > 0:
                q_table_new_state = self.model.predict(state.nn_input.reshape(1, state.nn_input.size), batch_size=1)
                maxQ = np.max(q_table_new_state)

                # update the q_table
                update = (reward + (gamma * maxQ))
                q_table[0][action_def] = update
                self.model.fit(nn_input_old.reshape(1, state.nn_input.size), q_table, batch_size=1, epochs=1, verbose=0)

            # move to the next state
            reward_sum += reward
            steps += 1
            self.log_object.chosen_action += "{0}-{1}|".format(action_att, action_def)

            # output additional data
            # if default_action == -1:

            #     # get the attack and defense type
            #     type_of_att = -1
            #     type_of_def = -1

            #     if action_att != state.size_graph:
            #         type_of_att = int(action_att % state.size_graph_cols < state.size_graph_col1)
            #     if action_def != state.size_graph:
            #         type_of_def = int(action_att % state.size_graph_cols < state.size_graph_col1)

            #     # output the check one, two and three information
            #     self.log_object.output_string2 += ("{0}) {1} {2} : {3} {4} : {5} {6} {7} \n".format(
            #         steps, action_att, action_def, type_of_att, type_of_def,
            #         check_one.astype(int), check_three.astype(int), int(reward)))

            #     self.log_object.output_string2 += ("{0}) {1} : {2} {3} {4} {5} \n".format(
            #         steps, action_def, check_one.astype(int), check_two.astype(int),
            #         check_three.astype(int), int(reward)))

            #     # output the q_table information
            #     self.log_object.output_string2 += ("{0}) {1} : {2!r} \n".format(
            #         steps, action_def, q_table.astype(int).tolist()))
            #     q_table = self.model.predict(nn_input_old.reshape(1, state.nn_input.size), batch_size=1)
            #     self.log_object.output_string2 += ("{0}) {1} : {2!r} \n".format(
            #         steps, action_def, q_table.astype(int).tolist()))

        # update the log object
        self.log_object.reward_sum = reward_sum
        self.log_object.vector_reward_sum = vector_reward_sum
        self.log_object.step_count = steps

    def run_epochs(self, in_reader, epochs, run_type=0):
        """Run the game for the given number of epochs"""

        # assign variables
        self.log_object = LogObject()
        self.reader = in_reader

        # read state and reset variables
        avg_sum = 0
        avg_vector_sum = np.zeros(3, dtype=np.int)
        state = self.reader.read_state()
        self.offset, self.normalizer = (0, 100) if isinstance(state, ChaosState) else self.calculate_offset()
        default_reward_matrix = state.reward_matrix if isinstance(state, ChaosState) else None
        pareto_filter = state.pareto_defense_actions()

        # create the DQN
        self.create_model(state)

        # start the runs
        print("------------------ configuration {} ({} : {}) {} ------------------".format(
            self.reader.default_file_name, state.config.num_nodes, state.config.sparcity, run_type))

        # dry run of the experiment to offset first move bias
        start_time = time.time()
        for i in range(state.size_graph + 1):
            state = self.reader.read_state(None, default_reward_matrix)
            self.run_game(0, state, run_type, pareto_filter, state.size_graph - i)
        print("--- run 0 %s seconds ---" % (time.time() - start_time))

        # actual run of the experiment
        start_time = time.time()
        for i in range(epochs):

            # play the game
            state = self.reader.read_state(None, default_reward_matrix)
            epsilon = (1 - (i / epochs)) if i < (epochs * 3 / 5) else 0.2
            self.run_game(epsilon, state, run_type, pareto_filter)

            # save the output data
            avg_sum += self.log_object.reward_sum
            avg_vector_sum += self.log_object.vector_reward_sum
            if (i % (epochs / 100)) == 0:
                self.log_object.output_string += "{0}\n".format(int((avg_sum * 100) / epochs))
                self.log_object.output_string2 += "{0!r}\n".format(
                    np.divide(self.log_object.vector_reward_sum * 100, epochs).astype(int).tolist())
                self.log_object.output_string3 += "({0}) {1} \n".format(
                    self.log_object.step_count, self.log_object.chosen_action)
                avg_vector_sum = np.zeros(3, dtype=np.int)
                avg_sum = 0

        # output the data
        print("--- run 1 %s seconds ---" % (time.time() - start_time))
        print(self.log_object.output_string)
        print("------------------")
        print(self.log_object.output_string2)
        print("------------------")
        # print (log_object.output_string3)
        # print ("------------------")
