# File: state.py
# Class file for the state
# Includes two classes
# - Config (default configuration for the state)
# - State (contains the current state and possible actions on the state)

import csv
import ast
import numpy as np
from . import Config, State
from . import ChaosState
from . import ChainState


class StateReader(object):
    """Read config and state files"""
    def __init__(self, file_name=None):
        # initialize
        self.default_file_name = 'state.csv' if file_name is None else file_name

    def write_state(self, state, file_name=None):
        """Read a state file"""

        if isinstance(state, State):
            self.write_game_state(state, file_name)
        else:
            self.write_step_state(state, file_name)

    def read_state(self, file_name=None, default_reward_matrix=None):
        """Read a state file"""

        # revert to the default file
        if file_name is None:
            file_name = self.default_file_name

        # read from the given file
        state_type = -1
        with open(file_name, 'r') as csvfile:
            reader = csv.reader(csvfile, delimiter=';', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            state_type = next(reader)

        # read the file as per the state_type
        if int(state_type[0]) == 0 or int(state_type[0]) == 1:
            return self.read_game_state(file_name, default_reward_matrix)
        else:
            return self.read_step_state(file_name, default_reward_matrix)

    def write_game_state(self, state, file_name=None):
        """Read a state file"""

        # revert to the default file
        if file_name is None:
            file_name = self.default_file_name

        # write to the defined config file
        with open(file_name, 'w') as csvfile:
            writer = csv.writer(csvfile, delimiter=';', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            writer.writerow([1 if isinstance(state, ChaosState) else 0])
            writer.writerow([state.config.num_service])
            writer.writerow([state.config.num_viruses])
            writer.writerow([state.config.num_datadir])
            writer.writerow([state.config.num_nodes])

            writer.writerow([state.config.sparcity])
            writer.writerow([state.config.server_client_ratio])
            writer.writerow(state.config.ratios)

            writer.writerow(state.config.low_value_nodes)
            writer.writerow(state.config.high_value_nodes)

            writer.writerow([state.config.att_points])
            writer.writerow([state.config.def_points])
            writer.writerow(state.config.att_cost_weights)
            writer.writerow(state.config.def_cost_weights)

            writer.writerow(state.config.scalarization)

            writer.writerow(state.nn_input)
            writer.writerow(state.graph_edges)
            writer.writerow(state.graph_weights)
            if state.reward_matrix is not None:
                writer.writerow(state.reward_matrix.tolist())
            else:
                writer.writerow([0])

    def write_step_state(self, state, file_name=None):
        """Read a state file"""

        # revert to the default file
        if file_name is None:
            file_name = self.default_file_name

        # write to the defined config file
        with open(file_name, 'w') as csvfile:
            writer = csv.writer(csvfile, delimiter=';', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            writer.writerow([2])
            writer.writerow([state.config.size_def_strategies])
            writer.writerow([state.config.size_att_strategies])
            writer.writerow([state.config.num_nodes])
            writer.writerow([state.config.max_lfr])
            writer.writerow([state.config.alpha])
            writer.writerow([state.config.beta])

            writer.writerow(state.config.cap_values)
            writer.writerow(state.nodes_con)
            writer.writerow(state.nodes_cap)
            writer.writerow(state.nodes_acc)
            writer.writerow(state.edges_lfr)
            writer.writerow(state.edges_sco)
            writer.writerow(state.strat_def)

            writer.writerow(state.strat_att_chain)
            writer.writerow(state.strat_att_conn)

    def read_game_state(self, file_name=None, default_reward_matrix=None):
        """Read a state file"""

        # basic variables
        config = Config()
        state = None
        content = None

        # revert to the default file
        if file_name is None:
            file_name = self.default_file_name

        # read from the defined config file
        with open(file_name, 'r') as csvfile:
            reader = csv.reader(csvfile, delimiter=';', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            content = list(reader)

        if isinstance(content, list) and len(content) == 19:
            is_basic_state = (int(content[0][0]) == 0)
            config.num_service = int(content[1][0])
            config.num_viruses = int(content[2][0])
            config.num_datadir = int(content[3][0])
            config.num_nodes = int(content[4][0])

            config.sparcity = float(content[5][0])
            config.server_client_ratio = float(content[6][0])
            config.ratios = np.array(content[7], dtype=np.int)

            config.low_value_nodes = self.__list_of_list_from_string_list(content[8])
            config.high_value_nodes = self.__list_of_list_from_string_list(content[9])

            config.att_points = int(content[10][0])
            config.def_points = int(content[11][0])
            config.att_cost_weights = np.array(content[12], dtype=np.int)
            config.def_cost_weights = np.array(content[13], dtype=np.int)

            config.scalarization = np.array(content[14], dtype=np.int)

            default_input = np.array(content[15], dtype=np.int)
            default_edges = self.__list_of_list_from_string_list(content[16])
            default_graph_weights = np.array(content[17], dtype=np.int)

            if default_reward_matrix is None and not is_basic_state:
                intial_list = self.__list_of_list_from_string_list(content[18])
                default_reward_matrix = np.array(intial_list)

            if is_basic_state:
                state = State(config, default_input, default_edges, default_graph_weights)
            else:
                state = ChaosState(config, default_input, default_edges, default_graph_weights, default_reward_matrix)

        else:
            print("Could not read config file.")

        return state

    def read_step_state(self, file_name=None, default_reward_matrix=None):
        """Read a state file"""

        # basic variables
        config = Config()
        state = None
        content = None

        # revert to the default file
        if file_name is None:
            file_name = self.default_file_name

        # read from the defined config file
        with open(file_name, 'r') as csvfile:
            reader = csv.reader(csvfile, delimiter=';', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            content = list(reader)

        if isinstance(content, list):
            is_chain_state = (int(content[0][0]) == 2)
            config.size_def_strategies = int(content[1][0])
            config.size_att_strategies = int(content[2][0])
            config.num_nodes = int(content[3][0])
            config.max_lfr = float(content[4][0])
            config.alpha = float(content[5][0])
            config.beta = float(content[6][0])
            config.cap_values = self.__list_of_list_from_string_list(content[7])

            if is_chain_state:
                state = ChainState(config)
                state.nodes_con = np.array(self.__list_of_list_from_string_list(content[8]), dtype=np.int)
                state.nodes_cap = np.array(self.__list_of_list_from_string_list(content[9]), dtype=np.int)
                state.nodes_acc = np.array(self.__list_of_list_from_string_list(content[10]), dtype=np.int)
                state.edges_lfr = np.array(self.__list_of_list_from_string_list(content[11]), dtype=float)
                state.edges_sco = np.array(self.__list_of_list_from_string_list(content[12]), dtype=np.int)
                state.strat_def = np.array(self.__list_of_list_from_string_list(content[13]), dtype=np.int)

                state.strat_att_chain = self.__list_of_list_from_string_list(content[14])
                state.strat_att_conn = self.__list_of_np_arrays_from_string_list(content[15])
                state.calculate_defense()
                state.calculate_results()

        else:
            print("Could not read config file.")

        return state

    def __list_of_list_from_string_list(self, input_list):
        """Convert a single list with strings to a list of lists"""
        output_list = []
        for item in input_list:
            output_list.append(ast.literal_eval(item))

        return output_list

    def __list_of_np_arrays_from_string_list(self, input_list):
        """Convert a single list with strings to a list of lists"""
        output_list = []
        for item in input_list:
            # create a numpy array from the item
            actual = list(filter(len, item[1:-1].split(' ')))
            conn_down = np.zeros(len(actual), dtype=np.int)
            for i in range(len(conn_down)):
                conn_down[i] = actual[i]

            # add the array to the list
            output_list.append(conn_down)

        return output_list
