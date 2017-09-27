# File: state.py
# Class file for the state
# Includes two classes
# - Config (default configuration for the state)
# - State (contains the current state and possible actions on the state)

import csv, ast
import numpy as np
from state import Config, State
from chaosstate import ChaosState

class StateReader(object):
    """Read config and state files"""
    def __init__(self, file_name=None):
        # initialize
        self.default_file_name = 'state.csv' if file_name is None else file_name

    def write_state(self, state, file_name=None):
        """Read a state file"""

        # revert to the default file
        if file_name is None:
            file_name = self.default_file_name

        # write to the defined config file
        with open(file_name, 'w') as csvfile:
            writer = csv.writer(csvfile, delimiter=';', quotechar='"', quoting=csv.QUOTE_MINIMAL)
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

    def read_state(self, file_name=None):
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

        if isinstance(content, list) and len(content) == 17:
            config.num_service = int(content[0][0])
            config.num_viruses = int(content[1][0])
            config.num_datadir = int(content[2][0])
            config.num_nodes = int(content[3][0])

            config.sparcity = float(content[4][0])
            config.server_client_ratio = float(content[5][0])
            config.ratios = np.array(content[6], dtype=np.int)

            config.low_value_nodes = self.__list_of_list_from_string_list(content[7])
            config.high_value_nodes = self.__list_of_list_from_string_list(content[8])

            config.att_points = int(content[9][0])
            config.def_points = int(content[10][0])
            config.att_cost_weights = np.array(content[11], dtype=np.int)
            config.def_cost_weights = np.array(content[12], dtype=np.int)

            config.scalarization = np.array(content[13], dtype=np.int)

            default_input = np.array(content[14], dtype=np.int)
            default_edges = self.__list_of_list_from_string_list(content[15])
            default_graph_weights = np.array(content[16], dtype=np.int)

            state = State(config, default_input, default_edges, default_graph_weights)

        else:
            print ("Could not read config file.")

        return state

    def __list_of_list_from_string_list(self, input_list):
        """Convert a single list with strings to a list of lists"""
        output_list = []
        for item in input_list:
            output_list.append(ast.literal_eval(item))

        return output_list
