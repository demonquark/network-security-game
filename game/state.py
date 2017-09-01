# File: state.py
# Class file for the state
# Includes two classes
# - Config (default configuration for the state)
# - State (contains the current state and possible actions on the state)

import numpy as np

class Config(object):
    """Define the configuration of the game"""
    # define game configuration
    service1_values = [1, 5, 20]
    service2_values = [2, 10]
    att_cost = [12, 24, 8]
    def_cost = [12, 12]
    att_points = 50
    def_points = 50
    num_nodes = 500
    weights = [6, 2, 2]

class State(object):
    """Save the values in a state"""
    def __init__(self, config, default_input=None):
        self.config = config
        self.size_graph_srvs = len(config.service1_values)
        self.size_graph_cols = (len(config.service1_values) + len(config.service2_values))
        self.size_graph_rows = config.num_nodes
        self.size_graph = self.size_graph_cols * self.size_graph_rows
        self.nn_input = np.zeros(self.size_graph + 2, dtype=np.int)
        self.actions_def = np.zeros(self.size_graph, dtype=np.int)
        self.actions_att = np.zeros(self.size_graph, dtype=np.int)
        self.reset_state(default_input)

    def reset_state(self, default_input):
        """Reset the default starting state"""
        if default_input is None:
            self.nn_input = np.concatenate([np.zeros(self.size_graph/3, dtype=np.int),
                                            np.ones(self.size_graph - self.size_graph/3,
                                                    dtype=np.int),
                                            np.zeros(2, dtype=np.int)])
            np.random.shuffle(self.nn_input)
            self.nn_input[-2] = self.config.att_points
            self.nn_input[-1] = self.config.def_points
        else:
            np.copyto(self.nn_input, default_input)
        self.reset_actions()

    def reset_actions(self):
        """Get an array showing the valid actions"""
        # loop through the graph and find the defender actions
        for i in range(0, self.size_graph):
            if i % self.size_graph_cols != self.size_graph_cols - 1 and self.nn_input[i] == 0:
                # check if we can afford to bring a service1 node back up
                if (i % self.size_graph_cols < self.size_graph_srvs and
                        self.config.def_cost[0] <= self.nn_input[-1]):
                    self.actions_def[i] = 1
                # check if we can afford to bring a service2 node back up
                elif (i % self.size_graph_cols >= self.size_graph_srvs and
                      self.config.def_cost[1] <= self.nn_input[-1]):
                    self.actions_def[i] = 1
                # action cost exceeds available points
                else:
                    self.actions_def[i] = 0
            # node is already up
            else:
                self.actions_def[i] = 0

        # loop through the graph and find the nodes that up
        for i in range(0, self.size_graph):
            if self.nn_input[i] == 1:
                # check if we can afford to bring a service1 node down
                if (i % self.size_graph_cols < self.size_graph_srvs and
                        self.config.att_cost[0] <= self.nn_input[-2]):
                    self.actions_att[i] = 1
                # check if we can steal data
                elif i % self.size_graph_cols == self.size_graph_cols - 1:
                    if (i > 0 and self.nn_input[i - 1] == 0 and
                            self.config.att_cost[2] <= self.nn_input[-2]):
                        self.actions_att[i] = 1
                    else:
                        self.actions_att[i] = 0
                # check if we can afford to bring a service2 node down
                elif (i % self.size_graph_cols >= self.size_graph_srvs and
                      self.config.att_cost[1] <= self.nn_input[-2]):
                    self.actions_att[i] = 1
                # action cost exceeds available points
                else:
                    self.actions_att[i] = 0
            # node is already up
            else:
                self.actions_att[i] = 0

    def make_move(self, defender=True, action=-1):
        """Make a move"""
        if action >= 0 and action < self.size_graph:
            if defender and self.actions_def[action] == 1:
                # bring the node up
                self.nn_input[action] = 1

                # update the valid actions
                self.actions_def[action] = 0
                self.actions_att[action] = 1

                # incur a cost
                cost = self.config.def_cost[0]
                if action % self.size_graph_cols >= self.size_graph_srvs:
                    cost = self.config.def_cost[1]
                self.nn_input[-1] -= cost

            elif not defender and self.actions_att[action] == 1:
                # bring the node down
                self.nn_input[action] = 0

                #update the valid actions
                self.actions_att[action] = 0
                if action % self.size_graph_cols != self.size_graph_cols - 1:
                    self.actions_def[action] = 1
                if (action % self.size_graph_cols == self.size_graph_cols - 2 and
                        self.nn_input[action + 1] == 1):
                    self.actions_att[action + 1] = 1

                # incur a cost
                cost = self.config.att_cost[0]
                if action % self.size_graph_cols == self.size_graph_cols - 1:
                    cost = self.config.att_cost[2]
                elif action % self.size_graph_cols >= self.size_graph_srvs:
                    cost = self.config.att_cost[1]
                self.nn_input[-2] -= cost

        if not defender:
            data_layer = self.nn_input[self.size_graph_cols-1:self.size_graph:self.size_graph_cols]
            maintenance_cost = self.size_graph - self.size_graph_rows
            maintenance_cost += np.sum(data_layer) - np.sum(self.nn_input[0:self.size_graph])
            self.nn_input[-2] -= maintenance_cost

    def get_score(self):
        """Get the state score (will be used to calculate rewards)"""
        score = [0, 0, 0]
        # calculate network availability
        for i in range(0, self.size_graph_srvs):
            num_online = 0
            for j in range(0, self.size_graph_rows):
                num_online += self.nn_input[j * self.size_graph_cols + i]
            score[0] += self.config.service1_values[i] * ((num_online * (num_online - 1)) / 2)

        # calculate node security
        for i in range(self.size_graph_srvs, self.size_graph_cols):
            num_online = 0
            for j in range(0, self.size_graph_rows):
                num_online += self.nn_input[j * self.size_graph_cols + i]
            score[1] += self.config.service2_values[i - self.size_graph_srvs] * num_online

        # calculate defense strategy cost effectiveness
        score[2] = self.nn_input[-1] - self.nn_input[-2]

        return score

    def get_actions(self, defender=True):
        """Get a 2D representation of the graph"""
        if defender:
            return self.actions_def.reshape(self.size_graph_rows, self.size_graph_cols)
        else:
            return self.actions_att.reshape(self.size_graph_rows, self.size_graph_cols)

    def get_graph(self):
        """Get a 2D representation of the graph"""
        return self.nn_input[:-2].reshape(self.size_graph_rows, self.size_graph_cols)

    def get_points(self, defender=True):
        """Get a 2 element array showing current game points"""
        return self.nn_input[-1] if defender else self.nn_input[-2]
