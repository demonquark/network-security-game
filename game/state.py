# File: state.py
# Class file for the state
# Includes two classes
# - Config (default configuration for the state)
# - State (contains the current state and possible actions on the state)

import random
import numpy as np

class Config(object):
    """Define the configuration of the game"""
    # define graph
    num_service = 3
    num_viruses = 1
    num_datadir = 1
    num_nodes = 5
    sparcity = 0.01

    # define the attack and defence costs
    att_points = 50
    def_points = 50
    att_cost = np.concatenate([np.ones(num_service, dtype=np.int) * 5,
                               np.ones(num_viruses, dtype=np.int) * 10,
                               np.ones(num_datadir, dtype=np.int) * 20])
    def_cost = np.concatenate([np.ones(num_service, dtype=np.int) * 5,
                               np.ones(num_viruses, dtype=np.int) * 10,
                               np.ones(num_datadir, dtype=np.int) * 0])

    # define scalarization weights
    weights = [6, 2, 2]

class State(object):
    """Save the values in a state"""
    def __init__(self, config, default_input=None, default_edges=None, default_graph_weights=None):
        self.config = config
        self.size_graph_col1 = config.num_service
        self.size_graph_col2 = config.num_service + config.num_viruses
        self.size_graph_cols = config.num_service + config.num_viruses + config.num_datadir
        self.size_graph_rows = config.num_nodes
        self.size_graph = self.size_graph_cols * self.size_graph_rows
        self.graph_edges = default_edges
        self.graph_weights = default_graph_weights
        self.nn_input = np.zeros(self.size_graph + 2, dtype=np.int) # +2 for the game points
        self.actions_def = np.zeros(self.size_graph + 1, dtype=np.int) # +1 for do nothing
        self.actions_att = np.zeros(self.size_graph + 1, dtype=np.int) # +1 for do nothing
        self.generate_graph(default_input, default_edges, default_graph_weights)

    def generate_graph(self, default_input=None, default_edges=None, default_graph_weights=None):
        """Genereate a new graph with the provided characteristics"""
        # create edges (or use provided)
        if default_edges is not None and all(isinstance(i, list) for i in default_edges):
            self.graph_edges = default_edges
            self.size_graph_edges = 0
            for i in range(0, self.size_graph_rows):
                for j in range(0, len(default_edges[i]):
                    self.size_graph_edges += 1 if default_edges[i][j] < i else 0
        else:
            # first make sure the graph is fully connected
            self.graph_edges[0] = []
            for i in range(1, self.size_graph_rows):
                connected_node = random.randint(0, i)
                self.graph_edges[i] = [connected_node]
                self.graph_edges[connected_node].append(i)
            # next keep adding edges until we reach the sparcity
            self.size_graph_edges = self.size_graph_rows - 1
            edge_max = int(self.config.sparcity * self.size_graph_rows * (self.size_graph_rows - 1)) 
            while self.size_graph_edges < edge_max:
                random_node = random.randint(0, self.size_graph_rows)
                connected_node = random.randint(0, self.size_graph_rows)
                if random_node != connected_node and self.graph_edges[random_node].con:
                    self.graph_edges[i] = [connected_node]
                    self.graph_edges[connected_node].append(i)
                

        # create a new the state (or use provided)
        self.reset_state(default_input)

    def reset_state(self, default_input):
        """Reset the state to the one provided"""
        if default_input is None:
            self.nn_input = np.concatenate([np.zeros(self.size_graph/3, dtype=np.int),
                                            np.ones(self.size_graph/3, dtype=np.int) * -1,
                                            np.ones(self.size_graph - 2 * (self.size_graph/3),
                                                    dtype=np.int),
                                            np.zeros(2, dtype=np.int)])
            np.random.shuffle(self.nn_input)
            self.nn_input[-2] = self.config.att_points
            self.nn_input[-1] = self.config.def_points
        else:
            np.copyto(self.nn_input, default_input)

    def reset_actions(self):
        """Get an array showing the valid actions"""
        # loop through the graph and find the defender actions
        for i in range(0, self.size_graph):
            if i % self.size_graph_cols < self.size_graph_col2 and self.nn_input[i] == 0:
                # check if we can afford to bring a service node back up or uninstall a virus
                if self.config.def_cost[(i % self.size_graph_cols)] <= self.nn_input[-1]:
                    self.actions_def[i] = 1
                # action cost exceeds available points
                else:
                    self.actions_def[i] = 0
            # node is already up (or is not a valid service / virus / datadir)
            else:
                self.actions_def[i] = 0

        # loop through the graph and find the nodes that up
        for i in range(0, self.size_graph):
            if self.nn_input[i] == 1:
                # check if this a service or virus
                if i % self.size_graph_cols < self.size_graph_col2:
                    # check if we can afford to bring a service node down or install a virus
                    if self.config.def_cost[(i % self.size_graph_cols)] <= self.nn_input[-1]:
                        self.actions_att[i] = 1
                    else:
                        # action cost exceeds available points
                        self.actions_att[i] = 0
                # it's data, so check if we can steal data
                elif self.nn_input[i - (i%self.size_graph_cols) + self.size_graph_col1
                                   :i -(i%self.size_graph_cols) + self.size_graph_col2] == 0:
                    # check if we can afford to steal data
                    if self.config.def_cost[(i % self.size_graph_cols)] <= self.nn_input[-2]:
                        self.actions_att[i] = 1
                    # action cost exceeds available points
                    else:
                        self.actions_att[i] = 0
                else:
                    self.actions_att[i] = 0
            # node is already down
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
        for i in range(0, self.size_graph_col1):
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
