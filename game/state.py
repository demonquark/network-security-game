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
    sparcity = 0.1

    # define the possible graph weights
    low_value_nodes = [[1, 10], [5, 15], [45, 75]]
    high_value_nodes = [[20, 30], [45, 60], [150, 300]]
    weights = [low_value_nodes, high_value_nodes]

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
    scalarization = [6, 2, 2]

class State(object):
    """Save the values in a state"""
    def __init__(self, config, default_input=None, default_edges=None, default_graph_weights=None):
        # graph variables
        self.config = config
        self.size_graph_col1 = config.num_service
        self.size_graph_col2 = config.num_service + config.num_viruses
        self.size_graph_cols = config.num_service + config.num_viruses + config.num_datadir
        self.size_graph_rows = config.num_nodes
        self.size_graph = self.size_graph_cols * self.size_graph_rows
        self.size_graph_edges = 0
        self.graph_edges = default_edges
        self.graph_weights = default_graph_weights

        # score variables
        self.reward_sum = 0
        self.maintenance_cost = 0
        self.score_old = [0, 0, 0]
        self.score_now = [0, 0, 0]

        # neural network variables
        self.nn_input = np.zeros(self.size_graph + 2, dtype=np.int) # +2 for the game points
        self.actions_def = np.zeros(self.size_graph + 1, dtype=np.int) # +1 for do nothing
        self.actions_att = np.zeros(self.size_graph + 1, dtype=np.int) # +1 for do nothing
        self.generate_graph(default_input, default_edges, default_graph_weights)

    def generate_graph(self, default_input=None, default_edges=None, default_graph_weights=None):
        """Genereate a new graph with the provided characteristics"""
        # create edges (or use provided)
        if default_edges is None or not all(isinstance(i, list) for i in default_edges):
            # connect all the nodes in the graph
            self.graph_edges = [[]]
            for i in range(1, self.size_graph_rows):
                self.graph_edges.append([])
                connected_node = random.randint(0, i-1)
                self.graph_edges[i] = [connected_node]
                self.graph_edges[connected_node].append(i)

            # keep adding edges until we reach the sparcity
            self.size_graph_edges = self.size_graph_rows - 1
            edge_max = int(self.config.sparcity * self.size_graph_rows * (self.size_graph_rows - 1))
            while self.size_graph_edges < edge_max:
                # find two random nodes
                random_node = random.randint(0, self.size_graph_rows)
                connected_node = random.randint(0, self.size_graph_rows)

                # connect them, unless they're not already connected
                if not (random_node == connected_node
                        or connected_node in self.graph_edges[random_node]):
                    self.graph_edges[random_node].append(connected_node)
                    self.graph_edges[connected_node].append(random_node)
                    self.size_graph_edges += 1
        else:
            # set the edges to the provided list
            np.copyto(self.graph_edges, default_edges)

            # calculate the number of edges
            self.size_graph_edges = 0
            for node in default_edges:
                self.size_graph_edges += len(node)
            self.size_graph_edges = self.size_graph_edges / 2

        # create new weights for the graph
        self.graph_weights = []
        if default_graph_weights is None or not isinstance(default_graph_weights, list):
            for i in range(0, self.size_graph_rows):

                # set the possible values
                node_value = random.randint(0, 1)

                # service weights
                for j in range(0, self.size_graph_col1):
                    self.graph_weights.append(random.randint(self.config.weights[node_value][0][0],
                                                             self.config.weights[node_value][0][1]))
                # virus weights
                for j in range(self.size_graph_col1, self.size_graph_col2):
                    self.graph_weights.append(random.randint(self.config.weights[node_value][1][0],
                                                             self.config.weights[node_value][1][1]))
                # data weights
                for j in range(self.size_graph_col2, self.size_graph_cols):
                    self.graph_weights.append(random.randint(self.config.weights[node_value][2][0],
                                                             self.config.weights[node_value][2][1]))

        else:
            np.copyto(self.graph_weights, default_graph_weights)

        # create a new the state (or use provided)
        self.reset_state(default_input)

    def reset_state(self, default_input):
        """Reset the state to the one provided"""
        # reset the service statuses
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

        # reset the scores
        self.reset_scores()

        # reset the actions
        self.reset_actions()

    def reset_scores(self):
        """Calcute the state score from the graph"""
        self.reward_sum = 0
        self.maintenance_cost = 0
        self.score_old = np.zeros(3, dtype=np.int)
        self.score_now = np.zeros(3, dtype=np.int)

        # f_1: sum of weights for connected services
        for node in range(0, self.graph_edges):
            for connected_node in self.graph_edges[node]:
                if connected_node < node:
                    # loop through the services:
                    for i in range(0, self.size_graph_col1):
                        id1 = (node * self.size_graph_cols) + i
                        id2 = (connected_node * self.size_graph_cols) + i
                        if self.nn_input[id1] == 1 and self.nn_input[id2] == 1:
                            self.score_now[0] += self.graph_weights[id1]
                            self.score_now[0] += self.graph_weights[id2]

        # f_2: sum of weight for data and viruses
        for i in range(0, self.size_graph_rows):
            for j in range(self.size_graph_col2, self.size_graph_cols):
                if self.nn_input[(i * self.size_graph_cols) + j] == 1:
                    self.score_now[1] += self.graph_weights[(i * self.size_graph_cols) + j]

        # f_3: difference in game points
        self.score_now[2] = self.nn_input[-1] - self.nn_input[-2]

        for i in (0, self.size_graph_rows):
            for j in (0, self.size_graph_col2):
                self.maintenance_cost += self.nn_input[(i * self.size_graph_cols) + j] == 1

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

    def make_move(self, def_action=-1, att_action=-1):
        """Make a move"""

        # default action is do nothing
        if def_action >= self.size_graph:
            def_action = -1
        if att_action >= self.size_graph:
            att_action = -1

        # attacker action
        if att_action >= 0:
            # bring the node down
            self.nn_input[att_action] = 0
            
            # incur a cost for performing the action
            self.nn_input[-2] -= self.config.att_cost[att_action % self.size_graph_cols]

            # update the valid actions
            self.actions_att[att_action] = 0
            if att_action % self.size_graph_cols >= self.size_graph_col2:
                self.actions_def[att_action] = 1

            # update the maintenance cost
            self.maintenance_cost += att_action % self.size_graph_cols < self.size_graph_col2

        # defender action
        if def_action >= 0:
            # bring the node up
            self.nn_input[def_action] = 1

            # update the valid actions
            self.actions_def[def_action] = 0
            self.actions_att[def_action] = 1

            # incur a cost for performing the action
            self.nn_input[-2] -= self.config.def_cost[def_action % self.size_graph_cols]

        # update scores
        self.__update_score(def_action, att_action)

        return self.score_now

    def __update_score(self, def_action=-1, att_action=-1):
        """Update the score based on the supplied moves"""

        # default action is do nothing
        if def_action >= self.size_graph:
            def_action = -1
        if att_action >= self.size_graph:
            att_action = -1

        # save the old score
        np.copyto(self.score_old, self.score_now)

        # f_1: sum of weights for connected services
        if def_action != -1 and def_action % self.size_graph_cols < self.size_graph_col1:
            for connected_node in self.graph_edges[int(def_action / self.size_graph_cols)]:
                id2 = (connected_node * self.size_graph_cols) + (def_action % self.size_graph_cols)
                if self.nn_input[id2] == 1:
                    self.score_now[0] += self.graph_weights[def_action]
                    self.score_now[0] += self.graph_weights[id2]

        if att_action != -1 and att_action % self.size_graph_cols < self.size_graph_col1:
            for connected_node in self.graph_edges[int(att_action / self.size_graph_cols)]:
                id2 = (connected_node * self.size_graph_cols) + (att_action % self.size_graph_cols)
                if self.nn_input[id2] == 1 and id2 != def_action:
                    self.score_now[0] -= self.graph_weights[att_action]
                    self.score_now[0] -= self.graph_weights[id2]

        # f_2: sum of weight for data and viruses
        if def_action != -1 and def_action % self.size_graph_cols >= self.size_graph_col1:
            self.score_now[1] += self.graph_weights[def_action]
        if att_action != -1 and self.size_graph_cols >= self.size_graph_col1:
            self.score_now[1] -= self.graph_weights[att_action]

        # f_3: difference in game points
        self.score_now[2] = self.nn_input[-1] - self.nn_input[-2]

    def get_actions(self, defender=True):
        """Get a 2D representation of the graph"""
        if defender:
            return self.actions_def.reshape(self.size_graph_rows, self.size_graph_cols)
        else:
            return self.actions_att.reshape(self.size_graph_rows, self.size_graph_cols)

    def get_graph(self):
        """Get a 2D representation of the graph"""
        return self.nn_input[:-2].reshape(self.size_graph_rows, self.size_graph_cols)

    def get_weight(self):
        """Get a 2D representation of the graph"""
        return self.graph_weights.reshape(self.size_graph_rows, self.size_graph_cols)

    def get_points(self, defender=True):
        """Get a 2 element array showing current game points"""
        return self.nn_input[-1] if defender else self.nn_input[-2]

    def get_score(self):
        """Get the score of the current state"""
        return self.score_now
