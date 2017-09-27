# File: state.py
# Class file for the state
# Includes two classes
# - Config (default configuration for the state)
# - State2 (contains the current state and possible actions on the state)

import numpy as np
from state import Config, State

class ChaosState(State):

    def reset_scores(self):
        """Calcute the state score from the graph"""
        self.score_old = np.zeros(3, dtype=np.int)
        self.score_now = np.zeros(3, dtype=np.int)
        self.__reset_reward_matrix(True)

    def __reset_reward_matrix(self, recalculate=False):
        """Calculate a matrix showing the rewards for defense/attack action on the state"""

        if recalculate or not hasattr(self, 'reward_matrix'):
            # create a new variables
            f1_matrix = np.zeros(self.size_graph * self.size_graph, dtype=float)
            f2_matrix = np.zeros(self.size_graph * self.size_graph, dtype=float)
            f3_matrix = np.zeros(self.size_graph * self.size_graph, dtype=float)

            # calculate the edge count
            edge_filter = np.zeros(self.size_graph, dtype=np.int)
            for i in range(self.size_graph_rows):
                for j in range(self.size_graph_cols):
                    edge_filter[(i * self.size_graph_cols) + j] = len(self.graph_edges[i])

            # calculate the initial x values (average of the services axis)
            x_values = np.zeros(self.size_graph, dtype=float)
            np.copyto(x_values, self.graph_weights)
            x_values = np.transpose(x_values.reshape(self.size_graph_rows, self.size_graph_cols))
            x_values = (x_values - np.min(x_values, axis=0) + 1)
            x_values /= (2 + np.max(x_values, axis=0) - np.min(x_values, axis=0))
            x_values = np.transpose(x_values).reshape(self.size_graph)

            # calculate the initial y values (average of node axis)
            y_values = np.zeros(self.size_graph, dtype=float)
            np.copyto(y_values, self.graph_weights)
            y_values = y_values * edge_filter
            y_values = np.transpose(y_values.reshape(self.size_graph_rows, self.size_graph_cols))
            y_values = np.transpose(y_values)
            y_values = (y_values - np.min(y_values, axis=0) + 1)
            y_values /= (2 + np.max(y_values, axis=0) - np.min(y_values, axis=0))
            y_values = y_values.reshape(self.size_graph)

            # set the initial reward values for the defender
            def_f1 = np.zeros(self.size_graph, dtype=float)
            def_f2 = np.zeros(self.size_graph, dtype=float)
            np.copyto(def_f1, x_values)
            np.copyto(def_f2, y_values)

            # run the topological mixing formula to get the defender reward
            for j in range(np.max(edge_filter)):
                def_f2 = ((((def_f1 + def_f2) - (def_f1 + def_f2 >= 1)) * (edge_filter > j))
                          + (def_f2 * (edge_filter <= j)))
                def_f1 = ((4 * def_f1 * (1 - def_f1) * (edge_filter > j))
                          + (def_f1 * (edge_filter <= j)))

            # copy the initial values for the attacker
            att_f1 = np.zeros(self.size_graph, dtype=float)
            att_f2 = np.zeros(self.size_graph, dtype=float)
            np.copyto(att_f2, x_values)
            np.copyto(att_f1, y_values)

            # run the topological mixing formula for the attacker
            for j in range(np.max(edge_filter)):
                att_f2 = ((((att_f1 + att_f2) - (att_f1 + att_f2 >= 1)) * (edge_filter > j))
                          + (att_f2 * (edge_filter <= j)))
                att_f1 = ((4 * att_f1 * (1 - att_f1) * (edge_filter > j))
                          + (att_f1 * (edge_filter <= j)))

            # combine the attack and defense moves in a final topological mixing
            for i in range(self.size_graph):
                for j in range(self.size_graph):
                    x_k = (def_f1[i] + att_f1[j]) / 2
                    y_k = (def_f2[i] + att_f2[j]) / 2
                    f1_matrix[(i * self.size_graph) + j] = 4 * x_k * (1 - x_k)
                    f2_matrix[(i * self.size_graph) + j] = (x_k + y_k) - (0 if x_k + y_k < 1 else 1)

            self.reward_matrix = np.stack((f1_matrix, f2_matrix, f3_matrix), axis=-1)

    def __update_score(self, att_action=-1, def_action=-1):
        """Update the score based on the supplied moves"""

        # default action is do nothing
        if def_action >= self.size_graph:
            def_action = -1
        if att_action >= self.size_graph:
            att_action = -1

        # save the old score
        self.nn_input[-2] += self.maintenance_cost
        np.copyto(self.score_old, self.score_now)
        np.copyto(self.score_now, self.reward_matrix[def_action * test_state.size_graph + att_action])

    def open_pareto(self, scores):
        # Assume that all the points are in the front
        costs_len = len(scores)
        in_front = np.ones(costs_len, dtype=bool)

        # for each point in the front, check if it is dominated by another point
        for i in range(costs_len):
            if in_front[i]:
                in_front[i] = False
                # A point is in the front if none/not-any of the other scores is objectively better
                in_front[i] = not np.any(np.all(scores[i] >= scores[in_front], axis=1))

        # print ("lengths: {} {} | {}".format(len(scores), len(scores[in_front]), (scores[in_front]).tolist()))
        return scores[in_front]



