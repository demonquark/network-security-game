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
        self.reset_reward_matrix()

    def reset_reward_matrix(self):
        """Calculate a matrix showing the rewards for defense/attack action on the state"""

        if self.reward_matrix is None:
            # create a new variables
            num_possible_moves = self.size_graph + 1
            f1_matrix = np.zeros(num_possible_moves * num_possible_moves, dtype=float)
            f2_matrix = np.zeros(num_possible_moves * num_possible_moves, dtype=float)
            f3_matrix = np.zeros(num_possible_moves * num_possible_moves, dtype=float)

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

            # add the do nothing action
            att_f1 = np.append(att_f1, 0)
            att_f2 = np.append(att_f2, 0)
            def_f1 = np.append(def_f1, 0)
            def_f2 = np.append(def_f2, 0)

            # combine the attack and defense moves in a final topological mixing
            for i in range(num_possible_moves):
                for j in range(num_possible_moves):
                    x_k = (def_f1[i] + att_f1[j]) / 2
                    y_k = (def_f2[i] + att_f2[j]) / 2
                    f1_matrix[(i * num_possible_moves) + j] = 4 * x_k * (1 - x_k)
                    f2_matrix[(i * num_possible_moves) + j] = (x_k + y_k) - (x_k + y_k >= 1)

            self.reward_matrix = np.stack(((f1_matrix * 1000).astype(int),
                                           (f2_matrix * 1000).astype(int),
                                           (f3_matrix * 1000).astype(int)), axis=-1)

    def _update_score(self, att_action=-1, def_action=-1):
        """Update the score based on the supplied moves"""

        # default action is do nothing
        if def_action > self.size_graph or def_action < 0:
            def_action = self.size_graph
        if att_action > self.size_graph or att_action < 0:
            att_action = self.size_graph

        # save the old score
        self.nn_input[-2] += self.maintenance_cost

        # update the score
        self.score_old = np.zeros(3, dtype=np.int)
        self.score_now = np.zeros(3, dtype=np.int)
        np.copyto(self.score_now,
                  self.reward_matrix[(def_action * (self.size_graph + 1)) + att_action])

    def pareto_defense_actions(self):
        """return: A boolean array with the Pareto efficient defences"""

        num_possible_moves = self.size_graph + 1

        # get the indices of the valid defenses
        indices = np.zeros(num_possible_moves, np.int)
        for i in range(num_possible_moves):
            indices[i] = i
        indices = indices[self.actions_def]

        # get rewards per defense move
        truncated_scores = []
        for i in indices:
            reward_set = self.reward_matrix[i * num_possible_moves:(i+1) * num_possible_moves]
            truncated_scores.append(np.unique(reward_set[self.actions_att], axis=0))

        # get the pareto fronts
        pareto_fronts = np.array([self._pareto_front(score) for score in truncated_scores])

        # assume that all the fronts are efficient
        is_efficient = np.ones(pareto_fronts.shape[0], dtype=bool)
        size_fronts = len(pareto_fronts)
        for i in range(size_fronts):

            defense_row = pareto_fronts[i]
            max_reward = np.average(defense_row, axis=0)
            is_efficient[i] = False
            really_false = False
            for other_row in pareto_fronts[is_efficient]:
                if np.any(other_row[np.all(other_row >= max_reward, axis=1)] > max_reward):
                    really_false = True
                    break

            if not really_false:
                is_efficient[i] = True

        self.actions_pareto_def = np.zeros(self.size_graph + 1, dtype=bool) # +1 for do nothing
        for i, j in enumerate(indices[is_efficient]):
            self.actions_pareto_def[j] = True
        self.actions_pareto_def[self.size_graph] = True

        return self.actions_pareto_def
