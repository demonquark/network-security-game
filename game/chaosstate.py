# File: chaosstate.py
# Class file for the Chaos state (ChaosState)
# - contains the current state and possible actions on the state
# - child of the State class (see state.py)
# - goal functions determined by a chaos function

import numpy as np
from . import State


class ChaosState(State):
    """State object that uses chaos functions"""

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

            np.copyto(f3_matrix, self.random_matrix)

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

    def pareto_reward_matrix(self):
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

        pareto_indices_for_gui = np.zeros(num_possible_moves, dtype=bool)  # +1 for do nothing
        for i, j in enumerate(indices[is_efficient]):
            pareto_indices_for_gui[j] = True

        return pareto_fronts[is_efficient], pareto_indices_for_gui

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

        self.actions_pareto_def = np.zeros(self.size_graph + 1, dtype=bool)  # +1 for do nothing
        for i, j in enumerate(indices[is_efficient]):
            self.actions_pareto_def[j] = True
        self.actions_pareto_def[self.size_graph] = True

        return self.actions_pareto_def

    def scalarized_attack_actions(self, action_att, scalarization_approach=0):
        """Return a set of attack actions"""

        # get the indices of the valid defenses
        num_moves = self.size_graph + 1
        def_indices = np.zeros(num_moves, dtype=np.int)
        for i in range(num_moves):
            def_indices[i] = i
        def_indices = def_indices[self.actions_def]

        # get the goal vector
        goal_vector = np.multiply(int(1000 / np.sum(self.config.scalarization)), self.config.scalarization)

        # get the rewards for the attacks made on valid defenses
        worst_case_scores = np.zeros(len(def_indices) * 3, dtype=np.int).reshape(len(def_indices), 3)
        for i in range(len(def_indices)):
            worst_case_scores[i] = self.reward_matrix[(def_indices[i] * num_moves) + action_att]
        # remove the non-optimal defense moves
        self.actions_pareto_def = np.zeros(num_moves, dtype=bool)
        if scalarization_approach < 2:
            # Approach 0 (RANDOM) and 1 (Q-LEARNING) use the entire defense set
            np.copyto(self.actions_pareto_def, self.actions_def)
        elif scalarization_approach == 2 or scalarization_approach == 3:
            # Approach 2 (PARETO_Q_LEARNING) just use the front (do nothing)
            self.actions_pareto_def = self.pareto_defense_actions()

            # Approach 3 (TARGET_PARETO_Q_LEARNING) use the pareto front closest to the goal vector
            if scalarization_approach == 3:
                # calculate the magnitude and cos of the angle with the goal vector
                mag_goal = np.sqrt(np.einsum('i,i', goal_vector, goal_vector))
                score = np.zeros(len(self.config.scalarization), dtype=np.int)
                cosine = np.zeros(len(def_indices), dtype=float)

                # go through the pareto fronts and calculate the angle with the target vector
                for i in range(len(def_indices)):
                    if self.actions_pareto_def[def_indices[i]]:
                        # the cosine of the angle is the dot product divided by the distances
                        score = worst_case_scores[i]
                        cosine[i] = np.absolute(np.dot(goal_vector, score)
                                                / (mag_goal * np.sqrt(np.einsum('i,i', score, score))))
                    else:
                        cosine[i] = 0

                # find the pareto solution nearest to the goal vector
                cos_max = np.max(cosine)

                # save the pareto solution as a valid option
                for i in range(len(def_indices)):
                    self.actions_pareto_def[def_indices[i]] = (self.actions_pareto_def[def_indices[i]]
                                                               and cosine[i] == cos_max)

        elif scalarization_approach == 4:
            #  Approach 4 (SCALARIZATION) take the highest scalarized value
            scalarized_defense_scores = np.zeros(len(def_indices), np.int)
            for i in range(len(def_indices)):
                scalarized_defense_scores[i] = np.dot(self.config.scalarization, worst_case_scores[i, :])
            maximum_score = np.max(scalarized_defense_scores)
            for i in range(len(def_indices)):
                if scalarized_defense_scores[i] == maximum_score:
                    self.actions_pareto_def[def_indices[i]] = True
        elif scalarization_approach == 5:
            #  Approach 5 (STOM_SCALARIZATION) use STOM to determine the scalarized value
            scalarized_defense_scores = np.zeros(len(def_indices), np.int)
            for i in range(len(def_indices)):
                scalarized_defense_scores[i] = np.max(np.absolute(np.subtract(goal_vector, worst_case_scores[i, :])))
            minimum_score = np.min(scalarized_defense_scores)
            for i in range(len(def_indices)):
                if scalarized_defense_scores[i] == minimum_score:
                    self.actions_pareto_def[def_indices[i]] = True
        elif scalarization_approach == 6:
            #  Approach 6 (GUESS_SCALARIZATION) use GUESS to determine the scalarized value
            scalarized_defense_scores = np.zeros(len(def_indices), np.int)
            for i in range(len(def_indices)):
                scalarized_defense_scores[i] = np.max(
                    np.absolute(np.divide(np.subtract(goal_vector, worst_case_scores[i, :]), goal_vector)))
            minimum_score = np.min(scalarized_defense_scores)
            for i in range(len(def_indices)):
                if scalarized_defense_scores[i] == minimum_score:
                    self.actions_pareto_def[def_indices[i]] = True
