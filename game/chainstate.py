# File: chainstate.py
# Class file for the state of the first paper
# Includes two classes
# - Config (default configuration for the state)
# - State2 (contains the current state and possible actions on the state)

import numpy as np
from state import Config
import random
import time

class ChainState(object):
    """Save the values in a state"""
    def __init__(self, config):
        # graph variables
        self.config = config
        self.max_con = 150
        self.cap_values = [50, 100]
        self.max_lfr = 0.15
        self.alpha = 1
        self.beta = 0.6
        self.gamma = 1

        # sizes
        self.size_defs = 12
        self.size_atts = 12
        self.size_nodes = config.num_nodes
        self.size_edges = int((self.size_nodes * (self.size_nodes - 1)) / 2)

        # nodes and edges
        self.nodes_con = np.zeros(self.size_nodes, dtype=np.int)    # number of connections
        self.nodes_cap = np.zeros(self.size_nodes, dtype=np.int)    # capture cost
        self.nodes_acc = np.zeros(self.size_nodes, dtype=np.int)    # access boolean
        self.edges_lfr = np.zeros(self.size_edges, dtype=float)
        self.edges_sco = np.zeros(self.size_edges, dtype=np.int)
        self.edges_scf = np.zeros(self.size_edges, dtype=np.int)

        self.strat_def = np.zeros(self.size_nodes * self.size_defs, dtype=np.int)
        self.strat_att_chain = []
        self.strat_att_conn = []

        # strategy effect on edges
        self.mult_edges_scd = []
        self.mult_edges_sca = []
        for i in range(self.size_defs):
            self.mult_edges_scd.append(np.zeros(self.size_edges, dtype=np.int))
        for i in range(self.size_atts):
            self.mult_edges_sca.append(np.zeros(self.size_edges, dtype=np.int))

        # scores
        self.results = np.zeros(shape=(self.size_defs, self.size_atts, 2))

    def generate_graph(self):
        """Genereate a new graph with the provided characteristics"""
        # generate node characteristics
        for i in range(self.size_nodes):
            self.nodes_con[i] = np.random.randint(0, self.max_con)
            self.nodes_cap[i] = self.cap_values[0 if np.random.rand() < 0.8 else 1]
            self.nodes_acc[i] = 0 if np.random.rand() < 0.66 else 1

        # generate edge characteristics
        node1 = 0
        node2 = self.size_nodes - 1
        for i in range(self.size_edges):
            self.edges_lfr[i] = np.round(np.random.rand() * (self.max_lfr - 0.01), 2) + 0.01
            self.edges_sco[i] = np.minimum(self.nodes_con[node1], self.nodes_con[node2])

            # update references to the next nodes
            node2 -= 1
            if node1 == node2:
                node1 += 1
                node2 = self.size_nodes - 1

        # generate defense actions
        for i in range(len(self.strat_def)):
            self.strat_def[i] = int(np.random.normal(0, self.max_con / 3))

        # generate attack actions
        for i in range(self.size_atts):
            self.generate_attack_chains()

        # calculate the effect on the shared connections
        self.calculate_defense()


    def calculate_defense(self):
        """Perform the defense action"""

        for i in range(self.size_defs):
            # generate edge characteristics
            node1 = 0
            node2 = self.size_nodes - 1
            for j in range(self.size_edges):
                self.mult_edges_scd[i][j] = np.minimum(self.max_con, np.maximum(0,
                        np.minimum(self.nodes_con[node1] +
                                   self.strat_def[(i * self.size_nodes) + node1],
                                   self.nodes_con[node2] +
                                   self.strat_def[(i * self.size_nodes) + node2])))

                # update references to the next nodes
                node2 -= 1
                if node1 == node2:
                    node1 += 1
                    node2 = self.size_nodes - 1

            for j in range(self.size_edges):
                self.mult_edges_scd[i][j] -= self.edges_sco[j]

    def generate_attack_chains(self):
        """Generate a set of attack chains"""

        chains = []

        # generate attack actions
        for i in range(random.randint(2, 6)):
            node1 = 0
            chain = []

            # find a starting node for a chain
            for j in range(self.size_nodes * 10):
                node1 = random.randint(0, self.size_nodes - 1)
                if self.nodes_acc[node1] > 0:
                    chain.append(node1)
                    break

            # don't create a chains without a valid starting node
            if not chain:
                continue

            # add nodes to the chain
            for j in range(int(self.size_nodes * 2 / random.randint(2, 6))):
                # find a connecting node the chain
                for k in range(self.size_nodes * 10):
                    node1 = random.randint(0, self.size_nodes - 1)
                    if node1 not in chain:
                        chain.append(node1)
                        break

            # add the chain to your attack strategies
            chains.append(chain)

        # generate attack shared connections count
        conn_down = np.zeros(len(chains), dtype=np.int)
        for i in range(len(conn_down)):
            conn_down[i] = np.random.randint(0, self.max_con)

        # append the chain to the attack strategies
        if chains:
            self.strat_att_chain.append(chains)
            self.strat_att_conn.append(conn_down)

    def nodes_to_edge_index(self, node1, node2):
        """Convert the connection between the two nodes to an edge index"""

        # make sure we have a valid connection
        if node1 < 0 or node2 < 0 or node1 == node2:
            return -1

        # make sure that node1 is less than node 2
        if node1 > node2:
            temp = node2
            node2 = node1
            node1 = temp

        # determine the index
        index = (self.size_nodes - 1) - node2
        for i in range(node1):
            index += (self.size_nodes - 1) - i

        return index

    def calculate_results(self):
        """Calculate the final goal functions"""

        # reset the results
        self.results = np.zeros(shape=(self.size_defs, self.size_atts, 2))

        for defense in range(self.size_defs):
            for attack, chains in enumerate(self.strat_att_chain):

                # reset the shared connections
                np.copyto(self.edges_scf, self.edges_sco)

                # reset the bottlenecks
                bottlenecks = np.zeros(len(chains), dtype=np.int)
                np.copyto(bottlenecks, self.strat_att_conn[attack])

                # update the current shared connections by applying to the defense
                self.edges_scf += self.mult_edges_scd[defense]

                # calculate the bottlenecks
                for chain in enumerate(chains):
                    for i in range(1, len(chain[1])):
                        edge_index = self.nodes_to_edge_index(chain[1][i-1], chain[1][i])
                        if bottlenecks[chain[0]] > self.edges_scf[edge_index]:
                            bottlenecks[chain[0]] = self.edges_scf[edge_index]

                # update the current shared connections by applying to the attack
                for chain in enumerate(chains):
                    for i in range(1, len(chain[1])):
                        edge_index = self.nodes_to_edge_index(chain[1][i-1], chain[1][i])
                        self.edges_scf[edge_index] -= bottlenecks[chain[0]]
                        if self.edges_scf[edge_index] < 0:
                            self.edges_scf[edge_index] = 0

                # calculate the AFR
                self.results[defense][attack][0] = ((np.sum(self.edges_scf * self.edges_lfr)
                                                     / np.sum(self.edges_scf)) -
                                                    (np.sum(self.edges_sco * self.edges_lfr)
                                                     / np.sum(self.edges_sco)))

                # calculate defense cost
                cost_defense = self.alpha * np.sum(np.absolute(self.mult_edges_scd[defense]))

                # calculate attack cost
                cost_attack = 0
                captured_nodes = np.zeros(self.size_nodes, dtype=bool)
                for chain in enumerate(chains):
                    cost_attack += self.beta * (len(chain[1]) - 1)
                    if not captured_nodes[chain[1][0]]:
                        captured_nodes[chain[1][0]] = True
                        cost_attack += self.nodes_cap[chain[1][0]]

                # calculate the cost
                self.results[defense][attack][1] = cost_defense - cost_attack

                self.results[np.isnan(self.results)] = 0

    def _pareto_front_filter(self, scores, maximize=False):
        """return: A boolean array, indicating whether each point is part of a Pareto front"""

        # Assume that all the points are in the front
        costs_len = len(scores)
        in_front = np.ones(costs_len, dtype=bool)

        # for each point in the front, check if it is dominated by another point
        for i in range(costs_len):
            if in_front[i]:
                in_front[i] = False
                # A point is in the front if none/not-any of the other scores is objectively better
                if maximize:
                    in_front[i] = not np.any(np.all(scores[i] <= scores[in_front], axis=1))
                else:
                    in_front[i] = not np.any(np.all(scores[i] >= scores[in_front], axis=1))

        return in_front

    def _pareto_front(self, scores, maximize=False):
        """return: A boolean array, indicating whether each point is part of a Pareto front"""
        return scores[self._pareto_front_filter(scores, maximize)]

    def pareto_defense_actions(self):
        """return: A boolean array with the Pareto efficient defences"""

        # get the pareto fronts
        pareto_fronts = np.array([self._pareto_front(score, True) for score in self.results])

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

        # filter out the dominated fronts
        pareto_fronts = pareto_fronts[is_efficient]


        # print (pareto_fronts)
        # print("{}".format(is_efficient))

        return is_efficient

    def minimax(self):
        """return: A boolean array with the Pareto efficient defences"""

        # get the pareto fronts
        is_efficient = np.zeros(self.size_defs, dtype=bool)
        normalized_results = np.zeros(shape=(self.size_defs, self.size_atts))

        # calculate the normalized values for the minimax formula
        for i in range(self.size_defs):
            for j in range(self.size_atts):
                minpts = np.min(self.results[i], axis=0)
                maxpts = np.max(self.results[i], axis=0)
                if maxpts[0] == minpts[0]:
                    maxpts[0] += 1
                if maxpts[1] == minpts[1]:
                    maxpts[1] += 1
                normalized_results[i][j] = np.sqrt(
                    np.square((self.results[i][j][0] - minpts[0]) / (maxpts[0] - minpts[0]))
                    + np.square((self.results[i][j][1] - minpts[1]) / (maxpts[1] - minpts[1])))

        lowest_value = np.min(np.max(normalized_results, axis=1))
        is_efficient = np.max(normalized_results, axis=1) == lowest_value

        # print("-------")
        # print("{:0.3f} {}".format(lowest_value, is_efficient))

        return is_efficient

    def calculate_average_distance(self, is_efficient):
        """return: The distance to the """

        # calculate the minimum and maximum of the results
        minpts = np.min(np.min(self.results, axis=0), axis=0)
        maxpts = np.max(np.max(self.results, axis=0), axis=0)
        if maxpts[0] == minpts[0]:
            maxpts[0] += 1
        if maxpts[1] == minpts[1]:
            maxpts[1] += 1

        # select the defense strategies
        defense_results = self.results[is_efficient]

        # calculate the minimum distance for the filtered strategies
        norm_set = np.square((defense_results - minpts) / (maxpts - minpts))
        distances = np.average(np.sqrt(norm_set[:,:,0] + norm_set[:,:,1]), axis=1)
        lowest_value = np.min(distances)

        return lowest_value

    def print_graph(self):
        """Output the graph"""
        print ("---------")
        print ("Nodes: {}, Edges: {}".format(self.size_nodes, self.size_edges))
        print ("Attacks: {}, Defenses: {}".format(self.size_atts, self.size_defs))
        print ("Node Connections: {}".format(self.nodes_con))
        print ("Node capture cost: {}".format(self.nodes_cap))
        print ("Node access: {}".format(self.nodes_acc))
        print ("Edges sco: {}".format(self.edges_sco.tolist()))
        print ("Edges lfr: {}".format(self.edges_lfr))
        print ("Strategy def: {}".format(self.strat_def))
        print ("Strategy att: {}".format(self.strat_att_chain))
        print ("Strategy att (cost): {}".format(self.strat_att_conn))
        # print ("Results: {}".format(self.results))s
        print ("---------")

# Configuration
config = Config()
config.num_nodes = 25
np.set_printoptions(formatter={'float': '{: 0.3f}'.format})

histogram_options = np.zeros(20, dtype=np.int)
histogram_distance = np.zeros(30, dtype=np.int)
duration_paretos = 0
duration_minimax = 0
distance_paretos = 0
distance_minimax = 0
matches = 0

for i in range(1000):
    # run the experiment
    config.num_nodes = 20
    state = ChainState(config)
    state.generate_graph()
    state.calculate_results()

    # do the pareto calculation
    start_time = time.time()
    paretos_efficient = state.pareto_defense_actions()
    duration_paretos += time.time() - start_time

    # do the minimax calculation
    start_time = time.time()
    minimax_efficient = state.minimax()
    duration_minimax += time.time() - start_time

    # calculate the minimum distances
    paretos_distance = state.calculate_average_distance(paretos_efficient)
    minimax_distance = state.calculate_average_distance(minimax_efficient)
    p_index = int((paretos_distance - minimax_distance) * 10 ) + 15 
    if p_index >= 30:
        p_index = 29
        print ("error p = {}".format(p_index))
    if p_index < 0:
        p_index = 0
        print ("error p = {}".format(p_index))
        
    # add the options to the histogram
    histogram_options[np.count_nonzero(paretos_efficient) - 1] += 1
    histogram_distance[p_index] += 1


    # add the total matches
    if np.any(paretos_efficient & minimax_efficient):
        matches += 1

print ("---------")
print (matches)
print (histogram_options)
print (histogram_distance)
print (duration_paretos)
print (duration_minimax)


# print (np.any(paretos_efficient & minimax_efficient))
# print("-------")
# print("{:0.3f} {} {}".format(minimax_low, minimax_efficient, np.count_nonzero(minimax_efficient)))
# print("-------")
# print("{:0.3f} {} {}".format(paretos_low, paretos_efficient, np.count_nonzero(paretos_efficient)))
