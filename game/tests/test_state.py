# test for game.py

import numpy as np
import random
from game import State, Config

class TestState(object):
    """Tests for the State and Config class"""
    def __init__(self, config=None):
        # create a new state
        self.state = State(Config() if config is None else config)

    def generate_graph(self):
        """Start again with a fresh graph"""
        self.state.generate_graph()

    def print_graph(self):
        """Print the graph"""
        # generate a graph
        headerline1 = "[["
        headerline2 = "[["
        for i in range (0, self.state.config.num_service):
            headerline1 += " {} ".format(i)
            headerline2 += " s "
        for i in range (0, self.state.config.num_viruses):
            headerline1 += " {} ".format(i + self.state.size_graph_col1)
            headerline2 += " v "
        for i in range (0, self.state.config.num_datadir):
            headerline1 += " {} ".format(i + self.state.size_graph_col2)
            headerline2 += " d "
        headerline2 += "]]"
        print (headerline1)
        print (headerline2)
        print (self.state.get_graph())
        print (self.state.get_weight())
        print (self.state.get_actions(False))
        print (self.state.get_actions(True))
        print ("Nodes: {}, Services: {}, Viruses: {}, Datadir: {}, Edges: {}".format(self.state.config.num_nodes, self.state.config.num_service, 
                                                                        self.state.config.num_viruses, self.state.config.num_datadir, self.state.size_graph_edges))
        print ("Cost Attack: {}, Cost Defense: {}".format(self.state.att_cost, self.state.def_cost))
        print ("Game points: {}, {} | State score: {} -> {} | Maintenance: {}".format(self.state.get_points(False), self.state.get_points(True), 
                                                            self.state.score_old, self.state.get_score(), self.state.maintenance_cost))

    def test_weights(self):
        """Test the weights"""
        print ("--------- WEIGHTS -----------")
        self.print_graph()
        print (self.state.get_weight())


    def test_edges(self):
        """Test the edges"""
        print ("--------- EDGES -----------")
        self.print_graph()

        for i in range(0, len(self.state.graph_edges)):
            line = "{}: ".format(i)
            for node in self.state.graph_edges[i]:
                line += " {} ".format(node)
            print (line)
        print ("Edges count: {}".format(self.state.size_graph_edges))


    def test_valid_actions(self):
        """Tests for valid actions graph"""
        print ("--------- VALID ACTIONS -----------")
        self.print_graph()
        for i in range(0, self.state.size_graph):
            if self.state.nn_input[i] == 1:
                if self.state.actions_att[i] != 1:
                    print ("A Check Data for Attack Action: {}, {}".format(i // self.state.size_graph_cols, i % self.state.size_graph_cols))
                if self.state.actions_def[i] != 0:
                    print ("A Invalid Defence Action: {}, {}".format(i // self.state.size_graph_cols, i % self.state.size_graph_cols))
            elif self.state.nn_input[i] == 0:
                if self.state.actions_att[i] != 0:
                    print ("B Invalid Attack Action: {}, {}".format(i // self.state.size_graph_cols, i % self.state.size_graph_cols))
                if self.state.actions_def[i] != 1 and i % self.state.size_graph_cols < self.state.size_graph_col2:
                    print ("B Invalid Defence Action: {}, {}".format(i // self.state.size_graph_cols, i % self.state.size_graph_cols))
            else:
                if self.state.actions_att[i] != 0:
                    print ("C Invalid Attack Action: {}, {}".format(i // self.state.size_graph_cols, i % self.state.size_graph_cols))
                if self.state.actions_def[i] != 0:
                    print ("C Invalid Defence Action: {}, {}".format(i // self.state.size_graph_cols, i % self.state.size_graph_cols))
        
    def test_make_move(self):
        """Test for make a move"""

        pre_att = np.zeros(self.state.size_graph + 1, dtype=np.int)
        post_att = np.zeros(self.state.size_graph + 1, dtype=np.int)
        pre_def = np.zeros(self.state.size_graph + 1, dtype=np.int)
        post_def = np.zeros(self.state.size_graph + 1, dtype=np.int)

        action_att = -1
        action_def = -1
        j = 0

        while action_att < 0 or (self.state.actions_att[action_att] == 0 and j < 100) :
            action_att = np.random.randint(0, self.state.size_graph)
            j += 1

        while action_def < 0 or (self.state.actions_def[action_def] == 0 and j < 100) :
            action_def = np.random.randint(0, self.state.size_graph)
            j += 1

        print ("ATTACK: ({}, {}) | DEFENCE: ({}, {}) | {}".format(action_att // self.state.size_graph_cols, action_att % self.state.size_graph_cols,
                                                                action_def // self.state.size_graph_cols, action_def % self.state.size_graph_cols, j))

        np.copyto(pre_att, self.state.actions_att)
        np.copyto(pre_def, self.state.actions_def)
        self.state.make_move(action_att, action_def)
        np.copyto(post_att, self.state.actions_att)
        np.copyto(post_def, self.state.actions_def)

        if np.sum(post_att - pre_att == 1) == 1 and action_att != self.state.size_graph:
            print ("Successful attack")
        elif np.any(post_att - pre_att != 0):
            print ("Invalid do nothing attack move")
        else:
            print ("Successful do nothing attack")

        if np.sum(post_def - pre_def  == 1) == 1 and action_def != self.state.size_graph:
            print ("Successful defence")
        elif np.any(post_def - pre_def != 0):
            print ("Invalid do nothing defence move")
        else:
            print ("Successful do nothing defence")
