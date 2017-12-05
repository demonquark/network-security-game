# File: spanconverter.py
# Class file for convertering SPAN data to game state data
# see https://snap.stanford.edu/data/email-Eu-core.html for source data
# see game/reader for destination data

import csv
import numpy as np
import networkx as nx
from game.state import Config, State
from game.reader import StateReader


class SnapConverter(object):
    """Convert the SNAP data to a """
    def __init__(self, file_name=None):
        # initialize
        self.default_file_name = 'snap_data.txt' if file_name is None else file_name

    def read_state(self, file_name=None, output_file_name=None):
        """Read a state file"""

        # basic variables
        config = Config()
        graph = nx.Graph()

        # revert to the default file
        if file_name is None:
            file_name = self.default_file_name

        # read from the file
        with open(file_name, 'r') as csvfile:
            reader = csv.reader(csvfile, delimiter=' ', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            for row in reader:
                if len(row) == 2:
                    graph.add_edge(int(row[0]), int(row[1]))
                else:
                    print("invalid edge")

        # get the largest connected subgraph
        print("0) edges and nodes: {} - {}".format(graph.number_of_nodes(), graph.number_of_edges()))
        if graph.number_of_nodes() > 0:
            graph = list(nx.connected_component_subgraphs(graph))[0]
        print("1) edges and nodes: {} - {}".format(graph.number_of_nodes(), graph.number_of_edges()))

        # calculate the radius
        radius = nx.radius(graph)
        print("radius: {} > {}".format(radius, radius // 2))

        # choose a random node
        random_node = np.random.choice(graph.nodes())
        paths = nx.single_source_shortest_path_length(graph, random_node, radius // 2)
        print("paths: {}".format(len(paths)))

        # make subgraph using the random node and the radius
        subgraph = nx.Graph(graph.subgraph(paths.keys()))
        print("2) edges and nodes: {} - {}".format(subgraph.number_of_nodes(), subgraph.number_of_edges()))

        # create a dictionary
        node_reference = {}
        for i, node in enumerate(subgraph.nodes()):
            node_reference[node] = i

        # create the default edges
        edges = []
        for i, node in enumerate(subgraph.nodes()):
            edges.append([])
            for neighbor in enumerate(graph[node]):
                if neighbor[1] in node_reference and (not i == node_reference[neighbor[1]]):
                    edges[i].append(node_reference[neighbor[1]])                    

        # calculate the number of edges
        size_edges = 0
        for node in edges:
            size_edges += len(node)
        size_edges = size_edges // 2

        print("3) edges and nodes: {} - {}".format(len(edges), size_edges))

        config.num_nodes = len(edges)

        state = State(config, None, edges)

        reader = StateReader("snap_data.csv" if output_file_name is None else output_file_name)
        reader.write_state(state)
