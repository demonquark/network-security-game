"""
# File: Main code
# Runs the game as requested
"""
# imports
import sys
import numpy as np
import tkinter as Tk

from .state import Config, State
from .chaosstate import ChaosState
from .chainstate import ChainState
from .reader import StateReader
from .gui import ModelGUI
from .game import GameObject


def run_game(run_type=0):
    """Generates random networks and outputs the results of running through several algorithms"""

    if run_type == 1:
        print("This model cannot be run iteratively. Please use the GUI to see the model in action.")
        return

    # Configuration
    config = Config()
    config.num_service = 3
    config.num_viruses = 3
    config.num_datadir = 0
    config.num_nodes = 3
    config.offset = np.zeros(3, dtype=np.int)
    config.scalarization = np.array([3, 7, 2], dtype=np.int)

    node_options = [50, 100, 250, 500]
    points_options = [60, 60, 60, 60]
    sparse_options = [0.001, 0.005, 0.01, 0.05]
    epochs_options = [200, 200, 200, 200]

    node_options = [50]
    sparse_options = [0.1]
    points_options = [60]
    epochs_options = [200]

    reader_files = []

    # game object creation
    game_object = GameObject()

    # create the states
    for node_count in enumerate(node_options):
        config.num_nodes = node_count[1]
        config.att_points = points_options[node_count[0]]
        config.def_points = points_options[node_count[0]]
        for sparsity in enumerate(sparse_options):
            config.sparcity = sparsity[1]
            in_reader = StateReader("state_{}_{}.csv".format(node_count[0], sparsity[0]))
            if run_type == 1:
                in_reader.write_state(ChainState(config))
            elif run_type == 2:
                in_reader.write_state(State(config))
            elif run_type == 3:
                in_reader.write_state(ChaosState(config))
            else:
                in_reader.write_state(State(config))

            reader_files.append(in_reader.default_file_name)

    # run the game with the states
    for node_count in enumerate(node_options):
        for sparsity in enumerate(sparse_options):
            # read the state and generate the starting pareto front
            in_reader = StateReader(reader_files[(node_count[0] * len(sparse_options)) + sparsity[0]])
            for k in range(7):
                if not (k == 2 or k == 1):
                    game_object.run_epochs(in_reader, epochs_options[node_count[0]], k)


def run_gui():
    """Opens the model GUI"""

    root = Tk.Tk()
    ModelGUI(root)
    root.mainloop()


def main():
    """ Run the game module"""

    # get the command line arguments
    valid_arguments = ["run", "gui"]
    arg_id = -1

    if len(sys.argv) > 1 and sys.argv[1] in valid_arguments:
        arg_id = valid_arguments.index(sys.argv[1])

    if arg_id == 0:
        # get the type of state
        state_type = 0
        if len(sys.argv) > 2:
            try:
                state_type = int(sys.argv[2])
            except (TypeError, ValueError):
                state_type = 0

        # run the game for the requested state
        run_game(state_type)

    elif arg_id == 1:
        # run the gui
        run_gui()
    else:
        print("Invalid command line argument. Please add one of the following arguments: {}".format(valid_arguments))


# ------- START MAIN CODE --------
np.set_printoptions(formatter={'float': '{: 0.3f}'.format})
main()
