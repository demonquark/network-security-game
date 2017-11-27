import matplotlib
matplotlib.use('TkAgg')

import networkx as nx
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import matplotlib.animation as animation

import sys
import tkinter as Tk

import numpy as np
from state import Config, State
from chaosstate import ChaosState
from reader import StateReader

class ModelGUI(object):
    """Create the GUI"""
    def __init__(self, app_root):

        # string vars 
        self.strvar_game_definition = Tk.StringVar()
        self.strvar_nodes = Tk.StringVar()
        self.strvar_server_ratio0 = Tk.StringVar()
        self.strvar_server_ratio1 = Tk.StringVar()
        self.strvar_capture_cost0 = Tk.StringVar()
        self.strvar_capture_cost1 = Tk.StringVar()
        self.strvar_link_fail = Tk.StringVar()
        self.strvar_alpha = Tk.StringVar()
        self.strvar_beta = Tk.StringVar()
        self.strvar_num_att_strat = Tk.StringVar()
        self.strvar_num_def_strat = Tk.StringVar()
        self.strvar_sparsity = Tk.StringVar()
        self.strvar_num_services = Tk.StringVar()
        self.strvar_service_server_cost0 = Tk.StringVar()
        self.strvar_service_server_cost1 = Tk.StringVar()
        self.strvar_service_client_cost0 = Tk.StringVar()
        self.strvar_service_client_cost1 = Tk.StringVar()
        self.strvar_num_virus = Tk.StringVar()
        self.strvar_virus_server_cost0 = Tk.StringVar()
        self.strvar_virus_server_cost1 = Tk.StringVar()
        self.strvar_virus_client_cost0 = Tk.StringVar()
        self.strvar_virus_client_cost1 = Tk.StringVar()
        self.strvar_num_data = Tk.StringVar()
        self.strvar_data_server_cost0 = Tk.StringVar()
        self.strvar_data_server_cost1 = Tk.StringVar()
        self.strvar_data_client_cost0 = Tk.StringVar()
        self.strvar_data_client_cost1 = Tk.StringVar()
        self.strvar_att_service_cost = Tk.StringVar()
        self.strvar_att_virus_cost = Tk.StringVar()
        self.strvar_att_data_cost = Tk.StringVar()
        self.strvar_def_service_cost = Tk.StringVar()
        self.strvar_def_virus_cost = Tk.StringVar()
        self.strvar_att_points = Tk.StringVar()
        self.strvar_def_points = Tk.StringVar()
        self.strvar_att_bots = Tk.StringVar()
        self.strvar_att_ratio0 = Tk.StringVar()
        self.strvar_att_ratio1 = Tk.StringVar()
        self.strvar_def_ratio0 = Tk.StringVar()
        self.strvar_def_ratio1 = Tk.StringVar()

        # GUI text (for translation)
        self.txt_app_title = "Pareto Network App"
        self.txt_game_definition = "Game Configuration"
        self.txt_game_def_options = ["Game Model 1", "Game Model 2", "Game Model 3"]
        self.txt_generate_graph = "Generate Graph"
        self.txt_quit = "Quit"
        self.txt_nodes = "Nodes"
        self.txt_server_ratio = "Server:Client Ratio"
        self.txt_capture_cost = "Capture Costs"
        self.txt_link_fail = "Link Fail Rate"
        self.txt_alpha = "Alpha"
        self.txt_beta = "Beta"
        self.txt_num_att_strat = "Num of Attack Strategies"
        self.txt_num_def_strat = "Num of Defense Strategies"
        self.txt_dash = "-"
        self.txt_colon = ":"
        self.txt_edge_sparsity = "Edge Sparsity"
        self.txt_node_char = "Node Characteristics"
        self.txt_num_services = "Num of Services"
        self.txt_service_server_cost = "Server Cost Range (service)"
        self.txt_service_client_cost = "Client Cost Range (service)"
        self.txt_num_virus = "Num of Virus Vulnerabilities"
        self.txt_virus_server_cost = "Server Cost Range (virus)"
        self.txt_virus_client_cost = "Client Cost Range (virus)"
        self.txt_num_data = "Num of Data Stores"
        self.txt_data_server_cost = "Server Cost Range (data)"
        self.txt_data_client_cost = "Client Cost Range (data)"
        self.txt_att_service_cost = "Attack cost (service)"
        self.txt_att_virus_cost = "Attack cost (virus)"
        self.txt_att_data_cost = "Attack cost (data)"
        self.txt_def_service_cost = "Defense cost (service)"
        self.txt_def_virus_cost = "Defense cost (virus)"
        self.txt_att_points = "Attacker Game Points"
        self.txt_def_points = "Defender Game Points"
        self.txt_bandwidth_server_cost = "Server Cost Range (bandwidth)"
        self.txt_bandwidth_client_cost = "Client Cost Range (bandwidth)"
        self.txt_cpu_server_cost = "Server Cost Range (CPU)"
        self.txt_cpu_client_cost = "Client Cost Range (CPU)"
        self.txt_att_bot_resources = "Attacker bot resources"
        self.txt_att_ratio = "Attacker scalarization ratio"
        self.txt_def_ratio = "Defender scalarization ratio"

        self.strvar_nodes.set("10")
        self.strvar_server_ratio0.set("10")
        self.strvar_server_ratio1.set("10")
        self.strvar_capture_cost0.set("10")
        self.strvar_capture_cost1.set("10")
        self.strvar_link_fail.set("10")
        self.strvar_alpha.set("10")
        self.strvar_beta.set("10")
        self.strvar_num_att_strat.set("10")
        self.strvar_num_def_strat.set("10")
        self.strvar_sparsity.set("0.1")
        self.strvar_num_services.set("10")
        self.strvar_service_server_cost0.set("10")
        self.strvar_service_server_cost1.set("10")
        self.strvar_service_client_cost0.set("10")
        self.strvar_service_client_cost1.set("10")
        self.strvar_num_virus.set("10")
        self.strvar_virus_server_cost0.set("10")
        self.strvar_virus_server_cost1.set("10")
        self.strvar_virus_client_cost0.set("10")
        self.strvar_virus_client_cost1.set("10")
        self.strvar_num_data.set("10")
        self.strvar_data_server_cost0.set("10")
        self.strvar_data_server_cost1.set("10")
        self.strvar_data_client_cost0.set("10")
        self.strvar_data_client_cost1.set("10")
        self.strvar_att_service_cost.set("10")
        self.strvar_att_virus_cost.set("10")
        self.strvar_att_data_cost.set("10")
        self.strvar_def_service_cost.set("10")
        self.strvar_def_virus_cost.set("10")
        self.strvar_att_points.set("100")
        self.strvar_def_points.set("100")
        self.strvar_att_bots.set("10")
        self.strvar_att_ratio0.set("10")
        self.strvar_att_ratio1.set("10")
        self.strvar_def_ratio0.set("10")
        self.strvar_def_ratio1.set("10")


        self.strvar_game_definition.set(self.txt_game_def_options[-1])

        # frames and widgets
        self.root = app_root
        self.root.title(self.txt_app_title)
        self.frame_container = Tk.Frame(self.root)
        self.frame_control_panel = Tk.Frame(self.frame_container, borderwidth=1, width=150, height=450, padx=10)
        self.frame_graphs = Tk.Frame(self.frame_container, borderwidth=1, width=450, height=300)
        self.frame_info = Tk.Frame(self.frame_container, borderwidth=1, width=450, height=150, bg="SeaGreen1")

        # figures
        self.figure0, self.ax0 = plt.subplots()
        self.figure1, self.ax1 = plt.subplots()
        self.canvas_plot0 = FigureCanvasTkAgg(self.figure0, master=self.frame_graphs)
        self.canvas_plot1 = FigureCanvasTkAgg(self.figure1, master=self.frame_graphs)
        self.canvas_plot0.show()
        self.canvas_plot1.show()

        # results
        self.canvas_network = Tk.Canvas(self.frame_info, width=200, height=200, bg="chartreuse")
        self.frame_results = Tk.Frame(self.frame_info, borderwidth=1, width=200, height=200, bg="yellow")

        # config
        self.label_game_definition = Tk.Label(self.frame_control_panel, text=self.txt_game_definition)
        self.entry_game_definition = Tk.OptionMenu(self.frame_control_panel, self.strvar_game_definition,
                                                   *self.txt_game_def_options, command=self.select_game_model)
        self.frame_config0 = Tk.Frame(self.frame_control_panel, borderwidth=1, width=150, height=450)
        self.frame_config1 = Tk.Frame(self.frame_control_panel, borderwidth=1, width=150, height=450)
        self.frame_config2 = Tk.Frame(self.frame_control_panel, borderwidth=1, width=150, height=450)
        self.button_generate = Tk.Button(self.frame_control_panel, text=self.txt_generate_graph, command=self.generate_graphs)
        self.button_quit = Tk.Button(self.frame_control_panel, text=self.txt_quit, command=sys.exit)

        # config 0 variables
        Tk.Label(self.frame_config0, text=self.txt_nodes).grid(column=0, row=0, sticky='w')
        Tk.Label(self.frame_config0, text=self.txt_server_ratio).grid(column=0, row=1, sticky='w')
        Tk.Label(self.frame_config0, text=self.txt_capture_cost).grid(column=0, row=2, sticky='w')
        Tk.Label(self.frame_config0, text=self.txt_link_fail).grid(column=0, row=3, sticky='w')
        Tk.Label(self.frame_config0, text=self.txt_alpha).grid(column=0, row=4, sticky='w')
        Tk.Label(self.frame_config0, text=self.txt_beta).grid(column=0, row=5, sticky='w')
        Tk.Label(self.frame_config0, text=self.txt_num_att_strat).grid(column=0, row=6, sticky='w')
        Tk.Label(self.frame_config0, text=self.txt_num_def_strat).grid(column=0, row=7, sticky='w')

        Tk.Entry(self.frame_config0, textvariable=self.strvar_nodes, width=3).grid(column=1, row=0, columnspan=4, sticky='w')
        Tk.Entry(self.frame_config0, textvariable=self.strvar_server_ratio0, width=2).grid(column=1, row=1, sticky='w')
        Tk.Label(self.frame_config0, text=self.txt_colon).grid(column=2, row=1, sticky='w')
        Tk.Entry(self.frame_config0, textvariable=self.strvar_server_ratio1, width=2).grid(column=3, row=1, sticky='w')
        Tk.Entry(self.frame_config0, textvariable=self.strvar_capture_cost0, width=2).grid(column=1, row=2, sticky='w')
        Tk.Label(self.frame_config0, text=self.txt_colon).grid(column=2, row=2, sticky='w')
        Tk.Entry(self.frame_config0, textvariable=self.strvar_capture_cost1, width=2).grid(column=3, row=2, sticky='w')
        Tk.Entry(self.frame_config0, textvariable=self.strvar_link_fail, width=3).grid(column=1, row=3, columnspan=4, sticky='w')
        Tk.Entry(self.frame_config0, textvariable=self.strvar_alpha, width=3).grid(column=1, row=4, columnspan=4, sticky='w')
        Tk.Entry(self.frame_config0, textvariable=self.strvar_beta, width=3).grid(column=1, row=5, columnspan=4, sticky='w')
        Tk.Entry(self.frame_config0, textvariable=self.strvar_num_att_strat, width=3).grid(column=1, row=6, columnspan=4, sticky='w')
        Tk.Entry(self.frame_config0, textvariable=self.strvar_num_def_strat, width=3).grid(column=1, row=7, columnspan=4, sticky='w')

        # config 1 variables
        Tk.Label(self.frame_config1, text=self.txt_nodes).grid(column=0, row=0, sticky='w')
        Tk.Label(self.frame_config1, text=self.txt_server_ratio).grid(column=0, row=1, sticky='w')
        Tk.Label(self.frame_config1, text=self.txt_edge_sparsity).grid(column=0, row=2, sticky='w')
        Tk.Label(self.frame_config1, text=self.txt_node_char).grid(column=0, row=3, columnspan=4, sticky='w')
        Tk.Label(self.frame_config1, text=self.txt_num_services).grid(column=0, row=4, sticky='w')
        Tk.Label(self.frame_config1, text=self.txt_service_server_cost).grid(column=0, row=5, sticky='w')
        Tk.Label(self.frame_config1, text=self.txt_service_client_cost).grid(column=0, row=6, sticky='w')
        Tk.Label(self.frame_config1, text=self.txt_num_virus).grid(column=0, row=7, sticky='w')
        Tk.Label(self.frame_config1, text=self.txt_virus_server_cost).grid(column=0, row=8, sticky='w')
        Tk.Label(self.frame_config1, text=self.txt_virus_client_cost).grid(column=0, row=9, sticky='w')
        Tk.Label(self.frame_config1, text=self.txt_num_data).grid(column=0, row=10, sticky='w')
        Tk.Label(self.frame_config1, text=self.txt_data_server_cost).grid(column=0, row=11, sticky='w')
        Tk.Label(self.frame_config1, text=self.txt_data_client_cost).grid(column=0, row=12, sticky='w')
        Tk.Label(self.frame_config1, text=self.txt_att_service_cost).grid(column=0, row=13, sticky='w')
        Tk.Label(self.frame_config1, text=self.txt_att_virus_cost).grid(column=0, row=14, sticky='w')
        Tk.Label(self.frame_config1, text=self.txt_att_data_cost).grid(column=0, row=15, sticky='w')
        Tk.Label(self.frame_config1, text=self.txt_def_service_cost).grid(column=0, row=16, sticky='w')
        Tk.Label(self.frame_config1, text=self.txt_def_virus_cost).grid(column=0, row=17, sticky='w')
        Tk.Label(self.frame_config1, text=self.txt_att_points).grid(column=0, row=18, sticky='w')
        Tk.Label(self.frame_config1, text=self.txt_def_points).grid(column=0, row=19, sticky='w')

        Tk.Entry(self.frame_config1, textvariable=self.strvar_nodes, width=3).grid(column=1, row=0, columnspan=2, sticky='w')
        Tk.Entry(self.frame_config1, textvariable=self.strvar_server_ratio0, width=2).grid(column=1, row=1, sticky='w')
        Tk.Label(self.frame_config1, text=self.txt_colon).grid(column=2, row=1, sticky='w')
        Tk.Entry(self.frame_config1, textvariable=self.strvar_server_ratio1, width=2).grid(column=3, row=1, sticky='w')
        Tk.Entry(self.frame_config1, textvariable=self.strvar_sparsity, width=3).grid(column=1, row=2, columnspan=4, sticky='w')
        Tk.Entry(self.frame_config1, textvariable=self.strvar_num_services, width=3).grid(column=1, row=4, columnspan=4, sticky='w')
        Tk.Entry(self.frame_config1, textvariable=self.strvar_service_server_cost0, width=2).grid(column=1, row=5, sticky='w')
        Tk.Label(self.frame_config1, text=self.txt_dash).grid(column=2, row=5, sticky='w')
        Tk.Entry(self.frame_config1, textvariable=self.strvar_service_server_cost1, width=2).grid(column=3, row=5, sticky='w')
        Tk.Entry(self.frame_config1, textvariable=self.strvar_service_client_cost0, width=2).grid(column=1, row=6, sticky='w')
        Tk.Label(self.frame_config1, text=self.txt_dash).grid(column=2, row=6, sticky='w')
        Tk.Entry(self.frame_config1, textvariable=self.strvar_service_client_cost1, width=2).grid(column=3, row=6, sticky='w')
        Tk.Entry(self.frame_config1, textvariable=self.strvar_num_virus, width=3).grid(column=1, row=7, columnspan=4, sticky='w')
        Tk.Entry(self.frame_config1, textvariable=self.strvar_virus_server_cost0, width=2).grid(column=1, row=8, sticky='w')
        Tk.Label(self.frame_config1, text=self.txt_dash).grid(column=2, row=8, sticky='w')
        Tk.Entry(self.frame_config1, textvariable=self.strvar_virus_server_cost1, width=2).grid(column=3, row=8, sticky='w')
        Tk.Entry(self.frame_config1, textvariable=self.strvar_virus_client_cost0, width=2).grid(column=1, row=9, sticky='w')
        Tk.Label(self.frame_config1, text=self.txt_dash).grid(column=2, row=9, sticky='w')
        Tk.Entry(self.frame_config1, textvariable=self.strvar_virus_client_cost1, width=2).grid(column=3, row=9, sticky='w')
        Tk.Entry(self.frame_config1, textvariable=self.strvar_num_data, width=3).grid(column=1, row=10, columnspan=4, sticky='w')
        Tk.Entry(self.frame_config1, textvariable=self.strvar_data_server_cost0, width=2).grid(column=1, row=11, sticky='w')
        Tk.Label(self.frame_config1, text=self.txt_dash).grid(column=2, row=11, sticky='w')
        Tk.Entry(self.frame_config1, textvariable=self.strvar_data_server_cost1, width=2).grid(column=3, row=11, sticky='w')
        Tk.Entry(self.frame_config1, textvariable=self.strvar_data_client_cost0, width=2).grid(column=1, row=12, sticky='w')
        Tk.Label(self.frame_config1, text=self.txt_dash).grid(column=2, row=12, sticky='w')
        Tk.Entry(self.frame_config1, textvariable=self.strvar_data_client_cost1, width=2).grid(column=3, row=12, sticky='w')
        Tk.Entry(self.frame_config1, textvariable=self.strvar_att_service_cost, width=4).grid(column=1, row=13, columnspan=4, sticky='w')
        Tk.Entry(self.frame_config1, textvariable=self.strvar_att_virus_cost, width=4).grid(column=1, row=14, columnspan=4, sticky='w')
        Tk.Entry(self.frame_config1, textvariable=self.strvar_att_data_cost, width=4).grid(column=1, row=15, columnspan=4, sticky='w')
        Tk.Entry(self.frame_config1, textvariable=self.strvar_def_service_cost, width=4).grid(column=1, row=16, columnspan=4, sticky='w')
        Tk.Entry(self.frame_config1, textvariable=self.strvar_def_virus_cost, width=4).grid(column=1, row=17, columnspan=4, sticky='w')
        Tk.Entry(self.frame_config1, textvariable=self.strvar_att_points, width=4).grid(column=1, row=18, columnspan=4, sticky='w')
        Tk.Entry(self.frame_config1, textvariable=self.strvar_def_points, width=4).grid(column=1, row=19, columnspan=4, sticky='w')

        # config 1 variables
        Tk.Label(self.frame_config2, text=self.txt_nodes).grid(column=0, row=0, sticky='w')
        Tk.Label(self.frame_config2, text=self.txt_server_ratio).grid(column=0, row=1, sticky='w')
        Tk.Label(self.frame_config2, text=self.txt_edge_sparsity).grid(column=0, row=2, sticky='w')
        Tk.Label(self.frame_config2, text=self.txt_node_char).grid(column=0, row=3, columnspan=4, sticky='w')
        Tk.Label(self.frame_config2, text=self.txt_num_services).grid(column=0, row=4, sticky='w')
        Tk.Label(self.frame_config2, text=self.txt_bandwidth_server_cost).grid(column=0, row=5, sticky='w')
        Tk.Label(self.frame_config2, text=self.txt_bandwidth_client_cost).grid(column=0, row=6, sticky='w')
        Tk.Label(self.frame_config2, text=self.txt_cpu_server_cost).grid(column=0, row=7, sticky='w')
        Tk.Label(self.frame_config2, text=self.txt_cpu_client_cost).grid(column=0, row=8, sticky='w')
        Tk.Label(self.frame_config2, text=self.txt_att_bot_resources).grid(column=0, row=9, sticky='w')
        Tk.Label(self.frame_config2, text=self.txt_att_ratio).grid(column=0, row=10, sticky='w')
        Tk.Label(self.frame_config2, text=self.txt_def_ratio).grid(column=0, row=11, sticky='w')

        Tk.Entry(self.frame_config2, textvariable=self.strvar_nodes, width=3).grid(column=1, row=0, columnspan=2, sticky='w')
        Tk.Entry(self.frame_config2, textvariable=self.strvar_server_ratio0, width=2).grid(column=1, row=1, sticky='w')
        Tk.Label(self.frame_config2, text=self.txt_colon).grid(column=2, row=1, sticky='w')
        Tk.Entry(self.frame_config2, textvariable=self.strvar_server_ratio1, width=2).grid(column=3, row=1, sticky='w')
        Tk.Entry(self.frame_config2, textvariable=self.strvar_sparsity, width=3).grid(column=1, row=2, columnspan=4, sticky='w')
        Tk.Entry(self.frame_config2, textvariable=self.strvar_num_services, width=3).grid(column=1, row=4, columnspan=4, sticky='w')
        Tk.Entry(self.frame_config2, textvariable=self.strvar_service_server_cost0, width=2).grid(column=1, row=5, sticky='w')
        Tk.Label(self.frame_config2, text=self.txt_dash).grid(column=2, row=5, sticky='w')
        Tk.Entry(self.frame_config2, textvariable=self.strvar_service_server_cost1, width=2).grid(column=3, row=5, sticky='w')
        Tk.Entry(self.frame_config2, textvariable=self.strvar_service_client_cost0, width=2).grid(column=1, row=6, sticky='w')
        Tk.Label(self.frame_config2, text=self.txt_dash).grid(column=2, row=6, sticky='w')
        Tk.Entry(self.frame_config2, textvariable=self.strvar_service_client_cost1, width=2).grid(column=3, row=6, sticky='w')
        Tk.Entry(self.frame_config2, textvariable=self.strvar_virus_server_cost0, width=2).grid(column=1, row=7, sticky='w')
        Tk.Label(self.frame_config2, text=self.txt_dash).grid(column=2, row=7, sticky='w')
        Tk.Entry(self.frame_config2, textvariable=self.strvar_virus_server_cost1, width=2).grid(column=3, row=7, sticky='w')
        Tk.Entry(self.frame_config2, textvariable=self.strvar_virus_client_cost0, width=2).grid(column=1, row=8, sticky='w')
        Tk.Label(self.frame_config2, text=self.txt_dash).grid(column=2, row=8, sticky='w')
        Tk.Entry(self.frame_config2, textvariable=self.strvar_virus_client_cost1, width=2).grid(column=3, row=8, sticky='w')
        Tk.Entry(self.frame_config2, textvariable=self.strvar_att_bots, width=3).grid(column=1, row=9, columnspan=4, sticky='w')
        Tk.Entry(self.frame_config2, textvariable=self.strvar_att_ratio0, width=2).grid(column=1, row=10, sticky='w')
        Tk.Label(self.frame_config2, text=self.txt_colon).grid(column=2, row=10, sticky='w')
        Tk.Entry(self.frame_config2, textvariable=self.strvar_att_ratio1, width=2).grid(column=3, row=10, sticky='w')
        Tk.Entry(self.frame_config2, textvariable=self.strvar_def_ratio0, width=2).grid(column=1, row=11, sticky='w')
        Tk.Label(self.frame_config2, text=self.txt_colon).grid(column=2, row=11, sticky='w')
        Tk.Entry(self.frame_config2, textvariable=self.strvar_def_ratio1, width=2).grid(column=3, row=11, sticky='w')

        # create a state
        self.state = None

        # load the grid
        self.load_grid()

    def load_grid(self):
        """Place the frames and widgets in the GUI"""
        # main frames
        self.frame_container.grid(column=0, row=0, sticky='nsew')
        self.frame_control_panel.grid(column=0, row=0, rowspan=2, sticky='nsew')
        self.frame_graphs.grid(column=1, row=0, sticky='nsew')
        self.frame_info.grid(column=1, row=1, sticky='nsew')

        # content frames
        self.label_game_definition.grid(column=0, row=0, columnspan=2, sticky='nsew')
        self.entry_game_definition.grid(column=0, row=1, columnspan=2, sticky='nsew')
        self.frame_config0.grid(column=0, row=2, columnspan=2, sticky='nsew')
        self.frame_config1.grid(column=0, row=2, columnspan=2, sticky='nsew')
        self.frame_config2.grid(column=0, row=2, columnspan=2, sticky='nsew')
        self.button_generate.grid(column=0, row=3, sticky='nsew')
        self.button_quit.grid(column=1, row=3, sticky='nsew')

        # plot figures
        self.canvas_plot0.get_tk_widget().grid(column=0, row=0, sticky='nsew')
        self.canvas_plot1.get_tk_widget().grid(column=1, row=0, sticky='nsew')

        # results
        self.canvas_network.grid(column=0, row=0, sticky='nsew')
        self.frame_results.grid(column=1, row=0, sticky='nsew')

        # configuration elements
        # self.label_nodes0.grid(column=0, row=0, sticky='w')
        # self.entry_nodes0.grid(column=1, row=0, sticky='w')

        # column and row configuration
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        self.frame_container.columnconfigure(0, weight=1)
        self.frame_container.columnconfigure(1, weight=3)
        self.frame_container.rowconfigure(0, weight=2)
        self.frame_container.rowconfigure(1, weight=1)
        self.frame_control_panel.columnconfigure(0, weight=1)
        self.frame_control_panel.columnconfigure(1, weight=1)
        self.frame_control_panel.rowconfigure(0, weight=1)
        self.frame_control_panel.rowconfigure(1, weight=1)
        self.frame_control_panel.rowconfigure(2, weight=7)
        self.frame_control_panel.rowconfigure(3, weight=1)
        self.frame_graphs.columnconfigure(0, minsize=100, weight=1)
        self.frame_graphs.rowconfigure(0, weight=1)
        self.frame_graphs.rowconfigure(1, weight=1)
        self.frame_info.columnconfigure(0, minsize=100, weight=1)
        self.frame_info.rowconfigure(0, weight=1)
        self.frame_info.rowconfigure(1, weight=1)

    def select_game_model(self, game_model):
        """Update the interface based on the chosen game model"""

        # clear the figures
        self.ax0.clear()
        self.ax1.clear()
        self.canvas_plot0.draw()
        self.canvas_plot1.draw()

        # move to the next game model
        if game_model == self.txt_game_def_options[0]:
            self.frame_config0.tkraise()
        elif game_model == self.txt_game_def_options[1]:
            self.frame_config1.tkraise()
        else:
            self.frame_config2.tkraise()

    def generate_graphs(self):
        """Graph the plot in a canvas"""

        # Configuration
        config = Config()
        config.num_service = 2
        config.num_viruses = 2
        config.num_datadir = 0
        config.num_nodes = 4
        config.sparcity = 0.1
        config.low_value_nodes = [[1, 100], [50, 150], [150, 250]]
        config.high_value_nodes = [[200, 300], [450, 650], [60, 80]]

        # move to the next game model
        if self.strvar_game_definition.get() == self.txt_game_def_options[0]:
            self.state = ChainState(config)
            self.ax0 = self.plot_graph0(self.canvas_plot0, self.figure0, self.ax0, False)
            self.ax1 = self.plot_graph0(self.canvas_plot1, self.figure1, self.ax1, True)
        elif self.strvar_game_definition.get() == self.txt_game_def_options[1]:
            self.state = State(config)
            self.ax0 = self.plot_graph1(self.canvas_plot0, self.figure0, self.ax0, False)
            self.ax1 = self.plot_graph1(self.canvas_plot1, self.figure1, self.ax1, True)
        else:
            self.state = ChaosState(config)
            self.ax0 = self.plot_graph2(self.canvas_plot0, self.figure0, self.ax0, False)
            self.ax1 = self.plot_graph2(self.canvas_plot1, self.figure1, self.ax1, True)

    def plot_graph0(self, canvas, figure, ax, calculate_pareto=False):
        """Graph the plot in a canvas"""

        #call the clear method on your axes
        ax.clear()
        ax = figure.add_subplot(111)

        # define color
        a_map = plt.get_cmap('brg')
        cNorm  = colors.Normalize(vmin=0, vmax=self.state.size_graph)
        scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=a_map)

        # plot the the data
        for i in range(self.state.size_graph):
            if calculate_pareto:
                pareto_front = self.state._pareto_front(self.state.reward_matrix[i * self.state.size_graph:(i+1) * self.state.size_graph])
            else:
                pareto_front = self.state.reward_matrix[i * self.state.size_graph:(i+1) * self.state.size_graph]
            temp_x = pareto_front[:,0]
            temp_y = pareto_front[:,1]
            ax.scatter(temp_x, temp_y, color=scalarMap.to_rgba(i))

        #call the draw method on your canvas
        ax.grid()
        canvas.draw()

        return ax

    def plot_graph1(self, canvas, figure, ax, calculate_pareto=False):
        """Graph the plot in a canvas"""
        #call the clear method on your axes
        ax.clear()
        ax = figure.add_subplot(111, projection='3d')

        # calculate the data points
        data_points = self.state.reset_reward_matrix()
        if calculate_pareto:
            data_points = self.state.pareto_reward_matrix()

        # define color
        a_map = plt.get_cmap('brg')
        cNorm  = colors.Normalize(vmin=0, vmax=len(data_points))
        scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=a_map)

        # plot the the data
        for i in range(len(data_points)):
            pareto_front = data_points[i]
            temp_x = pareto_front[:,0]
            temp_y = pareto_front[:,1]
            temp_z = pareto_front[:,2]
            ax.scatter(temp_x, temp_y, temp_z, color=scalarMap.to_rgba(i))

        #call the draw method on your canvas
        ax.grid()
        canvas.draw()

        return ax

    def plot_graph2(self, canvas, figure, ax, calculate_pareto=False):
        """Graph the plot in a canvas"""

        #call the clear method on your axes
        ax.clear()
        ax = figure.add_subplot(111)

        # define color
        a_map = plt.get_cmap('brg')
        cNorm  = colors.Normalize(vmin=0, vmax=self.state.size_graph)
        scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=a_map)

        # plot the the data
        for i in range(self.state.size_graph):
            if calculate_pareto:
                pareto_front = self.state._pareto_front(self.state.reward_matrix[i * self.state.size_graph:(i+1) * self.state.size_graph])
            else:
                pareto_front = self.state.reward_matrix[i * self.state.size_graph:(i+1) * self.state.size_graph]
            temp_x = pareto_front[:,0]
            temp_y = pareto_front[:,1]
            ax.scatter(temp_x, temp_y, color=scalarMap.to_rgba(i))

        #call the draw method on your canvas
        ax.grid()
        canvas.draw()

        return ax

#------- START MAIN CODE --------

root = Tk.Tk()
gameApp = ModelGUI(root)
root.mainloop()


#--------------------------------

    # def plot_graph(self, canvas, ax, calculate_pareto=False):
    #     """Graph the plot in a canvas"""

    #     #call the clear method on your axes
    #     ax.clear()

    #     # define some random data that emulates your indeded code:
    #     number_of_points = 30
    #     x_values = np.random.random_integers(1, 100, number_of_points)
    #     y_values = np.random.random_integers(1, 100, number_of_points)
    #     ax.scatter(x_values, y_values)
    #     ax.grid()

    #     #call the draw method on your canvas
    #     canvas.draw()
