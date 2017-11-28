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
from chainstate import ChainState
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
        self.strvar_att_ratio2 = Tk.StringVar()
        self.strvar_def_ratio0 = Tk.StringVar()
        self.strvar_def_ratio1 = Tk.StringVar()
        self.strvar_def_ratio2 = Tk.StringVar()
        self.strvar_results0 = Tk.StringVar()
        self.strvar_results1 = Tk.StringVar()

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
        self.txt_results0 = "Possible defense moves"
        self.txt_results1 = "Non-dominated defense moves"
        self.txt_load_file = "Load Game from existing file"

        self.strvar_nodes.set("6")
        self.strvar_server_ratio0.set("1")
        self.strvar_server_ratio1.set("4")
        self.strvar_capture_cost0.set("50")
        self.strvar_capture_cost1.set("100")
        self.strvar_link_fail.set("0.15")
        self.strvar_alpha.set("1.0")
        self.strvar_beta.set("0.6")
        self.strvar_num_att_strat.set("12")
        self.strvar_num_def_strat.set("12")
        self.strvar_sparsity.set("0.1")
        self.strvar_num_services.set("3")
        self.strvar_service_server_cost0.set("1")
        self.strvar_service_server_cost1.set("10")
        self.strvar_service_client_cost0.set("20")
        self.strvar_service_client_cost1.set("30")
        self.strvar_num_virus.set("1")
        self.strvar_virus_server_cost0.set("45")
        self.strvar_virus_server_cost1.set("60")
        self.strvar_virus_client_cost0.set("5")
        self.strvar_virus_client_cost1.set("15")
        self.strvar_num_data.set("1")
        self.strvar_data_server_cost0.set("60")
        self.strvar_data_server_cost1.set("80")
        self.strvar_data_client_cost0.set("15")
        self.strvar_data_client_cost1.set("25")
        self.strvar_att_service_cost.set("6")
        self.strvar_att_virus_cost.set("8")
        self.strvar_att_data_cost.set("12")
        self.strvar_def_service_cost.set("12")
        self.strvar_def_virus_cost.set("12")
        self.strvar_att_points.set("100")
        self.strvar_def_points.set("100")
        self.strvar_att_bots.set("60")
        self.strvar_att_ratio0.set("3")
        self.strvar_att_ratio1.set("7")
        self.strvar_att_ratio2.set("0")
        self.strvar_def_ratio0.set("7")
        self.strvar_def_ratio1.set("3")
        self.strvar_def_ratio2.set("0")
        self.strvar_results0.set("-")
        self.strvar_results1.set("-")

        self.strvar_game_definition.set(self.txt_game_def_options[-1])

        # frames and widgets
        self.root = app_root
        self.root.title(self.txt_app_title)
        self.frame_container = Tk.Frame(self.root)
        self.frame_control_panel = Tk.Frame(self.frame_container, borderwidth=1, width=150, height=450, padx=10)
        self.frame_graphs = Tk.Frame(self.frame_container, borderwidth=1, width=450, height=300)
        self.frame_info = Tk.Frame(self.frame_container, borderwidth=1, width=450, height=150)

        # figures
        self.figure0 = plt.figure()
        self.figure1 = plt.figure()
        self.ax0 = None
        self.ax1 = None
        self.canvas_plot0 = FigureCanvasTkAgg(self.figure0, master=self.frame_graphs)
        self.canvas_plot1 = FigureCanvasTkAgg(self.figure1, master=self.frame_graphs)
        self.canvas_plot0.show()
        self.canvas_plot1.show()

        # results
        self.frame_results = Tk.Frame(self.frame_info, borderwidth=1, width=200, height=200)
        self.canvas_network = Tk.Frame(self.frame_info, width=200, height=200)

        # config
        self.label_game_definition = Tk.Label(self.frame_control_panel, text=self.txt_game_definition)
        self.entry_game_definition = Tk.OptionMenu(self.frame_control_panel, self.strvar_game_definition,
                                                   *self.txt_game_def_options, command=self.select_game_model)
        self.frame_config0 = Tk.Frame(self.frame_control_panel, borderwidth=1, width=150, height=450)
        self.frame_config1 = Tk.Frame(self.frame_control_panel, borderwidth=1, width=150, height=450)
        self.frame_config2 = Tk.Frame(self.frame_control_panel, borderwidth=1, width=150, height=450)
        self.button_generate = Tk.Button(self.frame_control_panel, text=self.txt_generate_graph, command=self.generate_graphs)
        self.button_quit = Tk.Button(self.frame_control_panel, text=self.txt_quit, command=sys.exit)
        self.button_file = Tk.Button(self.frame_control_panel, text=self.txt_load_file, command=self.load_file)

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
        Tk.Entry(self.frame_config0, textvariable=self.strvar_server_ratio0, width=3).grid(column=1, row=1, sticky='w')
        Tk.Label(self.frame_config0, text=self.txt_colon).grid(column=2, row=1, sticky='w')
        Tk.Entry(self.frame_config0, textvariable=self.strvar_server_ratio1, width=3).grid(column=3, row=1, sticky='w')
        Tk.Entry(self.frame_config0, textvariable=self.strvar_capture_cost0, width=3).grid(column=1, row=2, sticky='w')
        Tk.Label(self.frame_config0, text=self.txt_colon).grid(column=2, row=2, sticky='w')
        Tk.Entry(self.frame_config0, textvariable=self.strvar_capture_cost1, width=3).grid(column=3, row=2, sticky='w')
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
        Tk.Entry(self.frame_config1, textvariable=self.strvar_server_ratio0, width=3).grid(column=1, row=1, sticky='w')
        Tk.Label(self.frame_config1, text=self.txt_colon).grid(column=2, row=1, sticky='w')
        Tk.Entry(self.frame_config1, textvariable=self.strvar_server_ratio1, width=3).grid(column=3, row=1, sticky='w')
        Tk.Entry(self.frame_config1, textvariable=self.strvar_sparsity, width=3).grid(column=1, row=2, columnspan=4, sticky='w')
        Tk.Entry(self.frame_config1, textvariable=self.strvar_num_services, width=3).grid(column=1, row=4, columnspan=4, sticky='w')
        Tk.Entry(self.frame_config1, textvariable=self.strvar_service_server_cost0, width=3).grid(column=1, row=5, sticky='w')
        Tk.Label(self.frame_config1, text=self.txt_dash).grid(column=2, row=5, sticky='w')
        Tk.Entry(self.frame_config1, textvariable=self.strvar_service_server_cost1, width=3).grid(column=3, row=5, sticky='w')
        Tk.Entry(self.frame_config1, textvariable=self.strvar_service_client_cost0, width=3).grid(column=1, row=6, sticky='w')
        Tk.Label(self.frame_config1, text=self.txt_dash).grid(column=2, row=6, sticky='w')
        Tk.Entry(self.frame_config1, textvariable=self.strvar_service_client_cost1, width=3).grid(column=3, row=6, sticky='w')
        Tk.Entry(self.frame_config1, textvariable=self.strvar_num_virus, width=3).grid(column=1, row=7, columnspan=4, sticky='w')
        Tk.Entry(self.frame_config1, textvariable=self.strvar_virus_server_cost0, width=3).grid(column=1, row=8, sticky='w')
        Tk.Label(self.frame_config1, text=self.txt_dash).grid(column=2, row=8, sticky='w')
        Tk.Entry(self.frame_config1, textvariable=self.strvar_virus_server_cost1, width=3).grid(column=3, row=8, sticky='w')
        Tk.Entry(self.frame_config1, textvariable=self.strvar_virus_client_cost0, width=3).grid(column=1, row=9, sticky='w')
        Tk.Label(self.frame_config1, text=self.txt_dash).grid(column=2, row=9, sticky='w')
        Tk.Entry(self.frame_config1, textvariable=self.strvar_virus_client_cost1, width=3).grid(column=3, row=9, sticky='w')
        Tk.Entry(self.frame_config1, textvariable=self.strvar_num_data, width=3).grid(column=1, row=10, columnspan=4, sticky='w')
        Tk.Entry(self.frame_config1, textvariable=self.strvar_data_server_cost0, width=3).grid(column=1, row=11, sticky='w')
        Tk.Label(self.frame_config1, text=self.txt_dash).grid(column=2, row=11, sticky='w')
        Tk.Entry(self.frame_config1, textvariable=self.strvar_data_server_cost1, width=3).grid(column=3, row=11, sticky='w')
        Tk.Entry(self.frame_config1, textvariable=self.strvar_data_client_cost0, width=3).grid(column=1, row=12, sticky='w')
        Tk.Label(self.frame_config1, text=self.txt_dash).grid(column=2, row=12, sticky='w')
        Tk.Entry(self.frame_config1, textvariable=self.strvar_data_client_cost1, width=3).grid(column=3, row=12, sticky='w')
        Tk.Entry(self.frame_config1, textvariable=self.strvar_att_service_cost, width=3).grid(column=1, row=13, columnspan=4, sticky='w')
        Tk.Entry(self.frame_config1, textvariable=self.strvar_att_virus_cost, width=3).grid(column=1, row=14, columnspan=4, sticky='w')
        Tk.Entry(self.frame_config1, textvariable=self.strvar_att_data_cost, width=3).grid(column=1, row=15, columnspan=4, sticky='w')
        Tk.Entry(self.frame_config1, textvariable=self.strvar_def_service_cost, width=3).grid(column=1, row=16, columnspan=4, sticky='w')
        Tk.Entry(self.frame_config1, textvariable=self.strvar_def_virus_cost, width=3).grid(column=1, row=17, columnspan=4, sticky='w')
        Tk.Entry(self.frame_config1, textvariable=self.strvar_att_points, width=3).grid(column=1, row=18, columnspan=4, sticky='w')
        Tk.Entry(self.frame_config1, textvariable=self.strvar_def_points, width=3).grid(column=1, row=19, columnspan=4, sticky='w')

        # config 2 variables
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
        Tk.Entry(self.frame_config2, textvariable=self.strvar_server_ratio0, width=3).grid(column=1, row=1, sticky='w')
        Tk.Label(self.frame_config2, text=self.txt_colon).grid(column=2, row=1, sticky='w')
        Tk.Entry(self.frame_config2, textvariable=self.strvar_server_ratio1, width=3).grid(column=3, row=1, sticky='w')
        Tk.Entry(self.frame_config2, textvariable=self.strvar_sparsity, width=3).grid(column=1, row=2, columnspan=4, sticky='w')
        Tk.Entry(self.frame_config2, textvariable=self.strvar_num_services, width=3).grid(column=1, row=4, columnspan=4, sticky='w')
        Tk.Entry(self.frame_config2, textvariable=self.strvar_service_server_cost0, width=3).grid(column=1, row=5, sticky='w')
        Tk.Label(self.frame_config2, text=self.txt_dash).grid(column=2, row=5, sticky='w')
        Tk.Entry(self.frame_config2, textvariable=self.strvar_service_server_cost1, width=3).grid(column=3, row=5, sticky='w')
        Tk.Entry(self.frame_config2, textvariable=self.strvar_service_client_cost0, width=3).grid(column=1, row=6, sticky='w')
        Tk.Label(self.frame_config2, text=self.txt_dash).grid(column=2, row=6, sticky='w')
        Tk.Entry(self.frame_config2, textvariable=self.strvar_service_client_cost1, width=3).grid(column=3, row=6, sticky='w')
        Tk.Entry(self.frame_config2, textvariable=self.strvar_virus_server_cost0, width=3).grid(column=1, row=7, sticky='w')
        Tk.Label(self.frame_config2, text=self.txt_dash).grid(column=2, row=7, sticky='w')
        Tk.Entry(self.frame_config2, textvariable=self.strvar_virus_server_cost1, width=3).grid(column=3, row=7, sticky='w')
        Tk.Entry(self.frame_config2, textvariable=self.strvar_virus_client_cost0, width=3).grid(column=1, row=8, sticky='w')
        Tk.Label(self.frame_config2, text=self.txt_dash).grid(column=2, row=8, sticky='w')
        Tk.Entry(self.frame_config2, textvariable=self.strvar_virus_client_cost1, width=3).grid(column=3, row=8, sticky='w')
        Tk.Entry(self.frame_config2, textvariable=self.strvar_att_bots, width=3).grid(column=1, row=9, columnspan=4, sticky='w')
        Tk.Entry(self.frame_config2, textvariable=self.strvar_att_ratio0, width=2).grid(column=1, row=10, sticky='w')
        Tk.Label(self.frame_config2, text=self.txt_colon).grid(column=2, row=10, sticky='w')
        Tk.Entry(self.frame_config2, textvariable=self.strvar_att_ratio1, width=2).grid(column=3, row=10, sticky='w')
        Tk.Label(self.frame_config2, text=self.txt_colon).grid(column=4, row=10, sticky='w')
        Tk.Entry(self.frame_config2, textvariable=self.strvar_att_ratio2, width=2).grid(column=5, row=10, sticky='w')
        Tk.Entry(self.frame_config2, textvariable=self.strvar_def_ratio0, width=2).grid(column=1, row=11, sticky='w')
        Tk.Label(self.frame_config2, text=self.txt_colon).grid(column=2, row=11, sticky='w')
        Tk.Entry(self.frame_config2, textvariable=self.strvar_def_ratio1, width=2).grid(column=3, row=11, sticky='w')
        Tk.Label(self.frame_config2, text=self.txt_colon).grid(column=4, row=11, sticky='w')
        Tk.Entry(self.frame_config2, textvariable=self.strvar_def_ratio2, width=2).grid(column=5, row=11, sticky='w')

        # results variables
        Tk.Label(self.frame_results, text=self.txt_results0).grid(column=0, row=0, sticky='w')
        Tk.Label(self.frame_results, text=self.txt_results1).grid(column=0, row=1, sticky='w')
        self.label_results0 = Tk.Label(self.frame_results, textvariable=self.strvar_results0).grid(column=1, row=0, sticky='w')
        self.label_results1 = Tk.Label(self.frame_results, textvariable=self.strvar_results1).grid(column=1, row=1, sticky='w')

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
        self.button_file.grid(column=0, row=4, columnspan=2, sticky='nsew')

        # plot figures
        self.canvas_plot0.get_tk_widget().grid(column=0, row=0, sticky='nsew')
        self.canvas_plot1.get_tk_widget().grid(column=1, row=0, sticky='nsew')

        # results
        self.frame_results.grid(column=0, row=0, sticky='nsew')
        self.canvas_network.grid(column=1, row=0, sticky='nsew')

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
        self.frame_control_panel.rowconfigure(2, weight=8)
        self.frame_control_panel.rowconfigure(3, weight=1)
        self.frame_control_panel.rowconfigure(4, weight=1)
        self.frame_graphs.columnconfigure(0, minsize=100, weight=1)
        self.frame_graphs.rowconfigure(0, weight=1)
        self.frame_graphs.rowconfigure(1, weight=1)
        self.frame_info.columnconfigure(0, minsize=100, weight=1)
        self.frame_info.rowconfigure(0, weight=1)
        self.frame_info.rowconfigure(1, weight=1)

    def select_game_model(self, game_model):
        """Update the interface based on the chosen game model"""

        # clear the figures
        if self.ax0 is not None:
            self.ax0.clear()
        if self.ax0 is not None:
            self.ax1.clear()

        # move to the next game model
        if game_model == self.txt_game_def_options[0]:
            self.frame_config0.tkraise()
        elif game_model == self.txt_game_def_options[1]:
            self.frame_config1.tkraise()
        else:
            self.frame_config2.tkraise()

    def generate_graphs(self, loaded_state=None):
        """Graph the plot in a canvas"""

        # Configuration
        config = Config()
        config.num_nodes = self.toInt(self.strvar_nodes.get())
        config.server_client_ratio = (self.toFloat(self.strvar_server_ratio0.get())
                                      / (self.toFloat(self.strvar_server_ratio0.get())
                                         + self.toFloat(self.strvar_server_ratio1.get())))
        config.sparcity = self.toFloat(self.strvar_sparsity.get())
        config.num_service = self.toInt(self.strvar_num_services.get())
        config.num_viruses = self.toInt(self.strvar_num_virus.get())
        config.num_datadir = self.toInt(self.strvar_num_data.get())
        config.scalarization = np.array([self.toInt(self.strvar_def_ratio0),
                                         self.toInt(self.strvar_def_ratio1),
                                         self.toInt(self.strvar_def_ratio2)],
                                        dtype=np.int)
        config.scalarize_att = np.array([self.toInt(self.strvar_att_ratio0),
                                         self.toInt(self.strvar_att_ratio1),
                                         self.toInt(self.strvar_att_ratio2)],
                                        dtype=np.int)
        config.cap_values = [self.toInt(self.strvar_capture_cost0), self.toInt(self.strvar_capture_cost1)]
        config.max_lfr = self.toFloat(self.strvar_link_fail.get())
        config.alpha = self.toFloat(self.strvar_alpha.get())
        config.beta = self.toFloat(self.strvar_beta.get())
        config.size_att_strategies = self.toInt(self.strvar_num_att_strat.get())
        config.size_def_strategies = self.toInt(self.strvar_num_def_strat.get())

        config.low_value_nodes = [[self.toInt(self.strvar_service_client_cost0.get()), self.toInt(self.strvar_service_client_cost1.get())],
                                  [self.toInt(self.strvar_virus_client_cost0.get()), self.toInt(self.strvar_virus_client_cost1.get())],
                                  [self.toInt(self.strvar_data_client_cost0.get()), self.toInt(self.strvar_data_client_cost1.get())]]
        config.high_value_nodes = [[self.toInt(self.strvar_service_server_cost0.get()), self.toInt(self.strvar_service_server_cost1.get())],
                                   [self.toInt(self.strvar_virus_server_cost0.get()), self.toInt(self.strvar_virus_server_cost1.get())],
                                   [self.toInt(self.strvar_data_server_cost0.get()), self.toInt(self.strvar_data_server_cost1.get())]]
        config.att_points = self.toInt(self.strvar_att_points.get())
        config.def_points = self.toInt(self.strvar_def_points.get())
        config.att_cost_weights = np.array([self.toInt(self.strvar_att_service_cost),
                                            self.toInt(self.strvar_att_virus_cost),
                                            self.toInt(self.strvar_att_data_cost)],
                                            dtype=np.int)
        config.def_cost_weights = np.array([self.toInt(self.strvar_def_service_cost),
                                            self.toInt(self.strvar_def_virus_cost),
                                            0],
                                            dtype=np.int)

        data_points = None
        graph_is_3d = True

        # update the data based on the choice of game model
        if self.strvar_game_definition.get() == self.txt_game_def_options[0]:
            self.state = ChainState(config, True) if loaded_state is None else loaded_state
            data_points = self.state.results
            graph_is_3d = False
        elif self.strvar_game_definition.get() == self.txt_game_def_options[1]:
            self.state = State(config) if loaded_state is None else loaded_state
            data_points = self.state.reset_reward_matrix()
        else:
            config.num_viruses = self.toInt(self.strvar_num_services.get())
            config.num_datadir = 0
            self.state = ChaosState(config) if loaded_state is None else loaded_state
            self.state.reset_reward_matrix()
            data_points = []
            for i in range(self.state.size_graph):
                data_points.append(self.state.reward_matrix[i * (self.state.size_graph + 1):(i+1) * (self.state.size_graph + 1)])

        # show the graphs
        self.ax0 = self.plot_graph(self.figure0, self.ax0, data_points, graph_is_3d, False)
        self.ax1 = self.plot_graph(self.figure1, self.ax1, data_points, graph_is_3d, True)
        self.canvas_plot0.draw()
        self.canvas_plot1.draw()

    def plot_graph(self, figure, ax, data_points, graph_is_3d=True, calculate_pareto=False):
        """Graph the plot in a canvas"""

        #call the clear method on your axes
        if ax is not None:
            ax.clear()
            figure.delaxes(ax)
        
        # create the new axes
        if graph_is_3d:
            ax = figure.add_subplot(111, projection='3d')
        else:
            ax = figure.add_subplot(111)

        # calculate the data points
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
            if graph_is_3d:
                temp_z = pareto_front[:,2]
                ax.scatter(temp_x, temp_y, temp_z, color=scalarMap.to_rgba(i))
            else:
                ax.scatter(temp_x, temp_y, color=scalarMap.to_rgba(i))

        #call the draw method on your canvas
        ax.grid()

        if calculate_pareto:
            self.strvar_results1.set("{}".format(len(data_points)))
        else:
            self.strvar_results0.set("{}".format(len(data_points)))

        return ax

    def read_graph_file(self, file_name=None):
        """Read a graph from a loaded file"""

        reader = StateReader(file_name)
        state = reader.read_state()

        if isinstance(state, ChaosState):
            self.strvar_game_definition.set(self.txt_game_def_options[2])
            self.select_game_model(self.txt_game_def_options[2])
        else:
            self.strvar_game_definition.set(self.txt_game_def_options[1])
            self.select_game_model(self.txt_game_def_options[1])

        self.strvar_nodes.set("{}".format(state.config.num_nodes))
        self.strvar_server_ratio0.set("{}".format(1))
        self.strvar_server_ratio1.set("{}".format(self.toInt((1 / state.config.server_client_ratio) - 1)))
        self.strvar_capture_cost0.set("{}".format(state.config.cap_values[0]))
        self.strvar_capture_cost1.set("{}".format(state.config.cap_values[1]))
        self.strvar_link_fail.set("{}".format(state.config.max_lfr))
        self.strvar_alpha.set("{}".format(state.config.alpha))
        self.strvar_beta.set("{}".format(state.config.beta))
        self.strvar_num_att_strat.set("{}".format(state.config.size_att_strategies))
        self.strvar_num_def_strat.set("{}".format(state.config.size_def_strategies))
        self.strvar_sparsity.set("{}".format(state.config.sparcity))
        self.strvar_num_services.set("{}".format(state.config.num_service))
        self.strvar_service_server_cost0.set("{}".format(state.config.high_value_nodes[0][0]))
        self.strvar_service_server_cost1.set("{}".format(state.config.high_value_nodes[0][1]))
        self.strvar_service_client_cost0.set("{}".format(state.config.low_value_nodes[0][0]))
        self.strvar_service_client_cost1.set("{}".format(state.config.low_value_nodes[0][1]))
        self.strvar_num_virus.set("{}".format(state.config.num_viruses))
        self.strvar_virus_server_cost0.set("{}".format(state.config.high_value_nodes[1][0]))
        self.strvar_virus_server_cost1.set("{}".format(state.config.high_value_nodes[1][1]))
        self.strvar_virus_client_cost0.set("{}".format(state.config.low_value_nodes[1][0]))
        self.strvar_virus_client_cost1.set("{}".format(state.config.low_value_nodes[1][1]))
        self.strvar_num_data.set("{}".format(state.config.num_datadir))
        self.strvar_data_server_cost0.set("{}".format(state.config.high_value_nodes[2][0]))
        self.strvar_data_server_cost1.set("{}".format(state.config.high_value_nodes[2][1]))
        self.strvar_data_client_cost0.set("{}".format(state.config.low_value_nodes[2][0]))
        self.strvar_data_client_cost1.set("{}".format(state.config.low_value_nodes[2][1]))
        self.strvar_att_service_cost.set("{}".format(state.config.att_cost_weights[0]))
        self.strvar_att_virus_cost.set("{}".format(state.config.att_cost_weights[1]))
        self.strvar_att_data_cost.set("{}".format(state.config.att_cost_weights[2]))
        self.strvar_def_service_cost.set("{}".format(state.config.def_cost_weights[0]))
        self.strvar_def_virus_cost.set("{}".format(state.config.def_cost_weights[1]))
        self.strvar_att_points.set("{}".format(state.config.att_points))
        self.strvar_def_points.set("{}".format(state.config.def_points))
        self.strvar_att_bots.set("{}".format(state.config.size_bots))
        self.strvar_att_ratio0.set("{}".format(state.config.scalarize_att[0]))
        self.strvar_att_ratio1.set("{}".format(state.config.scalarize_att[1]))
        self.strvar_att_ratio2.set("{}".format(state.config.scalarize_att[2]))
        self.strvar_def_ratio0.set("{}".format(state.config.scalarization[0]))
        self.strvar_def_ratio1.set("{}".format(state.config.scalarization[1]))
        self.strvar_def_ratio2.set("{}".format(state.config.scalarization[2]))

        self.generate_graphs(state)

    def load_file(self):
        """Load a file"""
        fname = Tk.filedialog.askopenfilename(filetypes=(("Template files", "*.csv"), ("All files", "*.*") ))
        if fname:
            try:
                print("Ready to read file from: {}".format(fname))
                self.read_graph_file(fname)
            except:
                Tk.messagebox.showerror("Open Source File", "Failed to read file\n'%s'" % fname)

    def toFloat(self, value):
        try:
            float(value)
            return float(value)
        except:
            return float(0.2)

    def toInt(self, value):
        try:
            int(value)
            return int(value)
        except:
            return 1

#------- START MAIN CODE --------

root = Tk.Tk()
gameApp = ModelGUI(root)
root.mainloop()
