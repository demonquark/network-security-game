import matplotlib
matplotlib.use('TkAgg')

import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

import sys
import tkinter as Tk

import random
import time
import numpy as np

from state import Config, State
from chaosstate import ChaosState
from reader import StateReader


#------- START MAIN CODE --------

# Configuration
config = Config()
config.num_service = 3
config.num_viruses = 1
config.num_datadir = 0
config.num_nodes = 4
config.sparcity = 0.1
config.low_value_nodes = [[1, 100], [50, 150], [150, 250]]
config.high_value_nodes = [[200, 300], [450, 650], [60, 80]]

state = ChaosState(config)
state.print_graph()
steps = 0

#------- DEFINE WINDOW --------

def destroy(e):
    sys.exit()

root = Tk.Tk()
root.wm_title("Pareto Network App")


# create plot
fig, ax = plt.subplots()
plt.figure(figsize=(3, 2), dpi=60)

# define color
a_map = plt.get_cmap('brg')
cNorm  = colors.Normalize(vmin=0, vmax=state.size_graph)
scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=a_map)

# plot the the data
for i in range(state.size_graph):
    # pareto_front = state.reward_matrix[i * state.size_graph:(i+1) * state.size_graph]
    pareto_front = state._pareto_front(state.reward_matrix[i * state.size_graph:(i+1) * state.size_graph])
    temp_x = pareto_front[:,0]
    temp_y = pareto_front[:,1]
    ax.scatter(temp_x, temp_y, color=scalarMap.to_rgba(i))

ax.grid(True)

# a tk.DrawingArea
canvas = FigureCanvasTkAgg(fig, master=root)
canvas.show()
canvas.get_tk_widget().pack(side=Tk.RIGHT, fill=Tk.BOTH, expand=0)

canvas._tkcanvas.pack(side=Tk.RIGHT, fill=Tk.BOTH, expand=0)

fig2, ax2 = plt.subplots()
plt.figure(figsize=(3, 2), dpi=60)

# define color
a_map = plt.get_cmap('brg')
cNorm  = colors.Normalize(vmin=0, vmax=state.size_graph)
scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=a_map)

# plot the the data
for i in range(state.size_graph):
    pareto_front = state.reward_matrix[i * state.size_graph:(i+1) * state.size_graph]
    # pareto_front = state._pareto_front(state.reward_matrix[i * state.size_graph:(i+1) * state.size_graph])
    temp_x = pareto_front[:,0]
    temp_y = pareto_front[:,1]
    ax2.scatter(temp_x, temp_y, color=scalarMap.to_rgba(i))

ax2.grid(True)


canvas2 = FigureCanvasTkAgg(fig2, master=root)
canvas2.show()
canvas2.get_tk_widget().pack(side=Tk.RIGHT, fill=Tk.BOTH, expand=0)

canvas2._tkcanvas.pack(side=Tk.RIGHT, fill=Tk.BOTH, expand=0)


button = Tk.Button(master=root, text='Quit', command=sys.exit)
text = Tk.Label(text="号s")
entry_id = Tk.Entry()
text.pack()
entry_id.pack()
button.pack(side=Tk.BOTTOM)

Tk.mainloop()
