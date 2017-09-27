import random
import time
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx


#------- START MAIN CODE --------

# # Configuration
# config = Config()
# config.num_service = 3
# config.num_viruses = 1
# config.num_datadir = 0
# config.num_nodes = 4
# config.sparcity = 0.1
# config.low_value_nodes = [[1, 100], [50, 150], [150, 250]]
# config.high_value_nodes = [[200, 300], [450, 650], [60, 80]]


# test_state = ChaosState(config)
# test_state.print_graph()

# fig, ax = plt.subplots()

# # define color
# a_map = plt.get_cmap('brg')
# cNorm  = colors.Normalize(vmin=0, vmax=test_state.size_graph)
# scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=a_map)

# for i in range(test_state.size_graph):
#     # pareto_front = test_state.reward_matrix[i * test_state.size_graph:(i+1) * test_state.size_graph]
#     pareto_front = test_state.open_pareto(test_state.reward_matrix[i * test_state.size_graph:(i+1) * test_state.size_graph])
#     temp_x = pareto_front[:,0]
#     temp_y = pareto_front[:,1]
#     ax.scatter(temp_x, temp_y, color=scalarMap.to_rgba(i))

# ax.grid(True)
# plt.show()