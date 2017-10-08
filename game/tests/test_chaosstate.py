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

# state = ChaosState(config)
# state.print_graph()
# steps = 0

# while state.get_points(True) > 0 and state.get_points(False) > 0 and steps < 10:
#     # choose an defense action
#     for j in range(100):
#         action_def = np.random.randint(0, state.size_graph)
#         if state.actions_def[action_def] == 1:
#             break

#     # choose an attack action
#     for j in range(100):
#         action_att = np.random.randint(0, state.size_graph)
#         if state.actions_att[action_att] == 1:
#             break

#     # Take actions, observe new state
#     state.make_move(action_att, action_def)

#     # next step
#     steps += 1
#     print ("{}) {} | {} - {}".format(steps, state.get_points(True), state.get_points(False), state.score_now))



# fig, ax = plt.subplots()

# # define color
# a_map = plt.get_cmap('brg')
# cNorm  = colors.Normalize(vmin=0, vmax=test_state.size_graph)
# scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=a_map)

# for i in range(test_state.size_graph):
#     # pareto_front = test_state.reward_matrix[i * test_state.size_graph:(i+1) * test_state.size_graph]
#     pareto_front = test_state._pareto_front(test_state.reward_matrix[i * test_state.size_graph:(i+1) * test_state.size_graph])
#     temp_x = pareto_front[:,0]
#     temp_y = pareto_front[:,1]
#     ax.scatter(temp_x, temp_y, color=scalarMap.to_rgba(i))

# ax.grid(True)
# plt.show()


# #------- START MAIN CODE --------

# # Configuration
# config = Config()
# config.num_service = 3
# config.num_viruses = 3
# config.num_datadir = 0
# config.num_nodes = 3
# config.offset = np.zeros(3, dtype=np.int)
# config.scalarization = np.array([3, 7, 0], dtype=np.int)

# node_options = [50, 100, 250, 500]
# points_options = [60, 60, 60, 60]
# sparse_options = [0.001, 0.005, 0.01, 0.05]
# epochs_options = [200, 200, 200, 200]

# node_options = [50]
# sparse_options = [0.1]
# points_options = [150]
# epochs_options = [200]

# reader_files = []

# config.num_nodes = node_options[0]
# config.att_points = points_options[0]
# config.def_points = points_options[0]
# config.sparcity = sparse_options[0]

# run_state = ChaosState(config)
# log_text = ""
# for k in range(1, 7):
#     log_text += "--- {} ---\n".format(k)
#     for j in range (50):
#         duration = 0
#         for i in range (10):
#             duration += run_state.scalarized_attack_actions(np.array([7, 3, 0], dtype=np.int), k)
#         log_text += "{0}\n".format(duration)

# print (log_text)