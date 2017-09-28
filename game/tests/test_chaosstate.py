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