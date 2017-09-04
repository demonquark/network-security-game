
from test_state import TestState

# tests
test_case = TestState()

test_case.config.num_nodes = 50
test_case.config.sparcity = 0.001

test_case.config.low_value_nodes = [[1, 1], [3, 3], [9, 9]]
test_case.config.high_value_nodes = [[2, 2], [4, 4], [8, 8]]
test_case.config.weights = [test_case.config.low_value_nodes, test_case.config.high_value_nodes]

test_case.generate_graph()
test_case.test_weights()
test_case.test_edges()
# test.test_valid_actions()
