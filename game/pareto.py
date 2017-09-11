import numpy as np
from state import State, Config
from reader import StateReader
import time
import random

def is_pareto_efficient(costs):
    """
    :param costs: An (n_points, n_costs) array
    :return: A (n_points, ) boolean array, indicating whether each point is Pareto efficient
    """
    is_efficient = np.ones(costs.shape[0], dtype = bool)
    for i, c in enumerate(costs):
        if is_efficient[i]:
            is_efficient[is_efficient] = np.any(costs[is_efficient]>=c, axis=1)  # Remove dominated points
    return is_efficient

def is_pareto_efficient_all(costs):
    """
    :param costs: An (n_points, n_costs) array
    :return: A (n_points, ) boolean array, indicating whether each point is Pareto efficient
    """
    is_efficient = np.ones(costs.shape[0], dtype = bool)
    for i, c in enumerate(costs):
        if is_efficient[i]:
            is_efficient[is_efficient] = np.any(costs[is_efficient] <= c, axis=1)  # Remove dominated points
    return is_efficient


def is_pareto_efficient_def(costs):
    """
    :param costs: An (n_points, n_costs) array
    :return: A (n_points, ) boolean array, indicating whether each point is Pareto efficient
    """
    is_efficient = np.ones(costs.shape[0], dtype = bool)
    for i, c in enumerate(costs):
        #print("==loop %s ===" % i)
        #print(costs[is_efficient])
        #print("**")
        #print(c)
        #print(2)
        if is_efficient[i]:
            is_efficient[is_efficient] = np.any(costs[is_efficient]<=c, axis=1)  # Remove dominated points
    return is_efficient


def prep_pareto_efficient(defenceActions): #input is a numpy array of attacks for each diffence strategy

 
    #print(defenceActions)
    
    mixIndices = []
    attackActionsZero = defenceActions[0]
    d_rowZero = attackActionsZero[is_pareto_efficient(attackActionsZero)]
    #print(d_rowZero)

    for i,c in enumerate(defenceActions):
      
        attackActions = c
        paretoPoints  = is_pareto_efficient(attackActions)
        newRowFront = attackActions[paretoPoints]
        #print(newRowFront)
        
        #np.maximum(newRowFront[0], newRowFront[1], newRowFront[2]) 
        a_min = np.amin(newRowFront, axis=0)
        #a_min = np.amax(newRowFront, axis=0)
        #print("===amin= %s " % a_min)
        efficient = False

        for j,d in enumerate(d_rowZero):
            
            #st = np.vstack((d,a_min))
            #st = np.vstack((d,newRowFront))
            #ix = is_pareto_efficient_def(st)
            #print(st[ix])
            #print(ix)
            
            #if np.any(ix[1]):
            if np.any(np.subtract(a_min,d))>0:
                efficient = True
                break
            

        if efficient:
            mixIndices.append(i)

        d_rowZero = newRowFront
        
        
    return mixIndices



# read and write existing state
reader = StateReader()
state = State(Config())
reader.write_state(state)
state.print_graph()
action_att = 0

def calc_score(state_x, action_a, action_d):
    """Read the data file and reset the score"""
    state_x = reader.read_state()
    score = state_x.make_move(action_a, action_d)
    return score

def is_pareto_efficient_k0(costs):
    """
    :param costs: An (n_points, n_costs) array
    :return: A (n_points, ) boolean array, indicating whether each point is Pareto efficient
    """
    costs_len = len(costs)
    is_efficient = np.ones(costs_len, dtype=bool)

    for i in range(costs_len):
        # is_efficient[i] = False
        # print ("{} {}".format(i, np.all(costs[i] >= costs[is_efficient], axis=1)))
        # is_efficient[i] = True
        if is_efficient[i]:
            is_efficient[i] = False
            is_efficient[i] = not np.any(np.all(costs[i] >= costs[is_efficient], axis=1))  # Remove dominated points
    return is_efficient




# choose an attack action
start_time = time.time()
actions = []
for i in range(6):
    for j in range(100):
        action_att = np.random.randint(0, state.size_graph)
        if state.actions_att[action_att] == 1:
            break
    actions.append(action_att)
# print ("---choose attack %s seconds ---" % (time.time() - start_time))

start_time = time.time()
scores = np.array([([ calc_score(state, y, x) for x in range(state.size_graph + 1) if state.actions_def[x]]) for y in actions])
print ("---calculate scores %s seconds ---" % (time.time() - start_time))
print (scores)
print ("------------------")



test_value1 = np.array([90, 120, 10])
test_value2 = np.array([80, 122, 12])
test_value3 = np.array([80, 120, 10])
test_value4 = np.array([80, 122, 10])
test_value5 = np.array([80, 120, 12])
test_matrix1 = np.array([test_value1, test_value2, test_value3, test_value4, test_value5])

test_value1 = np.array([90, 100, 10])
test_value2 = np.array([81, 122, 12])
test_value3 = np.array([81, 120, 10])
test_value4 = np.array([81, 122, 10])
test_value5 = np.array([81, 120, 12])
test_matrix2 = np.array([test_value1, test_value2, test_value3, test_value4, test_value5])

test_value1 = np.array([90, 120, 10])
test_value2 = np.array([80, 72, 12])
test_value3 = np.array([65, 120, 10])
test_value4 = np.array([80, 122, 10])
test_value5 = np.array([80, 120, 6])
test_matrix3 = np.array([test_value1, test_value2, test_value3, test_value4, test_value5])

test_defenses = np.array([test_matrix1, test_matrix2, test_matrix3])

# start_time = time.time()
# result2 = is_pareto_efficient_k0(test_defenses[0])
# print ("---modified pareto %s seconds ---" % (time.time() - start_time))
# print (result2)
# print ("------------------")
# result2 = is_pareto_efficient_k0(test_defenses[1])
# print ("---modified pareto %s seconds ---" % (time.time() - start_time))
# print (result2)
# print ("------------------")
# result2 = is_pareto_efficient_k0(test_defenses[2])
# print ("---modified pareto %s seconds ---" % (time.time() - start_time))
# print (result2)
# print ("------------------")



start_time = time.time()
pareto_fronts = np.array([score[is_pareto_efficient_k0(score)] for score in scores])
# print ("---modified pareto %s seconds ---" % (time.time() - start_time))
# print (pareto_fronts)
# print ("------------------")

# start_time = time.time()
is_efficient = np.ones(pareto_fronts.shape[0], dtype = bool)
for i in range(len(pareto_fronts)):

    defense_row = pareto_fronts[i]
    max_reward = np.max(defense_row, axis=0)
    # print (max_reward)
    is_efficient[i] = False
    really_false = False
    for other_row in pareto_fronts[is_efficient]:
        # print ("{}) {} {} {}".format(i, other_row.tolist(), np.all(other_row >= max_reward, axis=1), other_row[np.all(other_row >= max_reward, axis=1)]))
        if np.any( other_row[np.all(other_row >= max_reward, axis=1)] > max_reward):
            really_false = True
            # print ("Index {} is dominated.".format(i))
            break
    
    if not really_false:
        is_efficient[i] = True
print ("---modified pareto %s seconds ---" % (time.time() - start_time))
# print (is_efficient)
# print ("------------------")



# start_time = time.time()
# result1 = is_pareto_efficient(scores1)
# print ("---original pareto %s seconds ---" % (time.time() - start_time))


# print out the valid actions
# outputstring = ""
# for i in range(state.size_graph + 1):
#     if state.actions_def[i]:
#         outputstring += "{}|".format(i)

# outputstring += "\n"
# print (outputstring)
# print ("------------------")

# start_time = time.time()
# scores1 = np.array([ calc_score(state, action_att, x) for x in range(state.size_graph + 1) if state.actions_def[x]])
# print ("---calculate scores %s seconds ---" % (time.time() - start_time))
# print (scores1)
# print ("------------------")

# start_time = time.time()
# result1 = is_pareto_efficient(scores1)
# print ("---original pareto %s seconds ---" % (time.time() - start_time))
# print (result1)
# print ("------------------")
# start_time = time.time()
# result2 = is_pareto_efficient_k0(scores1)
# print ("---modified pareto %s seconds ---" % (time.time() - start_time))
# print (result2)
# print ("------------------")


# test_value1 = np.array([90, 120, 10])
# test_value2 = np.array([80, 122, 12])
# test_value3 = np.array([80, 120, 10])
# test_value4 = np.array([80, 122, 10])
# test_value5 = np.array([80, 120, 12])
# test_matrix = np.array([test_value1, test_value2, test_value3, test_value4, test_value5])


# print (test_value1)
# print (test_value2)
# print (test_value3)
# print (test_value4)
# print (test_value5)


# print (test_matrix[np.ones(test_matrix.shape[0], dtype = bool)] >= test_matrix[0])
# print (np.any(test_matrix[np.ones(test_matrix.shape[0], dtype = bool)] >= test_matrix[0], axis=1))
# print (np.all(test_matrix[np.ones(test_matrix.shape[0], dtype = bool)] >= test_matrix[0], axis=1))

# print ("--------------------")
# result1 = is_pareto_efficient(test_matrix)
# print (result1)
# result1 = is_pareto_efficient_all(test_matrix)
# print (result1)
# print ("--------------------")


# print ("test1: {} {} {} {}".format(test_value1 >= test_value2, np.all(test_value1 >= test_value2), test_value1 < test_value2, not np.any(test_value1 < test_value2)))
# print ("test2: {} {} {} {}".format(test_value1 >= test_value3, np.all(test_value1 >= test_value3), test_value1 < test_value3, not np.any(test_value1 < test_value3)))
# print ("test3: {} {} {} {}".format(test_value1 >= test_value4, np.all(test_value1 >= test_value4), test_value1 < test_value4, not np.any(test_value1 < test_value4)))
# print ("test4: {} {} {} {}".format(test_value1 >= test_value5, np.all(test_value1 >= test_value5), test_value1 < test_value5, not np.any(test_value1 < test_value5)))



#bstart_time = time.time()
#while True:
#    testDef = []
#    for x in range(10):
#        testDef.append(np.array([[random.randint(70,100) for i in range(3)] for j in range(10)]))
#    print(testDef[0])
        

#    start_time = time.time()

#    pp = prep_pareto_efficient(testDef)

#    print(pp)
#    if len(pp) < 10:
#        print ("--big-loop- %s seconds ---" % (time.time() - bstart_time))
#        break

#    print ("--- %s seconds ---" % (time.time() - start_time))
