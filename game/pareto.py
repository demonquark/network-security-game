import numpy as np
from state import State, Config
import time

config = Config()
state = State(config)

def simple_cull(inputPoints, dominates):
    paretoPoints = []#set()
    actionIndices = []#set()
    candidateRowNr = 0
    #dominatedPoints = set()
    
    index = 0
    
    while True:
        candidateRow = inputPoints[candidateRowNr]
        inputPoints.remove(candidateRow)
        rowNr = 0
        nonDominated = True
        
        
        while len(inputPoints) != 0 and rowNr < len(inputPoints):
            row = inputPoints[rowNr]
            
            #print("row======")
            #print(row)
            #print("candidateRow")
            #print(candidateRow)
            if dominates(candidateRow, row):
                # If it is worse on all features remove the row from the array
                inputPoints.remove(row)
                #dominatedPoints.add(tuple(row))
            elif dominates(row, candidateRow):
                nonDominated = False
                #dominatedPoints.add(tuple(candidateRow))
                rowNr += 1
            else:
                rowNr += 1
                

        if nonDominated:
            # add the non-dominated point to the Pareto frontier
            actionIndices.append(index)
            paretoPoints.append(candidateRow)
            
        


        if len(inputPoints) == 0:
            break
        
    return paretoPoints, actionIndices

def dominates(row, candidateRow):
    return sum([row[x] >= candidateRow[x] for x in range(len(row))]) == len(row)  

def dominates_two(def_row, def_candidateRow):
    dominated = False
    candidateColNr = 0
    
    while len(def_row) != 0 and candidateColNr < len(def_row):
        #print("def_row")
        #print(def_row[candidateColNr])
        #def_row.remove(def_row[candidateColNr])
        #k = def_row[candidateColNr]
        #x = 0
        dominatedCount =0
        for i in range(0,len(def_candidateRow)):
            #print(i)
        
        #while len(def_candidateRow) != 0 and x < len(def_candidateRow):
            #print("de_candiisbelow")
            #print(def_candidateRow[i])
            if(dominates(def_row[candidateColNr],def_candidateRow[i])):
                dominatedCount += 1
            #x += 1

        #print(dominatedCount)
        #print(len(def_candidateRow))
        if dominatedCount == len(def_candidateRow):
            dominated = True
            break;
        #def_row.remove(k)
        candidateColNr += 1
        
    return dominated


def is_pareto_efficient(costs):
    """
    :param costs: An (n_points, n_costs) array
    :return: A (n_points, ) boolean array, indicating whether each point is Pareto efficient
    """
    is_efficient = np.ones(costs.shape[0], dtype = bool)
    for i, c in enumerate(costs):
        #print(2)
        if is_efficient[i]:
            is_efficient[is_efficient] = np.any(costs[is_efficient]>=c, axis=1)  # Remove dominated points
    return is_efficient


def all_equal(elements):
    """
    :param elements: A collection of things
    :return: True if all things are equal, otherwise False.  (Note that an empty list of elements returns true, just as all([]) is True
    """
    element_iterator = iter(elements)
    try:
        first = element_iterator.next() # Will throw exception
    except StopIteration:
        return True
    return all(a == first for a in element_iterator)


def find_pareto_ixs(cost_arrays):
    """
    :param cost_arrays: A collection of nd-arrays representing a grid of costs for different indices.
    :return: A tuple of indices which can be used to index the pareto-efficient points.
    """
    print(c.shape for c in cost_arrays)
    assert all_equal([c.shape for c in cost_arrays])
    flat_ixs, = np.nonzero(is_pareto_efficient(np.reshape(cost_arrays, (len(cost_arrays), -1)).T), )
    ixs = np.unravel_index(flat_ixs, dims=cost_arrays[0].shape)
    return ixs


def prep_perato(attackActions, defenceActions):
    mixInputs = []
    for i in defenceActions:
        tempMix = []
        for j in attackActions:
            #if i!=j:
                #tempReward = list(map(sum, zip(state.get_score(),i,j)))
                #tempReward = list(map(sum, zip(i,j)))
                #tempReward[2] = i[2]-j[2]
                #mixInputs.add(tuple(tempReward))
            tempMix.append(j)
                
        #print (tempMix)

        paretoPoints, actionIndices = simple_cull(tempMix, dominates)
        #paretoPoints  = is_pareto_efficient(np.array(tempMix))

        #print (paretoPoints)
        #print (actionIndices)
        mixInputs.append(paretoPoints)
        #mixInputs.append(np.array(tempMix))
        
        
    return mixInputs



def prep_pareto_efficient(attackActions,defenceActions): #input is a numpy array of attacks for each diffence strategy
    
    mixInputs = np.zeros((defenceActions.shape[0],attackActions.shape[0],attackActions.shape[1]))
    mixIndices = 0

    for i,c in enumerate(defenceActions):
        
        
        attackActions =np.array([[random.randint(70,100) for i in range(3)] for j in range(10)])
        
        paretoPoints  = is_pareto_efficient(attackActions)
        #print(len(attackActions[paretoPoints]))

        mixInputs[i,0:len(attackActions[paretoPoints]),:] = attackActions[paretoPoints]

        #print (paretoPoints)
        #print (i)
        #mixInputs.append(paretoPoints)
    #print(mixInputs)

    return mixInputs



import random

x = np.array([[random.randint(70,100) for i in range(3)] for j in range(10)])
attackActions = [[random.randint(70,100) for i in range(3)] for j in range(10)]
defenceActions = [[random.randint(70,100) for i in range(3)] for j in range(5)]

attackActions2 = np.array(attackActions)
defenceActions2 = np.array(defenceActions)


#inputPoints = [[random.randint(70,100) for i in range(3)] for j in range(1000)]
#inputPoints2 = inputPoints[:]

#inp = np.array(inputPoints2)




pp = prep_pareto_efficient(np.array(attackActions2),np.array(defenceActions2))

ins = find_pareto_ixs(pp)

print(ins)

print(inputPoints)
print("===")
print(inputPoints)

print("==simple_cull=")
start_time = time.time()
paretoPoints, actionIndices = simple_cull(inputPoints, dominates)
print(paretoPoints)
print ("--- %s seconds ---" % (time.time() - start_time))


print("==is_pareto_efficient=")


#print(inp)
start_time = time.time()
poindex = is_pareto_efficient(np.array(inputPoints2))
print(inp[poindex])
print ("--- %s seconds ---" % (time.time() - start_time))





#print("===")
#print(list(paretoPoints))
#print("===")
#print(actionIndices)


#mixInputs = prep_perato(attackActions,defenceActions)
#mixInputs2 = mixInputs[:]

#print("==mixInuptold=")
#start_time = time.time()
#print(mixInputs)
#simple_cull(mixInputs, dominates_two)

#print("final===")
#print(paretoPoints)
#print ("--- %s seconds ---" % (time.time() - start_time))


print("==mixefficent_findpareto=")
#start_time = time.time()
#print(mixInputs)
#find_pareto_ixs(np.array(mixInputs2))

#print("final===")
#print(paretoPoints)
#print ("--- %s seconds ---" % (time.time() - start_time))



#import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D

#fig = plt.figure()
#ax = fig.add_subplot(111, projection='3d')
#dp = np.array(list(dominatedPoints))
#pp = np.array(list(paretoPoints))
#print(pp.shape,dp.shape)
#ax.scatter(dp[:,0],dp[:,1],dp[:,2])
#ax.scatter(pp[:,0],pp[:,1],pp[:,2],color='red')

#import matplotlib.tri as mtri
#triang = mtri.Triangulation(pp[:,0],pp[:,1])
#ax.plot_trisurf(triang,pp[:,2],color='red')
#plt.show()
