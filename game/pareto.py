import numpy as np
from state import State, Config
import time


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



import random
bstart_time = time.time()
while True:
    testDef = []
    for x in range(10):
        testDef.append(np.array([[random.randint(70,100) for i in range(3)] for j in range(10)]))
    print(testDef[0])
        

    start_time = time.time()

    pp = prep_pareto_efficient(testDef)

    print(pp)
    if len(pp) < 10:
        print ("--big-loop- %s seconds ---" % (time.time() - bstart_time))
        break

    print ("--- %s seconds ---" % (time.time() - start_time))
