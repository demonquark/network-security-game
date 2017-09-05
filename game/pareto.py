import numpy as np
from state import State, Config
import time

config = Config()
state = State(config)


def simple_cull(inputPoints, dominates):
    paretoPoints = []#set()
    candidateRowNr = 0
    #dominatedPoints = set()
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
            #paretoPoints.add(tuple(candidateRow))
            paretoPoints.append(candidateRow)

        if len(inputPoints) == 0:
            break
    return paretoPoints

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

def cull_two(def_paretoPoints):
    mix_paretoPoints = set()
    candidateStartRow = 0
    nonDominated = True
    while True:
        currentRow = 0
        candidateRow = def_paretoPoints[candidateRowNr]
        def_paretoPoints.remove(candidateRow)
        setNonDominated = True
        while len(def_paretoPoints) != 0 and currentRow < len(def_paretoPoints):
            row = def_paretoPoints[currentRow]

        if len(def_paretoPoints) == 0:
            break
    
    return mix_paretoPoints

def prep_perato(attackActions, defenceActions):
    mixInputs = []
    for i in defenceActions:
        tempMix = []
        for j in attackActions:
            if i!=j:
                #tempReward = list(map(sum, zip(state.get_score(),i,j)))
                tempReward = list(map(sum, zip(i,j)))
                tempReward[2] = i[2]-j[2]
                #mixInputs.add(tuple(tempReward))
                tempMix.append(tempReward)
        #print (tempMix)
        paretoPoints = simple_cull(tempMix, dominates)
        #print (paretoPoints)
        mixInputs.append(paretoPoints)
        
        
    return mixInputs


import random


#inputPoints = [[random.randint(70,100) for i in range(3)] for j in range(10)]
attackActions = [[random.randint(70,100) for i in range(3)] for j in range(100)]
defenceActions = [[random.randint(70,100) for i in range(3)] for j in range(100)]
#print(inputPoints)
#paretoPoints, dominatedPoints = simple_cull(inputPoints, dominates)

#print("===")
#print(list(paretoPoints))
print("prep===")
start_time = time.time()
mixInputs = prep_perato(attackActions,defenceActions)

#print(mixInputs)

paretoPoints = simple_cull(mixInputs, dominates_two)
#print("final===")
#print(paretoPoints)

print ("--- %s seconds ---" % (time.time() - start_time))
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
