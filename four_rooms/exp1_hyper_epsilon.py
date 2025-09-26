import numpy as np
from matplotlib import pyplot as plt
import deepdish as dd
from GridWorld import GridWorld
from library import *

epsilon_values = [0.1, 0.3, 0.5, 0.7, 1.0]
def exp1(epsilon):
    env = GridWorld()
    maxiter=300
    T_states=[(3,3),(3,9),(9,3),(9,9)]
    T_states = [[pos,pos] for pos in T_states]
    Tasks = [[],[(3,3),(3,9),(9,3),(9,9)],[(3,3)],[(3,9)],[(9,3)],[(9,9)],[(3,3),(3,9)],[(3,9),(9,3)],[(9,3),(9,9)],[(3,3),(9,3)],[(3,3),(3,9),(9,3)],[(3,3),(3,9),(9,9)],[(3,3),(9,3),(9,9)],[(3,9),(9,3),(9,9)],[(3,3),(9,9)],[(3,9),(9,3)]]

    Qs = dd.io.load('exps_data/4Goals_Optimal_Qs.h5')
    Qs = [{s:v for (s,v) in Q} for Q in Qs]
    EQs = dd.io.load('exps_data/4Goals_Optimal_EQs.h5')
    EQs = [{s:{s__:v__ for (s__,v__) in v} for (s,v) in EQ} for EQ in EQs]

    num_runs = 1
    dataQ = np.zeros((num_runs,len(Tasks))) 
    dataEQ = np.zeros((num_runs,len(Tasks)))

    idxs=np.arange(len(Tasks))
    for i in range(num_runs):
        print("run: ",i)
        np.random.shuffle(idxs)
        for j in idxs:
            print("Task: ",j)
            goals = [[pos,pos] for pos in Tasks[j]]
            env = GridWorld(goals=goals, goal_reward=1, step_reward=-0.01, T_states=T_states)
            _,stats = Q_learning(env, epsilon=epsilon, Q_optimal=Qs[j])
            dataQ[i,j] = stats["T"]
            _,stats = Goal_Oriented_Q_learning(env, epsilon=epsilon, Q_optimal=EQs[j])
            dataEQ[i,j] = stats["T"]

    data1 = dd.io.save('exps_data/exp1_samples_Qs.h5_epsilon='+str(epsilon), dataQ )
    data2 = dd.io.save('exps_data/exp1_samples_EQs.h5_epsilon='+str(epsilon), dataEQ)


for epsilon in epsilon_values:
    print("Epsilon: ", epsilon)
    exp1(epsilon)

