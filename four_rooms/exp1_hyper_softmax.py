import numpy as np
from matplotlib import pyplot as plt
import deepdish as dd
from GridWorld import GridWorld
from library import *

tau_values = [1, 5, 10, 50, 100]
def exp1(tau):
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
            _,stats = Q_learning(env, tau=tau, Q_optimal=Qs[j])
            dataQ[i,j] = stats["T"]
            _,stats = Goal_Oriented_Q_learning(env, tau=tau, Q_optimal=EQs[j])
            dataEQ[i,j] = stats["T"]

    data1 = dd.io.save('exps_data/exp1_samples_Qs.h5_tau='+str(tau), dataQ )
    data2 = dd.io.save('exps_data/exp1_samples_EQs.h5_tau='+str(tau), dataEQ)


for tau in tau_values:
    print("Tau: ", tau)
    exp1(tau)

