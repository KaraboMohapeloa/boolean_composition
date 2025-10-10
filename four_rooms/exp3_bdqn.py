import numpy as np
from matplotlib import pyplot as plt
import deepdish as dd
from GridWorld import GridWorld
from bdqn_library import Bootstrapped_Goal_Oriented_Q_learning, average_ensemble_evf, AND, OR, NOT, EQ_P, EQ_V

T_states = [(3,3),(3,9),(9,3),(9,9)]
T_states = [[pos,pos] for pos in T_states]

Bases = [[(3,3),(3,9)], [(3,3),(9,3)]]
Tasks = [[],[(3,3),(3,9),(9,3),(9,9)],[(3,3)],[(3,9)],[(9,3)],[(9,9)],[(3,3),(3,9)],[(9,3),(9,9)],[(3,3),(9,3)],[(3,9),(9,9)],[(3,3),(3,9),(9,3)],[(3,3),(3,9),(9,9)],[(3,3),(9,3),(9,9)],[(3,9),(9,3),(9,9)],[(3,3),(9,9)],[(3,9),(9,3)]]

# Sparse rewards, Same terminal states
types = [(True,True),(True,False),(False,True),(False,False)] 

maxiter = 500
num_runs = 100000

def evaluate(goals, EQ):
    env = GridWorld(goals=goals, T_states=T_states)
    policy = EQ_P(EQ)
    state = env.reset()
    done = False
    t = 0
    G = 0
    while not done and t < 100:
        action = policy[state]
        state_, reward, done, _ = env.step(action)
        state = state_
        G += reward
        t += 1
    return G

for t in range(len(types)):
    print("type: ", t)
    
    # Learning universal bounds (min and max tasks) using BDQN
    env = GridWorld(goals=T_states, dense_rewards=not types[t][0])
    EQ_max_ensemble, _ = Bootstrapped_Goal_Oriented_Q_learning(env, maxiter=maxiter, verbose=False)
    EQ_max = average_ensemble_evf(EQ_max_ensemble)
    
    env = GridWorld(goals=T_states, goal_reward=-0.1, dense_rewards=not types[t][0])
    EQ_min_ensemble, _ = Bootstrapped_Goal_Oriented_Q_learning(env, maxiter=maxiter, verbose=False)
    EQ_min = average_ensemble_evf(EQ_min_ensemble)
    
    # Learning base tasks and doing composed tasks using BDQN
    goals = Bases[0]
    goals = [[pos,pos] for pos in goals]
    env = GridWorld(goals=goals, dense_rewards=not types[t][0], T_states=T_states if types[t][1] else goals)
    A_ensemble, _ = Bootstrapped_Goal_Oriented_Q_learning(env, maxiter=maxiter, verbose=False, T_states=None if types[t][1] else T_states)
    A = average_ensemble_evf(A_ensemble)
    
    goals = Bases[1]
    goals = [[pos,pos] for pos in goals]
    env = GridWorld(goals=goals, dense_rewards=not types[t][0], T_states=T_states if types[t][1] else goals)
    B_ensemble, _ = Bootstrapped_Goal_Oriented_Q_learning(env, maxiter=maxiter, verbose=False, T_states=None if types[t][1] else T_states)
    B = average_ensemble_evf(B_ensemble)
    
    NEG = lambda x: NOT(x, EQ_max=EQ_max, EQ_min=EQ_min)
    XOR = lambda EQ1, EQ2: OR(AND(EQ1, NEG(EQ2)), AND(EQ2, NEG(EQ1)))
    composed = [EQ_min, EQ_max, AND(A,B), AND(A,NEG(B)), AND(B,NEG(A)), NOT(OR(A,B), EQ_max=EQ_max, EQ_min=EQ_min), A, NEG(A), B, NEG(B), OR(A,B), OR(A,NEG(B)), OR(B,NEG(A)), NOT(AND(A,B), EQ_max=EQ_max, EQ_min=EQ_min), NOT(XOR(A,B), EQ_max=EQ_max, EQ_min=EQ_min), XOR(A,B)]
    
    data = np.zeros((num_runs, len(Tasks)))
    for i in range(num_runs):
        for j in range(len(Tasks)):
            goals = [[pos,pos] for pos in Tasks[j]]
            data[i,j] = evaluate(goals, composed[j])
    dd.io.save('exps_data/exp3_bdqn_returns_' + str(t) + '.h5', data)