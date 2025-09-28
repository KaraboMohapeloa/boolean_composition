import numpy as np
from collections import defaultdict
from matplotlib import pyplot as plt
from matplotlib import font_manager as fm, rcParams
from matplotlib import rc
import os
import pandas as pd
import seaborn as sns
import deepdish as dd

from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import



#####################################################################################

def plot1():
    n = np.linspace(1,50,50)
    everything = 2**(2**n)
    fact = np.array([np.math.factorial(x) for x in n])
    OR = 2**n - 1
    standard = n
    
    s = 20
    rc_ = {'figure.figsize':(11,8), 'axes.labelsize': 30, 'xtick.labelsize': s, 
           'ytick.labelsize': s, 'legend.fontsize': 25}
    sns.set(rc=rc_, style="darkgrid")
    rc('text', usetex=True)
    
    fig,ax=plt.subplots()
    plt.plot(everything, linewidth=5.0, label="Boolean task algebra")
    plt.plot(OR, linewidth=5.0, label="Disjunction only")
    plt.plot(standard, linewidth=5.0, label="No transfer")
    #plt.plot(fact, '--', label="reference, n!")
    plt.yscale('log', basey=10)
    plt.xlim(1, 10)
    plt.ylim(1, 10**18)
    plt.legend()
    plt.xlabel("Number of tasks")
    plt.ylabel('Number of solvable tasks')
    # plt.show()
    fig.savefig("plots/analytic.pdf", bbox_inches='tight')
#####################################################################################

def plot2():
    data1 = dd.io.load('exps_data/exp1_samples_Qs.h5')
    data2 = dd.io.load('exps_data/exp1_samples_EQs.h5')
    
    mean1 = np.cumsum(data1.mean(axis=0))
    std1 = data1.std(axis=0)
    mean2 = np.cumsum(data2.mean(axis=0))
    std2 = data2.std(axis=0)
    
    s = 20
    rc_ = {'figure.figsize':(11,8),'axes.labelsize': 30, 'xtick.labelsize': s, 
           'ytick.labelsize': s, 'legend.fontsize': 25}
    sns.set(rc=rc_, style="darkgrid")
    rc('text', usetex=True)
    
    fig,ax=plt.subplots()
    ax.bar(range(1,17), mean2, yerr=std2, align='center', ecolor='black', capsize=5, label=r"Extended $Q$-function")
    ax.bar(range(1,17), mean1, yerr=std1, align='center', ecolor='black', capsize=5, label=r"$Q$-function")
    plt.legend()
    plt.xlabel("Number of tasks")
    plt.ylabel('Cumulative timesteps to converge')
    ax.yaxis.get_major_formatter().set_powerlimits((0, 1))
    plt.xlim(0, 17)
    # plt.show()
    fig.savefig("plots/cum_bar.pdf", bbox_inches='tight')
# #####################################################################################

def plot3():
    data1 = dd.io.load('exps_data/exp2_samples_Qs.h5')
    data2 = dd.io.load('exps_data/exp2_samples_EQs.h5')
    
    n = 50
    x = np.arange(1,n+1)
    mean1 = np.cumsum(data1.mean(axis=0))
    mean1 = np.array(list(mean1)+[mean1[-1]]*(n-len(mean1)))
    std1 = data1.std(axis=0)
    std1 = np.array(list(std1)+[std1[-1]]*(n-len(std1)))
    mean2 = np.cumsum(data2.mean(axis=0))
    mean2 = np.array(list(mean2)+[mean2[-1]]*(n-len(mean2)))
    std2 = data2.std(axis=0)
    std2 = np.array(list(std2)+[std2[-1]]*(n-len(std2)))
    
    width = 0.5  # the width of the bars
    s = 20
    rc_ = {'figure.figsize':(11,8),'axes.labelsize': 30, 'xtick.labelsize': s, 
           'ytick.labelsize': s, 'legend.fontsize': 25}
    sns.set(rc=rc_, style="darkgrid")
    rc('text', usetex=True)
    
    fig, ax = plt.subplots()
    ax.bar(x - width/2, mean2, width, yerr=std2, align='center', ecolor='black', label="Boolean task algebra")
    ax.bar(x + width/2, mean1, width, yerr=std1, align='center', ecolor='black', label="Disjunction only")
    ax.legend()
    plt.xlabel("Number of tasks")
    plt.ylabel('Cumulative timesteps to converge')
    ax.yaxis.get_major_formatter().set_powerlimits((0, 1))
    #ax.ticklabel_format(axis='y',style='scientific', useOffset=True)
    fig.tight_layout()
    # plt.show()
    fig.savefig("plots/40goals_cum_bar.pdf", bbox_inches='tight')

#####################################################################################

def plot4():
    tasks = [r'${M_{\emptyset}}$',
              r'${M_{\mathcal{U}}}$',
              r'${M_{T}}\wedge{M_{L}}$',
              r'${M_{T}}\wedge\neg{M_{L}}$',
              r'${M_{L}}\wedge\neg{M_{T}}$',
              r'${M_{T}}\bar{\vee}{M_{L}}$',
              r'${M_{T}}$',
              r'$\neg {M_{T}}$',
              r'${M_{L}}$',
              r'$\neg {M_{L}}$',
              r'${M_{T}}\vee{M_{L}}$',
              r'${M_{T}}\vee\neg{M_{L}}$',
              r'${M_{L}}\vee\neg{M_{T}}$',
              r'${M_{T}}\bar{\wedge}{M_{L}}$',
              r'$\neg({M_{T}} \veebar {M_{L}})$',
              r'${M_{T}} \veebar {M_{L}}$'
              ]
    
    plt.ylim(-0.5, 2)
    rc_ = {'figure.figsize':(30,10),'axes.labelsize': 30, 'font.size': 30, 
          'legend.fontsize': 20, 'axes.titlesize': 30}
    sns.set(rc=rc_, style="darkgrid",font_scale = 1.8)
    rc('text', usetex=False)
    
    n = 2
    
    #data0 = dd.io.load('exps_data/trash/exp3_returns_optimal.h5')
    data0 = dd.io.load('exps_data/exp3_returns_0.h5')/10
    data1 = dd.io.load('exps_data/exp3_returns_2.h5')/10
    data2 = dd.io.load('exps_data/exp3_returns_1.h5')/10
    data3 = dd.io.load('exps_data/exp3_returns_3.h5')/10
    
    types = ["Sparse rewards and Same absorbing set",
              "Dense rewards and Same absorbing set",
              "Sparse rewards and Different absorbing set",
              "Dense rewards and Different absorbing set",
            ]
    
    data = pd.DataFrame(
    [[data0[i,t] for t in range(n,16)]+[types[0]] for i in range(len(data1))] +
    [[data1[i,t] for t in range(n,16)]+[types[1]] for i in range(len(data1))] +
    [[data2[i,t] for t in range(n,16)]+[types[2]] for i in range(len(data1))] +
    [[data3[i,t] for t in range(n,16)]+[types[3]] for i in range(len(data1))],
      columns=tasks[n:]+["Domain"])
    data = pd.melt(data, "Domain", var_name="Tasks", value_name="Average Returns")
    
    fig, ax = plt.subplots()
    ax = sns.boxplot(x="Tasks", y="Average Returns", hue="Domain", data=data, linewidth=3, showfliers = False)
    # plt.show()
    fig.savefig("plots/dense.pdf", bbox_inches='tight')


#####################################################################################

def plot5():
    tasks = [r'${M_{\emptyset}}$',
              r'${M_{\mathcal{U}}}$',
              r'${M_{T}}\wedge{M_{L}}$',
              r'${M_{T}}\wedge\neg{M_{L}}$',
              r'${M_{L}}\wedge\neg{M_{T}}$',
              r'${M_{T}}\bar{\vee}{M_{L}}$',
              r'${M_{T}}$',
              r'$\neg {M_{T}}$',
              r'${M_{L}}$',
              r'$\neg {M_{L}}$',
              r'${M_{T}}\vee{M_{L}}$',
              r'${M_{T}}\vee\neg{M_{L}}$',
              r'${M_{L}}\vee\neg{M_{T}}$',
              r'${M_{T}}\bar{\wedge}{M_{L}}$',
              r'$\neg({M_{T}} \veebar {M_{L}})$',
              r'${M_{T}} \veebar {M_{L}}$'
              ]
        
    s = 20
    rc_ = {'figure.figsize':(30,10),'axes.labelsize': 30, 'font.size': 30, 
          'legend.fontsize': 20, 'axes.titlesize': 30}
    sns.set(rc=rc_, style="darkgrid",font_scale = 1.8)
    rc('text', usetex=False)
    
    n = 2
    
    for i in range(4):
        data0 = dd.io.load('exps_data/exp5_returns_'+str(i)+'.h5')[:1000,:]
        data1 = dd.io.load('exps_data/exp4_returns_'+str(i)+'.h5')[:1000,:]
        
        types = ["Optimal",
                  "Composed",
                ]
        
        data = pd.DataFrame(
        [[data0[i,t] for t in range(n,16)]+[types[0]] for i in range(len(data1))] +
        [[data1[i,t] for t in range(n,16)]+[types[1]] for i in range(len(data1))],
          columns=tasks[n:]+[""])
        data = pd.melt(data, "", var_name="Tasks", value_name="Average Returns")
        
        fig, ax = plt.subplots()
        ax = sns.boxplot(x="Tasks", y="Average Returns", hue="", data=data, linewidth=3, showfliers = False)
        # plt.show()
        fig.savefig("plots/dense_sp_"+str(i)+".pdf", bbox_inches='tight')

#####################################################################################

def hyper_plot_general(param_values, param_name, file_suffix, plot_filename):
    """
    Generalized 3D bar plot for hyperparameter sweeps.
    param_values: list of parameter values (e.g., tau or epsilon)
    param_name: string, name for axis label (e.g., 'Tau', 'Epsilon')
    file_suffix: string, suffix for file loading (e.g., 'tau', 'epsilon')
    plot_filename: string, output filename for plot
    """
    Z1 = []
    Z2 = []
    for val in param_values:
        data_q = dd.io.load(f'exps_data/exp1_samples_Qs.h5_{file_suffix}={val}')
        data_eq = dd.io.load(f'exps_data/exp1_samples_EQs.h5_{file_suffix}={val}')
        mean_cum_q = np.cumsum(np.mean(data_q, axis=0)) / 1e5
        mean_cum_eq = np.cumsum(np.mean(data_eq, axis=0)) / 1e5
        Z1.append(mean_cum_q)
        Z2.append(mean_cum_eq)
    Z1 = np.array(Z1)
    Z2 = np.array(Z2)
    num_tasks = Z1.shape[1]
    y = np.arange(0, num_tasks)
    x = np.array(param_values)

    fig = plt.figure(figsize=(22, 8))
    ax1 = fig.add_subplot(121, projection='3d')
    ax2 = fig.add_subplot(122, projection='3d')
    # Make bar thickness proportional to number of tasks
    # Bars are thin along the epsilon axis (dx), wide along number of tasks axis (dy)
    dx = (x[1] - x[0]) * 0.2 if len(x) > 1 else 0.2  # thin along param axis
    dy = 0.8  # wide along tasks axis
    cmap = plt.get_cmap('tab10')
    param_colors = [cmap(i % 10) for i in range(len(x))]
    for i, val in enumerate(x):
        for j, task in enumerate(y):
            xpos = val - dx/2
            ypos = task - dy/2
            zpos = 0
            height_q = Z1[i, j]
            height_eq = Z2[i, j]
            ax1.bar3d(xpos, ypos, zpos, dx, dy, height_q, color=param_colors[i], alpha=0.7)
            ax2.bar3d(xpos, ypos, zpos, dx, dy, height_eq, color=param_colors[i], alpha=0.7)
            if j == num_tasks - 1:
                ax1.text(val, task, height_q, f'{height_q:.2f}', color='black', fontsize=12, ha='center', va='bottom')
                ax2.text(val, task, height_eq, f'{height_eq:.2f}', color='black', fontsize=12, ha='center', va='bottom')
    ax1.set_xlabel(param_name)
    ax1.set_ylabel('Number of tasks')
    ax1.set_zlabel('Cumulative Q ($\\times 10^5$)')
    ax1.set_title('$Q$-function')
    ax1.set_xticks(x)
    ax2.set_xlabel(param_name)
    ax2.set_ylabel('Number of tasks')
    ax2.set_zlabel('Cumulative EQ ($\\times 10^5$)')
    ax2.set_title('Extended $Q$-function')
    ax2.set_xticks(x)
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=param_colors[i], label=f'{param_name}={val}') for i, val in enumerate(x)]
    ax1.legend(handles=legend_elements, loc='upper left')
    ax2.legend(handles=legend_elements, loc='upper left')
    plt.tight_layout()
    plt.show()
    fig.savefig(plot_filename, bbox_inches='tight')

# Usage examples:
def hyper_plot_softmax():
    tau_values = [1, 5, 10, 50, 100]
    hyper_plot_general(tau_values, 'Tau', 'tau', 'plots/hyper_softmax.pdf')

def hyper_plot_epsilon():
    epsilon_values = [0.1, 0.3, 0.5, 0.7, 1.0]
    hyper_plot_general(epsilon_values, 'Epsilon', 'epsilon', 'plots/hyper_epsilon.pdf')


#####################################################################################

def plot_bdqn_bar():
    data1 = dd.io.load('exps_data/bdqn/exp1_epsilon_samples_Qs.h5')
    data2 = dd.io.load('exps_data/bdqn/exp1_bdqn_samples_EQs.h5')

    mean1 = np.cumsum(data1.mean(axis=0))
    std1 = data1.std(axis=0)
    mean2 = np.cumsum(data2.mean(axis=0))
    std2 = data2.std(axis=0)

    s = 20
    rc_ = {'figure.figsize':(11,8),'axes.labelsize': 30, 'xtick.labelsize': s, 
        'ytick.labelsize': s, 'legend.fontsize': 25}
    sns.set(rc=rc_, style="darkgrid")
    rc('text', usetex=True)

    fig,ax=plt.subplots()
    # Plot the bar with the smaller mean value last (so it is visually on top)
    if mean1[-1] < mean2[-1]:
        ax.bar(range(1,17), mean2, yerr=std2, align='center', ecolor='black', capsize=5, label=r"Extended $Q$-function")
        ax.bar(range(1,17), mean1, yerr=std1, align='center', ecolor='black', capsize=5, label=r"$Q$-function")
    else:
        ax.bar(range(1,17), mean1, yerr=std1, align='center', ecolor='black', capsize=5, label=r"$Q$-function")
        ax.bar(range(1,17), mean2, yerr=std2, align='center', ecolor='black', capsize=5, label=r"Extended $Q$-function")
    plt.legend()
    plt.xlabel("Number of tasks")
    plt.ylabel('Cumulative timesteps to converge')
    ax.yaxis.get_major_formatter().set_powerlimits((0, 1))
    plt.xlim(0, 17)
    # plt.show()
    fig.savefig("plots/bdqn_average_cum_bar.pdf", bbox_inches='tight')
#####################################################################################

plot_bdqn_bar()

