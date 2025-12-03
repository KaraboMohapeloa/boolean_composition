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
    data0 = dd.io.load('exps_data/exp3_returns_0.h5')
    data1 = dd.io.load('exps_data/exp3_returns_2.h5')
    data2 = dd.io.load('exps_data/exp3_returns_1.h5')
    data3 = dd.io.load('exps_data/exp3_returns_3.h5')
    
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

def plot4_filtered():
    """Generate filtered plot (non-BDQN) excluding specific Boolean task compositions - matches plot4_bdqn()"""
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

    # Exclude specific tasks: MT AND ¬ML, ML AND ¬MT, MT XOR ML, ¬MT, ¬ML, ¬(MT XOR ML), MT NOR ML
    # Indices to exclude: 3, 4, 5, 7, 9, 14, 15 
    exclude_indices = {3, 4, 5, 7, 9, 14, 15}
    
    # Filter tasks
    filtered_tasks = [task for i, task in enumerate(tasks) if i not in exclude_indices]

    plt.ylim(-0.5, 2)
    rc_ = {'figure.figsize':(25,10),'axes.labelsize': 30, 'font.size': 30, 
          'legend.fontsize': 20, 'axes.titlesize': 30}
    sns.set(rc=rc_, style="darkgrid",font_scale = 1.8)
    rc('text', usetex=False)

    n = 2

    data0 = dd.io.load('exps_data/exp3_returns_0.h5')
    data1 = dd.io.load('exps_data/exp3_returns_2.h5')
    data2 = dd.io.load('exps_data/exp3_returns_1.h5')
    data3 = dd.io.load('exps_data/exp3_returns_3.h5')

    types = ["Sparse rewards and Same absorbing set",
              "Dense rewards and Same absorbing set",
              "Sparse rewards and Different absorbing set",
              "Dense rewards and Different absorbing set",
            ]

    # Create filtered data by excluding specified task columns
    filtered_data_rows = []
    for data_array, type_name in zip([data0, data1, data2, data3], types):
        for i in range(len(data_array)):
            row = []
            for t in range(n, 16):  # Original range
                if t not in exclude_indices:  # Only include non-excluded tasks
                    row.append(data_array[i, t])
            row.append(type_name)
            filtered_data_rows.append(row)

    data = pd.DataFrame(filtered_data_rows, columns=filtered_tasks[n:]+["Domain"])
    data = pd.melt(data, "Domain", var_name="Tasks", value_name="Average Returns")

    fig, ax = plt.subplots()
    ax = sns.boxplot(x="Tasks", y="Average Returns", hue="Domain", data=data, linewidth=3, showfliers = False)
    ax.set_ylim(-0.5, 2)
    fig.savefig("plots/dense_filtered.pdf", bbox_inches='tight')

#####################################################################################

def dense_bdqn():
    """Generate filtered BDQN plot excluding specific Boolean task compositions"""
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

    # Exclude specific tasks: MT AND ¬ML, ML AND ¬MT, MT XOR ML, ¬MT, ¬ML, ¬(MT XOR ML), MT NOR ML
    # Indices to exclude: 3, 4, 5, 7, 9, 14, 15 
    # (MT AND ¬ML, ML AND ¬MT, MT NOR ML, ¬MT, ¬ML, ¬(MT XOR ML), MT XOR ML)
    exclude_indices = {3, 4, 5, 7, 9, 14, 15}
    
    # Filter tasks and create mapping for data indices
    filtered_tasks = [task for i, task in enumerate(tasks) if i not in exclude_indices]
    
    # Create mapping from original index to filtered index (after n=2 offset)
    original_to_filtered = {}
    filtered_idx = 0
    for i in range(len(tasks)):
        if i not in exclude_indices and i >= 2:  # Only include tasks from index 2 onwards
            original_to_filtered[i] = filtered_idx
            filtered_idx += 1

    plt.ylim(-1.5, 2)
    rc_ = {'figure.figsize':(25,10),'axes.labelsize': 30, 'font.size': 30, 
          'legend.fontsize': 20, 'axes.titlesize': 30}
    sns.set(rc=rc_, style="darkgrid",font_scale = 1.8)
    rc('text', usetex=False)

    n = 2

    data0 = dd.io.load('exps_data/exp3_bdqn_returns_0.h5')
    data1 = dd.io.load('exps_data/exp3_bdqn_returns_2.h5')
    data2 = dd.io.load('exps_data/exp3_bdqn_returns_1.h5')
    data3 = dd.io.load('exps_data/exp3_bdqn_returns_3.h5')

    types = ["Sparse rewards and Same absorbing set (BDQN)",
              "Dense rewards and Same absorbing set (BDQN)",
              "Sparse rewards and Different absorbing set (BDQN)",
              "Dense rewards and Different absorbing set (BDQN)",
            ]

    # Create filtered data by excluding specified task columns
    filtered_data_rows = []
    for data_array, type_name in zip([data0, data1, data2, data3], types):
        for i in range(len(data_array)):
            row = []
            for t in range(n, 16):  # Original range
                if t not in exclude_indices:  # Only include non-excluded tasks
                    row.append(data_array[i, t])
            row.append(type_name)
            filtered_data_rows.append(row)

    data = pd.DataFrame(filtered_data_rows, columns=filtered_tasks[n:]+["Domain"])
    data = pd.melt(data, "Domain", var_name="Tasks", value_name="Average Returns")

    fig, ax = plt.subplots()
    ax = sns.boxplot(x="Tasks", y="Average Returns", hue="Domain", data=data, linewidth=3, showfliers = False)
    ax.set_ylim(-0.5, 2) # <-- Set the y-axis limits here, on the axes object
    fig.savefig("plots/dense_bdqn.pdf", bbox_inches='tight')

#####################################################################################

def full_dense_bdqn():
    """Generate full BDQN plot including all Boolean task compositions"""
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

    plt.ylim(-1.5, 2)
    rc_ = {'figure.figsize':(25,10),'axes.labelsize': 30, 'font.size': 30, 
          'legend.fontsize': 20, 'axes.titlesize': 30}
    sns.set(rc=rc_, style="darkgrid",font_scale = 1.8)
    rc('text', usetex=False)

    n = 2

    data0 = dd.io.load('exps_data/exp3_bdqn_returns_0.h5')
    data1 = dd.io.load('exps_data/exp3_bdqn_returns_2.h5')
    data2 = dd.io.load('exps_data/exp3_bdqn_returns_1.h5')
    data3 = dd.io.load('exps_data/exp3_bdqn_returns_3.h5')

    types = ["Sparse rewards and Same absorbing set (BDQN)",
              "Dense rewards and Same absorbing set (BDQN)",
              "Sparse rewards and Different absorbing set (BDQN)",
              "Dense rewards and Different absorbing set (BDQN)",
            ]

    # Create full data including all tasks (no exclusions)
    full_data_rows = []
    for data_array, type_name in zip([data0, data1, data2, data3], types):
        for i in range(len(data_array)):
            row = []
            for t in range(n, 16):  # Include all tasks from index 2 onwards
                row.append(data_array[i, t])
            row.append(type_name)
            full_data_rows.append(row)

    data = pd.DataFrame(full_data_rows, columns=tasks[n:]+["Domain"])
    data = pd.melt(data, "Domain", var_name="Tasks", value_name="Average Returns")

    fig, ax = plt.subplots()
    ax = sns.boxplot(x="Tasks", y="Average Returns", hue="Domain", data=data, linewidth=3, showfliers = False)
    ax.set_ylim(-0.5, 2) # <-- Set the y-axis limits here, on the axes object
    fig.savefig("plots/full_dense_bdqn.pdf", bbox_inches='tight')

#####################################################################################

def plot4_bdqn():
    """Legacy function - calls dense_bdqn for backward compatibility"""
    dense_bdqn()

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

def bdqn_plot5():
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
        data0 = dd.io.load('exps_data/exp5_bdqn_returns_'+str(i)+'.h5')[:1000,:]
        data = pd.DataFrame(
            [[data0[j,t] for t in range(n,16)] for j in range(len(data0))],
            columns=tasks[n:]
        )
        data = pd.melt(data, var_name="Tasks", value_name="Average Returns")
        fig, ax = plt.subplots()
        ax = sns.boxplot(x="Tasks", y="Average Returns", data=data, linewidth=3, showfliers=False)
        fig.savefig(f"plots/dense_sp_bdqn_only_exp5_{i}.pdf", bbox_inches='tight')

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

    fig = plt.figure(figsize=(12, 16))
    ax1 = fig.add_subplot(211, projection='3d')
    ax2 = fig.add_subplot(212, projection='3d')
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

def plot_bdqn_EQ_vs_epsilon_bar_EQ():
    data1 = dd.io.load('exps_data/exp1_bdqn_samples_EQs.h5')
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
    # Plot the bar with the smaller mean value last (so it is visually on top)
    if mean1[-1] < mean2[-1]:
        ax.bar(range(1,17), mean2, yerr=std2, align='center', ecolor='black', capsize=5, label="Standard EQ-learning")
        ax.bar(range(1,17), mean1, yerr=std1, align='center', ecolor='black', capsize=5, label="BDQN EQ-learning")
    else:
        ax.bar(range(1,17), mean1, yerr=std1, align='center', ecolor='black', capsize=5, label="BDQN EQ-learning")
        ax.bar(range(1,17), mean2, yerr=std2, align='center', ecolor='black', capsize=5, label="Standard EQ-learning")
    plt.legend()
    plt.title("BDQN EQ-learning vs Standard EQ-learning\n(Cumulative Timesteps to Solve Tasks)")
    plt.xlabel("Number of tasks")
    plt.ylabel('Cumulative timesteps to solve all tasks')
    ax.yaxis.get_major_formatter().set_powerlimits((0, 1))
    plt.xlim(0, 17)
    # plt.show()
    fig.savefig("plots/plot_bdqn_EQ_vs_epsilon_bar_EQ.pdf", bbox_inches='tight')

#####################################################################################

def plot_bdqn_EQ_vs_epsilon_Q_bar():
    data1 = dd.io.load('exps_data/exp1_bdqn_samples_EQs.h5')
    data2 = dd.io.load('exps_data/exp1_samples_Qs.h5')
    
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
        ax.bar(range(1,17), mean2, yerr=std2, align='center', ecolor='black', capsize=5, label="Standard Q-learning")
        ax.bar(range(1,17), mean1, yerr=std1, align='center', ecolor='black', capsize=5, label="BDQN EQ-learning")
    else:
        ax.bar(range(1,17), mean1, yerr=std1, align='center', ecolor='black', capsize=5, label="BDQN EQ-learning")
        ax.bar(range(1,17), mean2, yerr=std2, align='center', ecolor='black', capsize=5, label="Standard Q-learning")
    plt.legend()
    plt.title("BDQN EQ-learning vs Standard Q-learning\n(Cumulative Timesteps to Solve Tasks)")
    plt.xlabel("Number of tasks")
    plt.ylabel('Cumulative timesteps to solve all tasks')
    ax.yaxis.get_major_formatter().set_powerlimits((0, 1))
    plt.xlim(0, 17)
    # plt.show()
    fig.savefig("plots/plot_bdqn_EQ_vs_epsilon_Q_bar.pdf", bbox_inches='tight')

#####################################################################################

def plot_bdqn_EQ_vs_bdqn_Q_bar():
    data1 = dd.io.load('exps_data/exp1_bdqn_samples_EQs.h5')
    data2 = dd.io.load('exps_data/exp1_bdqn_samples_Qs.h5')

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
        ax.bar(range(1,17), mean2, yerr=std2, align='center', ecolor='black', capsize=5, label="BDQN Q-learning")
        ax.bar(range(1,17), mean1, yerr=std1, align='center', ecolor='black', capsize=5, label="BDQN EQ-learning")
    else:
        ax.bar(range(1,17), mean1, yerr=std1, align='center', ecolor='black', capsize=5, label="BDQN EQ-learning")
        ax.bar(range(1,17), mean2, yerr=std2, align='center', ecolor='black', capsize=5, label="BDQN Q-learning")
    plt.legend()
    plt.title("BDQN EQ-learning vs BDQN Q-learning\n(Cumulative Timesteps to Solve Tasks)")
    plt.xlabel("Number of tasks")
    plt.ylabel('Cumulative timesteps to solve all tasks')
    ax.yaxis.get_major_formatter().set_powerlimits((0, 1))
    plt.xlim(0, 17)
    # plt.show()
    fig.savefig("plots/plot_bdqn_EQ_vs_bdqn_Q_bar.pdf", bbox_inches='tight')

#####################################################################################

                                                                                                          

def baseline():
    """
    Baseline comparison: Standard Q-learning vs Extended Q-learning (both epsilon-greedy)
    """
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
    ax.bar(range(1,17), mean2, yerr=std2, align='center', ecolor='black', capsize=5, label="Extended Q-learning (epsilon-greedy)")
    ax.bar(range(1,17), mean1, yerr=std1, align='center', ecolor='black', capsize=5, label="Standard Q-learning (epsilon-greedy)")
    plt.legend()
    plt.title("Standard vs Extended Q-learning (Epsilon-Greedy)\nCumulative Timesteps to Solve Tasks")
    plt.xlabel("Number of tasks")
    plt.ylabel('Cumulative timesteps to solve all tasks')
    ax.yaxis.get_major_formatter().set_powerlimits((0, 1))
    plt.xlim(0, 17)
    # plt.show()
    fig.savefig("plots/baseline.pdf", bbox_inches='tight')



#####################################################################################




# baseline()
# plot_bdqn_EQ_vs_epsilon_bar_EQ()
# plot_bdqn_EQ_vs_epsilon_Q_bar()
# plot_bdqn_EQ_vs_bdqn_Q_bar()
# plot4()
# plot4_bdqn()
# full_dense_bdqn()
# plot4_bdqn()
# plot4_filtered()

# hyper_plot_epsilon()

def compare_plot4_bdqn_vs_filtered():
    """
    Plot BDQN convergence (plot4_bdqn) and baseline convergence (plot4_filtered) side by side for easy comparison.
    """
    # --- Baseline (filtered) ---
    tasks = [r'${M_{\emptyset}}$', r'${M_{\mathcal{U}}}$', r'${M_{T}}\wedge{M_{L}}$', r'${M_{T}}\wedge\neg{M_{L}}$', r'${M_{L}}\wedge\neg{M_{T}}$', r'${M_{T}}\bar{\vee}{M_{L}}$', r'${M_{T}}$', r'$\neg {M_{T}}$', r'${M_{L}}$', r'$\neg {M_{L}}$', r'${M_{T}}\vee{M_{L}}$', r'${M_{T}}\vee\neg{M_{L}}$',
              r'${M_{L}}\vee\neg{M_{T}}$', r'${M_{T}}\bar{\wedge}{M_{L}}$', r'$\neg({M_{T}} \veebar {M_{L}})$', r'${M_{T}} \veebar {M_{L}}$']
    exclude_indices = {3, 4, 5, 7, 9, 14, 15}
    filtered_tasks = [task for i, task in enumerate(tasks) if i not in exclude_indices]
    n = 2
    # Load baseline data
    data0 = dd.io.load('exps_data/exp3_returns_0.h5')
    data1 = dd.io.load('exps_data/exp3_returns_2.h5')
    data2 = dd.io.load('exps_data/exp3_returns_1.h5')
    data3 = dd.io.load('exps_data/exp3_returns_3.h5')
    types = ["Sparse rewards and Same absorbing set", "Dense rewards and Same absorbing set", "Sparse rewards and Different absorbing set", "Dense rewards and Different absorbing set"]
    filtered_data_rows = []
    for data_array, type_name in zip([data0, data1, data2, data3], types):
        for i in range(len(data_array)):
            row = []
            for t in range(n, 16):
                if t not in exclude_indices:
                    row.append(data_array[i, t])
            row.append(type_name)
            filtered_data_rows.append(row)
    baseline_df = pd.DataFrame(filtered_data_rows, columns=filtered_tasks[n:]+["Domain"])
    baseline_df = pd.melt(baseline_df, "Domain", var_name="Tasks", value_name="Average Returns")
    # --- BDQN ---
    data0_bdqn = dd.io.load('exps_data/exp3_bdqn_returns_0.h5')
    data1_bdqn = dd.io.load('exps_data/exp3_bdqn_returns_2.h5')
    data2_bdqn = dd.io.load('exps_data/exp3_bdqn_returns_1.h5')
    data3_bdqn = dd.io.load('exps_data/exp3_bdqn_returns_3.h5')
    types_bdqn = ["Sparse rewards and Same absorbing set (BDQN)", "Dense rewards and Same absorbing set (BDQN)", "Sparse rewards and Different absorbing set (BDQN)", "Dense rewards and Different absorbing set (BDQN)"]
    filtered_data_rows_bdqn = []
    for data_array, type_name in zip([data0_bdqn, data1_bdqn, data2_bdqn, data3_bdqn], types_bdqn):
        for i in range(len(data_array)):
            row = []
            for t in range(n, 16):
                if t not in exclude_indices:
                    row.append(data_array[i, t])
            row.append(type_name)
            filtered_data_rows_bdqn.append(row)
    bdqn_df = pd.DataFrame(filtered_data_rows_bdqn, columns=filtered_tasks[n:]+["Domain"])
    bdqn_df = pd.melt(bdqn_df, "Domain", var_name="Tasks", value_name="Average Returns")
    # --- Plot side by side ---
    rc_ = {'figure.figsize':(25,16),'axes.labelsize': 30, 'font.size': 30, 'legend.fontsize': 20, 'axes.titlesize': 30}
    sns.set(rc=rc_, style="darkgrid",font_scale = 1.8)
    rc('text', usetex=False)
    fig, axes = plt.subplots(2, 1, sharex=True)
    # Baseline
    sns.boxplot(x="Tasks", y="Average Returns", hue="Domain", data=baseline_df, linewidth=3, showfliers=False, ax=axes[0])
    axes[0].set_ylim(-0.5, 2)
    axes[0].set_title("Baseline Convergence")
    # BDQN
    sns.boxplot(x="Tasks", y="Average Returns", hue="Domain", data=bdqn_df, linewidth=3, showfliers=False, ax=axes[1])
    axes[1].set_ylim(-0.5, 2)
    axes[1].set_title("BDQN Convergence")
    plt.tight_layout()
    fig.savefig("plots/compare_bdqn_vs_baseline_filtered.pdf", bbox_inches='tight')
    plt.show()

compare_plot4_bdqn_vs_filtered()