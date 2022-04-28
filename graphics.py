import matplotlib.pyplot as plt

#%%
gamma = [0,0.2,0.4,0.6,0.8,1]
values_g = ['0','0.2','0.4','0.6','0.8','1']
F1_g_clasif = [67.4,61.23,62.54,69,73.73,75.2]
cluster_g_acc = [74,70.2,69.2,76.5,82.1,86]
F1_g_cluster = [65.2,65.1,64.3,63,56.5,51.3]

tau = [1.5,1.75,2,2.25,2.5]
values_t = ['1.5','1.75','2','2.25','2.5']
F1_t_clasif = [42.33,56.9,69.1,72.27,76.34]
cluster_t_acc = [47.8,64.2,76.5,80.6,85.6]
F1_t_cluster = [62.3,63.11,63,59.5,56.83]

mb = [10,20,30,40,50]
values_mb = ['5/10','10/20','15/30','20/40','20/50']
F1_mb_clasif = [73.84, 71.5, 68.72, 68.48, 68.97]
cluster_mb_acc = [86.6,81.3,78.8,77.8,77.2]
F1_mb_structure = [37.53,49.61,55.78,61.26,62.82] 

plt.figure()
plt.plot(mb, F1_mb_clasif, label = 'F1 Sample Classification',marker = 'o') 
plt.plot(mb, cluster_mb_acc, label = 'Cluster Precision', marker = '^')
plt.plot(mb, F1_mb_structure, label = 'F1 Gallery Structure', marker = 's')
plt.ylim(35,90)
# plt.xlim(0,1)
plt.xticks(mb, values_mb)
plt.xlabel('memory budget (l/m)')
plt.ylabel('%')
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.grid()
# plt.show()
plt.savefig('./GRAPHS/memory_budget3.png',  dpi = 1000, bbox_inches = 'tight')


# plt.figure()
# plt.plot(gamma, F1_g_clasif, label = 'F1 Sample Classification', marker = 'o')
# plt.plot(gamma, cluster_g_acc, label = 'Cluster Precision', marker = '^')
# plt.plot(gamma, F1_g_cluster, label = 'F1 Cluster Structure',marker = 's') 
# plt.ylim(40,90)
# # plt.xlim(0,1)
# plt.xticks(gamma, values_g)
# plt.xlabel(r'$\gamma$')
# plt.ylabel('%')
# # plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
# plt.grid()

# plt.savefig('./GRAPHS/gamma_3.png',  dpi = 1000, bbox_inches = 'tight')

# plt.figure()
# plt.plot(tau, F1_t_clasif, label = 'F1 Sample Classification', marker = 'o')
# plt.plot(tau, cluster_t_acc, label = 'Cluster Precision', marker = '^')
# plt.plot(tau, F1_t_cluster, label = 'F1 Gallery Structure',marker = 's') 
# plt.ylim(40,90)
# plt.xticks(tau,values_t)
# plt.xlabel('expansion threshold (' + chr(964) + ')')
# plt.ylabel('%')
# # plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
# plt.grid()
# plt.savefig('./GRAPHS/exp_thr_3.png',  dpi = 1000, bbox_inches = 'tight')

#%%

memory_budget = np.array([20,30,50])

#CPR
CPR_F1_clasif = np.array([72.1,69.7,69.4])
CPR_F1_clasif_var = np.array([0.3,0.64,0.86])

CPR_F1_structure = np.array([49.2,56.4,62.67])
CPR_F1_structure_var = np.array([0.58,0.47,0.19])

CPR_acc_Cluster = np.array([81.1,78.1,76.2])
CPR_acc_cluster_var = np.array([0.77,0.7,0.42])

#IOM
IOM_F1_clasif = np.array([67,64.7,60.3])
IOM_F1_clasif_var = np.array([0.16,0.57,3.44])

IOM_F1_structure = np.array([51.7,59,64.5])
IOM_F1_structure_var = np.array([1.05,0.8,1.35])

IOM_acc_Cluster = np.array([74.8,72,66.6])
IOM_acc_cluster_var = np.array([0.28,1.05,2.55])

#TEMPORAL
T_F1_clasif = np.array([75.9,77.8,79.8])
T_F1_clasif_var = np.array([0.55,0.49,0.54])

T_F1_structure = np.array([34.8,40.6,46.7])
T_F1_structure_var = np.array([0.25,0.59,0.28])

T_acc_Cluster = np.array([92.9,92.2,90])
T_acc_Cluster_var = np.array([0.32,0.2,0.44])

#RANDOM
R_F1_clasif = np.array([65.7,64.2,61.4])
R_F1_clasif_var = [0.46,0.47,0.67]

R_F1_structure = np.array([47.5,55.1,62])
R_F1_structure_var = np.array([0.32,0.53,0.63])

R_acc_Cluster = np.array([72.3,69.2,64.8])
R_acc_Cluster_var = np.array([0.69,0.8,0.83])

#EXSTREAM
E_F1_clasif = np.array([86.1,86.2,85.8])
E_F1_clasif_var = np.array([0.3,0.36,1.6])

E_F1_structure = np.array([40.7,45.5,50.2])
E_F1_structure_var = np.array([0.06,0.6,0.09])

E_acc_Cluster = np.array([92,91,90])
E_acc_Cluster_var = np.array([0.2,0.09,0.47])


#CLASIFICATION GRAPH
# fig, ax = plt.subplots()
# ax.plot(memory_budget, CPR_F1_clasif, '-', label = 'Ours')
# ax.plot(memory_budget, IOM_F1_clasif, '-',label = 'IOM')
# ax.plot(memory_budget, T_F1_clasif, '-', label = 'Uniform')
# ax.plot(memory_budget, R_F1_clasif, '-', label = 'Random')
# ax.plot(memory_budget, E_F1_clasif, '-', label= 'ExStream')
# ax.fill_between(memory_budget, CPR_F1_clasif - CPR_F1_clasif_var, CPR_F1_clasif + CPR_F1_clasif_var, alpha=0.2)
# ax.fill_between(memory_budget, IOM_F1_clasif - IOM_F1_clasif_var, IOM_F1_clasif + IOM_F1_clasif_var, alpha=0.2)
# ax.fill_between(memory_budget, T_F1_clasif - T_F1_clasif_var, T_F1_clasif + T_F1_clasif_var, alpha=0.2)
# ax.fill_between(memory_budget, R_F1_clasif - R_F1_clasif_var, R_F1_clasif + R_F1_clasif_var, alpha=0.2)
# ax.fill_between(memory_budget, E_F1_clasif - E_F1_clasif_var, E_F1_clasif + E_F1_clasif_var, alpha=0.2)
# plt.ylim(55,90)
# fig.set_figwidth(4)
# plt.xticks(memory_budget, ['20','30','50'])
# # plt.legend(bbox_to_anchor=(1.05, 1))#, prop = {'size':10})
# plt.rc('xtick',labelsize = 15)
# plt.rc('ytick',labelsize = 15)
# plt.xlabel('Memory Budget',size = 15)
# plt.ylabel('Sample Classification', size = 15)
# plt.savefig('./GRAPHS/Sample_clasif.png', dpi = 1000, bbox_inches="tight")  
# plt.close('all')
# plt.grid()
# plt.show()

#CLUSTER STRUCTURE 
fig, ax = plt.subplots()
ax.plot(memory_budget, CPR_F1_structure, '-', label = 'Ours')
ax.plot(memory_budget, IOM_F1_structure, '-',label = 'IOM')
ax.plot(memory_budget, T_F1_structure, '-', label = 'Uniform')
ax.plot(memory_budget, R_F1_structure, '-', label = 'Random')
ax.plot(memory_budget, E_F1_structure, '-', label= 'ExStream')
ax.fill_between(memory_budget, CPR_F1_structure - CPR_F1_structure_var, CPR_F1_structure + CPR_F1_structure_var, alpha=0.2)
ax.fill_between(memory_budget, IOM_F1_structure - IOM_F1_structure_var, IOM_F1_structure + IOM_F1_structure_var, alpha=0.2)
ax.fill_between(memory_budget, T_F1_structure - T_F1_structure_var, T_F1_structure + T_F1_structure_var, alpha=0.2)
ax.fill_between(memory_budget, R_F1_structure - R_F1_structure_var, R_F1_structure + R_F1_structure_var, alpha=0.2)
ax.fill_between(memory_budget, E_F1_structure - E_F1_structure_var, E_F1_structure + E_F1_structure_var, alpha=0.2)
# plt.ylim(55,90)
fig.set_figwidth(4)
plt.xticks(memory_budget, ['20','30','50'])
# plt.legend(bbox_to_anchor=(1.05, 1))#, prop = {'size':10})
plt.rc('xtick',labelsize = 15)
plt.rc('ytick',labelsize = 15)
plt.xlabel('Memory Budget',size = 15)
plt.ylabel('Cluster Structure', size = 15)
plt.savefig('./GRAPHS/Cluster_structure.png', dpi = 1000, bbox_inches="tight")  
# plt.close('all')

# #CLUSTER ACCURACY
# fig, ax = plt.subplots()
# ax.plot(memory_budget, CPR_acc_Cluster, '-', label = 'Ours')
# ax.plot(memory_budget, IOM_acc_Cluster, '-',label = 'IOM')
# ax.plot(memory_budget, T_acc_Cluster, '-', label = 'Uniform')
# ax.plot(memory_budget, R_acc_Cluster, '-', label = 'Random')
# ax.plot(memory_budget, E_acc_Cluster, '-', label= 'ExStream')
# ax.fill_between(memory_budget, CPR_acc_Cluster - CPR_acc_cluster_var, CPR_acc_Cluster + CPR_acc_cluster_var, alpha=0.2)
# ax.fill_between(memory_budget, IOM_acc_Cluster - IOM_acc_cluster_var, IOM_acc_Cluster + IOM_acc_cluster_var, alpha=0.2)
# ax.fill_between(memory_budget, T_acc_Cluster - T_acc_Cluster_var, T_acc_Cluster + T_acc_Cluster_var, alpha=0.2)
# ax.fill_between(memory_budget, R_acc_Cluster - R_acc_Cluster_var, R_acc_Cluster + R_acc_Cluster_var, alpha=0.2)
# ax.fill_between(memory_budget, E_acc_Cluster - E_acc_Cluster_var, E_acc_Cluster + E_acc_Cluster_var, alpha=0.2)
# plt.ylim(60,95)
# fig.set_figwidth(4)
# plt.xticks(memory_budget, ['20','30','50'])
# # plt.legend(bbox_to_anchor=(1.05, 1))#, prop = {'size':10})
# plt.rc('xtick',labelsize = 15)
# plt.rc('ytick',labelsize = 15)
# plt.xlabel('Memory Budget',size = 15)
# plt.ylabel('Cluster Acc', size = 15)
# plt.savefig('./GRAPHS/Cluster_acc.png', dpi = 1000, bbox_inches="tight")  
# plt.close('all')

#%%
from matplotlib.lines import Line2D
colors = {'IOM': 'tab:orange', 'Random': 'tab:red', 'Ours': 'tab:blue', 'Uniform': 'tab:green', 'ExStream': 'tab:purple'}
K = [1,2,3,4]

CPR_clasif = [54.5, 63.78, 67.18, 69.4] 
CPR_clasif_var = [0.41, 0.22, 0.12, 0.86]
CPR_cluster_prec = np.array([76.9,76.9,76.9,76.9])
CPR_cluster_prec_var = np.array([0.42,0.42,0.42,0.42])

IOM_clasif = [50.53, 57.8, 59, 60]
IOM_clasif_var = [2.28,3.3,3.2, 3.4]
IOM_cluster_prec = np.array([67,67,67, 67])
IOM_cluster_prec_var = np.array([2.5,2.5,2.5,2.5])

R_clasif = [46.3, 56, 59, 61.4]
R_clasif_var = [0.7, 0.5, 0.4, 0.67]
R_cluster_prec = np.array([65.4,65.4,65.4, 65.4])
R_cluster_prec_var = np.array([0.8,0.8,0.8,0.8])

T_clasif = [55.8,70.6,76.57,79.8]
T_clasif_var = [0.76,0.29,0.68,0.53]
T_cluster_prec = np.array([91.5,91.5,91.5, 91.5])
T_cluster_prec_var = np.array([0.44,0.44,0.44,0.44])

E_clasif = [63.8,77.3,82.8,85.8]
E_clasif_var = [0.9,0.88,1,1.6]
E_cluster_prec = np.array([91.2,91.2,91.2, 91.2])
E_cluster_prec_var = np.array([0.47,0.47,0.47,0.47])

fig, ax0 = plt.subplots()
ax0.errorbar(K, IOM_clasif, yerr=IOM_clasif_var, fmt='-o', color = colors['IOM'])
# ax0.plot(K, IOM_cluster_prec, '-',  label = 'IOM', color = colors['IOM'])
ax0.plot(K, IOM_cluster_prec, '--',  label = 'IOM', color = colors['IOM'])
ax0.fill_between(K, IOM_cluster_prec - IOM_cluster_prec_var, IOM_cluster_prec + IOM_cluster_prec_var, alpha=0.2, color = colors['IOM'])

ax0.errorbar(K, R_clasif, yerr=R_clasif_var, fmt='-o', color = colors['Random'])
# ax0.plot(K, R_cluster_prec, '-',  label = 'Random',color = colors['Random'])
ax0.plot(K, R_cluster_prec, '--',  label = 'Random', color = colors['Random'])
ax0.fill_between(K, R_cluster_prec - R_cluster_prec_var, R_cluster_prec + R_cluster_prec_var, alpha=0.2, color = colors['Random'])

ax0.errorbar(K, CPR_clasif, yerr=CPR_clasif_var, fmt='-o', color = colors['Ours'])
# ax0.plot(K, CPR_cluster_prec, '-', label = 'Ours', color = colors['Ours'])
ax0.plot(K, CPR_cluster_prec, '--',  label = 'Ours', color = colors['Ours'])
ax0.fill_between(K, CPR_cluster_prec - CPR_cluster_prec_var, CPR_cluster_prec + CPR_cluster_prec_var, alpha=0.2, color = colors['Ours'])

ax0.errorbar(K, T_clasif, yerr=T_clasif_var, fmt='-o', color = colors['Uniform'])
# ax0.plot(K, T_cluster_prec, '-',  label = 'Uniform',color = colors['Uniform'])
ax0.plot(K, T_cluster_prec, '--',  label = 'Uniform', color = colors['Uniform'])
ax0.fill_between(K, T_cluster_prec - T_cluster_prec_var, T_cluster_prec + T_cluster_prec_var, alpha=0.2, color = colors['Uniform'])

ax0.errorbar(K, E_clasif, yerr=E_clasif_var, fmt='-o', color = colors['ExStream'])
# ax0.plot(K, E_cluster_prec, '-', label = 'ExStream', color = colors['ExStream'])
ax0.plot(K, E_cluster_prec, '--',  label = 'ExStream', color = colors['ExStream'])
ax0.fill_between(K, E_cluster_prec - E_cluster_prec_var, E_cluster_prec + E_cluster_prec_var, alpha=0.2, color = colors['ExStream'])

plt.ylim(45,95)
plt.xticks(K, ['1','2','3', '4'])
plt.rc('xtick',labelsize = 15)
plt.rc('ytick',labelsize = 15)
plt.legend(bbox_to_anchor=(1.05, 1))#, prop = {'size':10})
plt.xlabel('K',size = 15)
plt.ylabel('%', size = 15)
plt.grid()
plt.savefig('./GRAPHS/Sample_clasif_kVAR.png', dpi = 1000, bbox_inches="tight")  
plt.close('all')

#%%
import numpy as np 

colors = {'IOM': 'tab:orange', 'Random': 'tab:red', 'Ours': 'tab:blue', 'Uniform': 'tab:green', 'ExStream': 'tab:purple'}
X = ['F1','Precision', 'Recall']
w = 0.12

values = ['IOM', 'Random', 'Ours', 'ExStream', 'Uniform']
precision = [52.4,48.4,48.3,33.9,30.7]
prec_var = [1.6,0.43,0.12,0.05,0.24]

recall = [83.9,86.5,90,96.4,97.8]
recall_var = [0.8,1.1,0.4,0.43,0.23]

F1 = [64.5,62,62.7,50.2,46.7]
F1_var = [1.35,0.63,0.19,0.08,0.28]

IOM = [64.5,52.4,83.9]
Random = [62, 48.4,86.5]
Ours = [62.7, 48.3,90]
ExStream = [50.2, 33.9,96.4]
Uniform = [46.7, 30.7,97.8]

IOM_var = [1.358308053, 1.60623784, 0.8219218671]
Random_var = [0.6366229946, 0.4320493799, 1.10855261]
CPR_var = [0.1978035239, 0.1239175353, 0.4027681991]
ExStream_var = [0.08977985543,0.04714045208,0.4320493799]
Uniform_var = [0.2854266279, 0.2494438258, 0.2357022604]
										
bar1 = np.arange(len(X))
bar2 = [i+w for i in bar1]
bar3 = [i+w for i in bar2]
bar4 = [i+w for i in bar3]
bar5 = [i+w for i in bar4]
fig, ax = plt.subplots()
ax.bar(bar1, IOM, w, yerr=IOM_var, label='IOM', color = colors['IOM'])
ax.bar(bar2, Random, w, yerr=Random_var, label='Random', color = colors['Random'])
ax.bar(bar3, Ours, w,  yerr=CPR_var, label='Ours', color = colors['Ours'])
ax.bar(bar4, ExStream, w,  yerr=ExStream_var, label='ExStream', color = colors['ExStream'])
ax.bar(bar5, Uniform, w,  yerr=Uniform_var, label='Uniform', color = colors['Uniform'])
fig.set_figwidth(4)
# fig.tight_layout(pad=5)
plt.xticks(bar3,X)
plt.ylabel('%')
plt.xlabel('Gallery Structure')
plt.ylim(0,100)

# c = []
# for v in values:
#     c.append(colors[v])
# plt.ylim(30,70)
# plt.bar(values, F1, yerr=F1_var, color = c)
# plt.xticks(values)

# fig, ax = plt.subplots()
# for i,v in enumerate(values):
#     ax.errorbar(recall[i], precision[i], yerr=prec_var[i],  xerr=recall_var[i], fmt='-o', label = values[i], color = colors[values[i]])
# plt.legend(bbox_to_anchor=(1.05, 1))#, prop = {'size':10})
plt.savefig('./GRAPHS/ClusterPrecRecall_var_thin.png', dpi = 1000, bbox_inches="tight")  
# plt.show()
    


#%%
import numpy as np
N = 21
x = np.linspace(0, 10, 11)
y = [3.9, 4.4, 10.8, 10.3, 11.2, 13.1, 14.1,  9.9, 13.9, 15.1, 12.5]

# fit a linear curve an estimate its y-values and their error.
a, b = np.polyfit(x, y, deg=1)
y_est = a * x + b
y_err = x.std() * np.sqrt(1/len(x) +
                          (x - x.mean())**2 / np.sum((x - x.mean())**2))
fig, ax = plt.subplots()
ax.plot(x, y_est, '-')
ax.fill_between(x, y_est - y_err, y_est + y_err, alpha=0.2)
ax.plot(x, y, 'o', color='tab:brown')
plt.show()