import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme()

plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

costs_gpc = np.load("./gpc.npy", allow_pickle=True)
costs_optimal = np.load("./optimal.npy", allow_pickle=True)
costs_optimistic = np.load("./optftrlc_good.npy", allow_pickle=True)
costs_optimistic_bad = np.load("./optftrlc_bad.npy", allow_pickle=True)


costs_gpc_cum = costs_gpc.cumsum()
costs_optimistic_cum = costs_optimistic.cumsum()
costs_optimistic_bad_cum = costs_optimistic_bad.cumsum()

T = len(costs_gpc)
plt.figure(figsize=(6, 5))


############### accum. testing ########################
gpc_= (costs_gpc_cum - costs_optimal) / np.arange(1, T+1)
optimistic_= (costs_optimistic_cum - costs_optimal) / np.arange(1, T+1)
optimistic_bad_= (costs_optimistic_bad_cum - costs_optimal) / np.arange(1, T+1)


########################################################

plt.plot(np.arange(T)/1000, gpc_, color='Black', label="GPC", linewidth=2.5, markevery=199, marker='x',  markersize=15, markeredgecolor='Grey')
plt.plot(np.arange(T)/1000, optimistic_, color='Blue', label=r"OptFTRL-C, $\rho=0.9$", linewidth=2.5, markevery=199, marker='*',  markersize=15, markeredgecolor='Grey')
plt.plot(np.arange(T)/1000, optimistic_bad_, color='Red', label=r"OptFTRL-C, $\rho=0.1$", linewidth=2.5, markevery=199, marker='*',  markersize=15, markeredgecolor='Grey')


plt.figtext(0.9, 0, r'$\times 10^3$', ha="right", fontsize=14)

plt.legend(prop={'size': 17}, loc=0)

# for a
# plt.ylabel(r"Average regret $R_T/T$", fontsize=22)
# plt.xlabel(r'Horizon $T$', fontsize=22)
# plt.yticks(np.arange(0, 81, 20), fontsize=17, weight='bold')
# plt.xticks(np.arange(0, 1.1, 0.2), fontsize=17, weight='bold')
# plt.savefig("./aa.pdf", bbox_inches = 'tight',pad_inches = 0)

# for b
# plt.ylabel(r"Average regret $R_T/T$", fontsize=22)
# plt.xlabel(r'Horizon $T$', fontsize=22)
# plt.yticks(np.arange(0,141,20), fontsize=17, weight='bold')
# plt.xticks(np.arange(0,1.1,0.2), fontsize=17, weight='bold')
# plt.savefig("./bb.pdf", bbox_inches = 'tight',pad_inches = 0)

# for c
# plt.ylabel(r"Average regret $R_T/T$", fontsize=22)
# plt.xlabel(r'Horizon $T$', fontsize=22)
# plt.yticks(np.arange(0,18,2), fontsize=17, weight='bold')
# plt.xticks(np.arange(0,1.1,0.2), fontsize=17, weight='bold')
# plt.savefig("./cc.pdf", bbox_inches = 'tight',pad_inches = 0)


print("Accumelated costs are below")
print("GPC: {}".format(costs_gpc_cum[-1]))
print("OFTRL good: {}".format(costs_optimistic_cum[-1]))
print("OFTRL bad: {}".format(costs_optimistic_bad_cum[-1]))
print("Optimal: {}".format(costs_optimal[-1]))


print("Improvements  are below")
# print("OFTRL good over GPC 200: {}".format(np.abs(gpc_[200] - optimistic_[200])/gpc_[200]))
# print("OFTRL good over GPC 400: {}".format(np.abs(gpc_[400] - optimistic_[400])/gpc_[400]))
# print("OFTRL good over GPC 1000: {}".format(np.abs(gpc_[-1] - optimistic_[-1])/gpc_[-1]))
print("OFTRL good over GPC overall: {}".format(np.average(np.abs(gpc_[10:] - optimistic_[10:])/gpc_[10:])))
print("OFTRL bad over GPC overall: {}".format(np.average(np.abs(gpc_[10:] - optimistic_bad_[10:])/gpc_[10:])))

plt.show()