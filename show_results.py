import pickle
import matplotlib.pyplot as plt
from pylab import *
from matplotlib.ticker import FuncFormatter
from matplotlib.transforms import Bbox
import sys
from matplotlib.lines import fillStyles
from matplotlib.markers import MarkerStyle
from matplotlib.backends.backend_pdf import PdfPages

x_axis = pickle.load(open("./pickle/x_axis", "rb"))

# Markov_slot_1 = pickle.load(open("./pickle/Markov_slot_1", "rb"))
# Markov_slot_2 = pickle.load(open("./pickle/Markov_slot_2", "rb"))
Markov_slot_3 = pickle.load(open("./pickle/Markov_slot_3", "rb"))

# outage_slot_1 = pickle.load(open("./pickle/outage_slot_1", "rb"))
# outage_slot_2 = pickle.load(open("./pickle/outage_slot_2", "rb"))
outage_slot_3 = pickle.load(open("./pickle/outage_slot_3", "rb"))

# energy_slot_1 = pickle.load(open("./pickle/energy_slot_1", "rb"))
# energy_slot_2 = pickle.load(open("./pickle/energy_slot_2", "rb"))
energy_slot_3 = pickle.load(open("./pickle/energy_slot_3", "rb"))

plt.figure(11, figsize=(10, 4))
# plt.plot(x_axis, Markov_slot_1, color='blue', alpha=0.8, label='Case I',
#          linestyle='dotted', linewidth=4)
# plt.plot(x_axis, Markov_slot_2, color='green', alpha=0.9, label='Case II',
#          linestyle='--', linewidth=4)
plt.plot(x_axis, Markov_slot_3, color='red', alpha=0.7, label='DRLCI',
         linestyle='-', linewidth=4)
# plt.grid(linestyle='-.')     # 添加网格
# plt.tick_params(labelsize=15)
xlabel('Episode', fontsize=17)
ylabel(r'Average Reward', fontsize=17)
# legend(loc='best', fancybox=True, fontsize=10, ncol=2)
legend(fontsize=17)
plt.xticks(fontsize=17)
plt.yticks(fontsize=17)
pp = PdfPages('./resultfigs/shiyan11.pdf')
plt.savefig(pp, format='pdf')
pp.close()

plt.figure(12, figsize=(10, 4))
plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda outage_slot_1, _: '{:.0%}'.format(outage_slot_1)))
# plt.plot(x_axis, outage_slot_1, color='blue', alpha=0.8, label='Case I',
#          linestyle='dotted', linewidth=4)
# plt.plot(x_axis, outage_slot_2, color='green', alpha=0.9, label='Case II',
#          linestyle='--', linewidth=4)
plt.plot(x_axis, outage_slot_3, color='red', alpha=0.7, label='DRLCI',
         linestyle='-', linewidth=4)
# plt.grid(linestyle='-.')     # 添加网格
# plt.tick_params(labelsize=15)
xlabel('Episode', fontsize=17)
ylabel(r'Average Task Drop Rate', fontsize=17)
# legend(loc='best', fancybox=True, fontsize=10, ncol=2)
legend(fontsize=17)
plt.xticks(fontsize=17)
plt.yticks(fontsize=17)
pp = PdfPages('./resultfigs/shiyan12.pdf')
plt.savefig(pp, format='pdf')
pp.close()



plt.figure(13, figsize=(10, 4))
# plt.plot(x_axis, energy_slot_1, color='blue', alpha=0.8, label='Case I',
#          linestyle='dotted', linewidth=4)
# plt.plot(x_axis, energy_slot_2, color='green', alpha=0.9, label='Case II',
#          linestyle='--', linewidth=4)
plt.plot(x_axis, energy_slot_3, color='red', alpha=0.7, label='DRLCI',
         linestyle='-', linewidth=4)
# plt.grid(linestyle='-.')     # 添加网格
# plt.tick_params(labelsize=17)
xlabel('Episode', fontsize=17)
ylabel(r'Average Energy Queue (J)', fontsize=17)
# legend(loc='best', fancybox=True, fontsize=10, ncol=2)
legend(fontsize=17)
plt.xticks(fontsize=17)
plt.yticks(fontsize=17)
pp = PdfPages('./resultfigs/shiyan13.pdf')
plt.savefig(pp, format='pdf')
pp.close()

plt.show()

