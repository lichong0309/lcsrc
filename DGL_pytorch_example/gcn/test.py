# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
 
name_list = ['Cora','Citeser','Pubmed','PPI']
num_list = [2.598, 1.840, 5.412817, 6.7496]
num_list1 = [38.239, 41.9316, 95.9949, 63.475]
x =list(range(len(num_list)))
total_width, n = 0.8, 2
width = total_width / n
 
plt.bar(x, num_list, width=width, label='Aggregation time',fc = 'y')
for i in range(len(x)):
    x[i] = x[i] + width
plt.bar(x, num_list1, width=width, label='Training time',tick_label = name_list,fc = 'r')
plt.xlabel("Dataset")
plt.ylabel("Time(ms)")
plt.title("GCN breakdown analysis")
plt.legend()
plt.show()
