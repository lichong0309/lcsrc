# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
 
name_list = ['Cora','Citeser','Pubmed','PPI']
num_list = [30.7705, 73.1086, 90.2736, 26.603]
num_list1 = [37.753, 97.269, 109.840, 27.76575]
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
