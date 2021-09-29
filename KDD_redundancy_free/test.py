# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
 
# name_list = ['Cora','Citeser','Pubmed','PPI']
num_list = [21.22]
num_list1 = [30.05]
x =list(range(len(num_list)))
total_width, n = 0.8, 2
width = total_width / n
 
plt.bar(x, num_list, width=width, label='After-optimization inference time',fc = 'y')
for i in range(len(x)):
    x[i] = x[i] + width
plt.bar(x, num_list1, width=width, label='Befor-optimization inference time',fc = 'r')
# plt.xlabel("Dataset")
plt.ylabel("Time(ms)")
# plt.title("GCN breakdown analysis")
plt.legend()
plt.show()
