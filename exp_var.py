# -*- coding: utf-8 -*-
"""
Created on Thu Apr 17 17:23:01 2025

@author: PC
"""

import pickle
import matplotlib.pyplot as plt
import numpy as np
import os

plt.rcParams['font.family'] = 'Arial'

if __name__ == '__main__':
    file_path = './source_files/CPCA_HCP/cpca_cere_10pcs_HCP_complex_results.pkl'
    with open(file_path, 'rb') as f:
        data_HCP = pickle.load(f)['pca']
    
    file_path = './source_files/CPCA_HCD/cpca_cere_10pcs_HCD_complex_results.pkl'
    with open(file_path, 'rb') as f:
        data_HCD = pickle.load(f)['pca']
    
    data = [data_HCP['exp_var'], data_HCD['exp_var']]
    labels = ['HCP', 'HCD']
    data_rate = [[j/sum(i) for j in i] for i in data]
    colors = ['orange']
    styles = ['o']
    
    for ele in range(len(data)):
        fig = plt.figure(figsize=(2, 2), dpi=200)
        main_ax = fig.add_subplot(111)
        lines = []
        
        line, = main_ax.plot(np.arange(1, len(data[ele])+1), data_rate[ele], 
                             color='black', zorder=1)
        scatter = main_ax.scatter(np.arange(1, len(data[ele])+1), data_rate[ele], 
                                  color='orange', marker='o', zorder=2)
        lines.append((line, scatter))
    
        main_ax.set_xlabel('Index of PC', fontsize=8)
        main_ax.set_ylabel('Explained variance rate', fontsize=8)
        plt.xticks(np.arange(11), fontsize=8)
        plt.yticks(fontsize=8)
        plt.savefig(f"./output/{labels[ele]}_exp_vars", bbox_inches='tight', dpi=300)
        plt.show()