# -*- coding: utf-8 -*-
"""
Created on Tue Dec 20 14:44:35 2022

@author: Mit
"""

import time
import math
import PDE
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from PDE import Poisson, Wave
from scipy import interpolate

help(PDE)

#求解泊松方程

start_time = time.time()

def f(x, y) :
    #定义源函数
    return -20*math.pi**2*math.sin(2*math.pi*x)*math.sin(4*math.pi*y)
    
def bf(x, y):
    #定义边界值
    return math.sin(2*math.pi*x)*math.sin(4*math.pi*y)

#代入求解
h = Poisson(0.01, 0.01, 0)
s = h.solve(f, bf, 1, 1)[0]
# 真实值
def s_exact(x, y):
    return math.sin(2*math.pi*x)*math.sin(4*math.pi*y)

h.error_test(s, s_exact, 1, 1)

end_time = time.time()
print(end_time - start_time)

#波方程改良的证明
#定义原始算法

start_time = time.time()

class Wave_orig(Wave):
    def solve(self, c, f, g, bf, ic, iv, max_t, max_x):
        M = round(max_t / self.dt)
        N = round(max_x / self.dx)
        alpha = (self.dt * c / self.dx) ** 2 
        t, x, T, X = self.grid(M, N)
        solution_Mat = self.init_solution(alpha, M, N, bf, ic, iv)
        for i in range(2, M + 1):
            for j in range(1, N):
                solution_Mat[i][j] = (2 - 2 * alpha) * solution_Mat[i - 1][j] + \
                                     alpha * solution_Mat[i - 1][j + 1] + \
                                     alpha * solution_Mat[i - 1][j - 1] - \
                                     solution_Mat[i - 2][j]
        if self.fig:
            self.figure_3D(T, X, solution_Mat.T, 'Wave Equation')  
        s = interpolate.RectBivariateSpline(t, x, solution_Mat, kx = 3, ky = 3)
        return s, solution_Mat
    
def ic(x):
    return math.sin(x/2)

def iv(x):
    return math.cos(x/2)/2
    
def bf(t,x):
    return math.sin(x/2+t)

h = [0, 0, 0]
M = [0, 0, 0]

h[0] = Wave_orig(0.01, 0.01, 0)
M[0] = (h[0].solve(2, lambda x: 0, lambda t,x: 0 , bf, ic, iv, 1,1))[1]


h[1] = Wave(0.01, 0.01, 0)
M[1] = (h[1].solve(2, lambda x: 0, lambda t,x: 0 , bf, ic, iv, 1,1))[1]
        
t = np.linspace(0, 1, 101)
x = np.linspace(0, 1, 101)
T, X = np.meshgrid(t, x)    

M[2] = np.zeros([101,101])
for i in range(101):
    for j in range(101):
        M[2][i][j] = math.sin(x[j]/2+t[i])
        

def title(ax, title):
    label = ('t','x','u')
    ax.set_title(title, fontsize=20)
    ax.set_xlabel(label[0], fontsize=16)
    ax.set_ylabel(label[1], fontsize=16)
    ax.set_zlabel(label[2], fontsize=16)

#画图

fig, axes = plt.subplots(1,3, figsize=(15, 15), subplot_kw={'projection': '3d'})
subtitle = ['Original Algorithm', 'Imporved Algorithm', 'Exact Solution']

            
for k in range(3):
    norm = mpl.colors.Normalize(vmin = M[k].min(), vmax = M[k].max())
    p = axes[k].plot_surface(T, X, M[k].T, linewidth=0, rcount=1e4, 
                             cmap = mpl.cm.hot, ccount=20, norm = norm)
    fig.colorbar(p, ax = axes[k], pad=0.1, shrink=0.3)
    title(axes[k], subtitle[k])
    
maintitle = '\n\n\n\n\n\n\n\n Visualization of the improvemt'
fig.suptitle(maintitle, fontsize=25)
fig.tight_layout()

end_time = time.time()
print(end_time - start_time)
    





