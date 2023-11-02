# -*- coding: utf-8 -*-
"""

@author: Mit

本程序用于求解常见偏微分方程：一维热方程，一维波方程，二维泊松方程。

在使用时，用户输入所需参量，程序将据此进行数值求解，最后返还数值解函数，并且作出函数图像。

以下是不同方程所需参数列表：

设 inr_t,inr_x,inr_y分别表示t轴，x轴，y轴的微小增量，fig表示是否需要绘制可视化图，取0或1
1. 热方程：
   （1）初始化： h = Heat(inr_t, inr_x, fig = 1)
   （2）求解： f = h.solve(c, f, g, bf, ic, max_t, max_x)，c表示方程中的常系数，f是
       方程中 u 的函数，g是 t,x 的函数， bf 是边界条件， ic 是初值， max_t和 max_x 是
       t 和 x 的最大值（需大于0）
       
2. 波方程：
   （1）初始化： h = Wave(inr_t, inr_x, fig = 1)
   （2）求解： solve(c, f, g, bf, ic, iv, max_t, max_x)，c表示方程中的正常系数的平方
       根，f是方程中 u 的函数，g是 t,x 的函数， bf 是边界条件， ic 是初值， iv 是初始速
       度，max_t和 max_x 是 t 和 x 的最大值（需大于0）
      
3. 泊松方程：
   （1）初始化： h = Poisson(inr_x, inr_y, fig = 1)
   （2）求解： solve(f, bf, max_t, max_x)，f是方程的源函数， bf 是边界条件，max_t和 
        max_x 是 t 和 x 的最大值（需大于0）

请注意： 如果边值和初值的定义域有重叠，则在重叠处两种应相等

误差分析功能：error_test(f_num, f_exact, max_t, max_x)，f_num是训练的数值解，f_exact
     是真实值函数，max_t和 max_x 是区域的上界（需大于0）

"""


import math
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy import interpolate
from scipy import sparse
from scipy.sparse.linalg import spsolve
import time



class InputError(Exception):
    def __init__(self, message):
        self.message = message



class PDE():

    def __init__(self, inr_t=0.01, inr_x=0.01, fig = 1):
        err_msg = "fig参数只能取0或1，但输入的是: {}".format(fig)
        assert fig in [1, 0], err_msg
        if inr_t <= 0 or inr_x <= 0:
            raise InputError("您输入的步长为非正值，请重新输入！")
        self.dt = inr_t
        self.dx = inr_x
        self.type = None
        self.fig = fig  # 标识是否需要绘制大型可视化图

    def grid(self, M, N):
        t = np.linspace(0, self.dt*M, M+1)
        x = np.linspace(0, self.dx*N, N+1)
        T, X = np.meshgrid(t, x)
        return t, x, T, X
    
    def title(self, ax, title):
        if self.type == 'Heat':
            label = ('t','x','u')
        elif self.type == 'Wave':
            label = ('t','x','u')
        else:
            label = ('x','y','u')
        ax.set_title(title, fontsize=20)
        ax.set_xlabel(label[0], fontsize=16)
        ax.set_ylabel(label[1], fontsize=16)
        ax.set_zlabel(label[2], fontsize=16)
    
    def figure_3D(self, T, X, Z, title):
        fig = plt.figure(figsize=(16,16))
        ax = fig.gca(projection='3d')
        self.title(ax, title)
        norm = mpl.colors.Normalize(vmin = Z.min(), vmax = Z.max())
        p = ax.plot_surface(T, X, Z, linewidth=0, rcount=1e4, cmap = mpl.cm.hot,
                            ccount=20, norm = norm)
        fig.colorbar(p, ax = ax, pad = 0.1)
        
    def error_test(self, f_num, f_exact, max_t, max_x):
        if max_t <= 0 or max_x <= 0:
            raise InputError("您在进行误差分析时输入的区间上界为非正值，请重新输入！")
        M = round(max_t/self.dt)
        N = round(max_x/self.dx)
        t, x, T, X = self.grid(M, N)
        Z_num = np.zeros([M+1, N+1])
        Z_exact = np.zeros([M+1, N+1])
        for i in range(M+1):
            for j in range(N+1):
                Z_num[i][j] = f_num(t[i],x[j])
                Z_exact[i][j] = f_exact(t[i], x[j])
        fig, axes = plt.subplots(1,3, figsize=(15, 15), subplot_kw={'projection': '3d'})
        norm = mpl.colors.Normalize(vmin = Z_num.min(), vmax = Z_num.max())
        p = axes[0].plot_surface(T, X, Z_num.T, linewidth=0, rcount=1e4, 
                                 cmap = mpl.cm.hot, ccount=20, norm = norm)
        fig.colorbar(p, ax = axes[0],pad=0.1, shrink=0.3)
        self.title(axes[0], 'numerical soluton')
        norm = mpl.colors.Normalize(vmin = Z_exact.min(), vmax = Z_exact.max())
        p = axes[1].plot_surface(T, X, Z_exact.T, linewidth=0, rcount=1e4,
                                 cmap = mpl.cm.hot, ccount=20, norm = norm)
        fig.colorbar(p, ax = axes[1],pad=0.1, shrink=0.3)
        self.title(axes[1], 'exact solution')
        error = (Z_exact - Z_num)
        norm = mpl.colors.Normalize(vmin = error.min(), vmax = error.max())
        p = axes[2].plot_surface(T, X, error.T, linewidth=0, rcount=1e4, 
                                 cmap = mpl.cm.hot, ccount=20, norm = norm)
        fig.colorbar(p, ax = axes[2],pad=0.1, shrink=0.3)
        self.title(axes[2], 'error')
        title = '\n\n\n\n\n\n\n\n' + self.type + ' Equatoin Error Visualization'
        fig.suptitle(title, fontsize=25)
        fig.tight_layout()
        
        
        
class Heat(PDE):
    
    def init_solution(self, M, N, bf, ic):
        if abs(bf(0, 0) - ic(0)) > 1e-5:
               raise InputError("请注意：初值和边值在其定义域的交集内必须相等，但是(0,0)处初边值不相等！")
        if abs(bf(0, N*self.dx) - ic(N*self.dx)) > 1e-5:
               raise InputError("请注意：初值和边值在其定义域的交集内必须相等，但是(0,%.1f)处初边值不相等！"\
                                %(N*self.dx))
        solution = np.zeros([M+1, N+1])
        for i in range(M+1):
            solution[i][0] = bf(i*self.dt, 0)
            solution[i][-1] = bf(i*self.dt, N*self.dx)
        for j in range(N+1):
            solution[0][j] = ic(j*self.dx)
        return solution
    
    def solve(self, c, f, g, bf, ic, max_t, max_x):
        if max_t <= 0 or max_x <= 0:
            raise InputError("您在求解方程时输入的区间上界为非正值，请重新输入！")
        self.type = 'Heat'
        M = round(max_t/self.dt)
        N = round(max_x/self.dx)
        alpha = (self.dt*c) / (self.dx**2)
        t, x, T, X = self.grid(M, N)
        solution_Mat = self.init_solution(M, N, bf, ic)
        coef_Mat_1 = np.diag([1+alpha] * (N-1)) \
                     + np.diag([-alpha/2] * (N-2), 1) \
                     + np.diag([-alpha/2] * (N-2), -1)
        coef_Mat_2 = np.diag([1-alpha] * (N-1)) \
                     + np.diag([alpha/2] * (N-2), 1)  \
                     + np.diag([alpha/2] * (N-2), -1)
        coef_Mat_3 = np.linalg.inv(coef_Mat_1)
        for i in range(1, M+1):
            const = solution_Mat[i-1, 1:-1].dot(coef_Mat_2) \
                           + self.dt * (np.array(list(map(f, solution_Mat[i-1, 1:-1]))) \
                           + np.array([g(i*self.dt, j) for j in x[1:-1]]))
            const[0] += alpha/2 * (solution_Mat[i-1][0] + solution_Mat[i][0])
            const[-1] += alpha/2 * (solution_Mat[i-1][-1] + solution_Mat[i][-1])
            solution_Mat[i, 1:-1] = const.dot(coef_Mat_3)
        if self.fig:    
            self.figure_3D(T, X, solution_Mat.T, 'Heat Equation') 
        s = interpolate.RectBivariateSpline(t, x, solution_Mat, kx=3, ky=3)
        return s,solution_Mat
    

    
class Wave(PDE):
    
    def init_solution(self, alpha, M, N, bf, ic, iv):
        if abs(bf(0, 0) - ic(0)) > 1e-5:
            raise InputError("请注意：初值和边值在其定义域的交集内必须相等，但是(0,0)处初边值不相等！")
        if abs(bf(0, N * self.dx) - ic(N * self.dx)) > 1e-5:
            raise InputError("请注意：初值和边值在其定义域的交集内必须相等，但是(0,%.1f)处初边值不相等！"\
                             %(N*self.dx))
        solution = np.zeros([M+1, N+1])
        for i in range(M+1):
            solution[i][0] = bf(i*self.dt, 0)
            solution[i][-1] = bf(i*self.dt, N*self.dx)
        for j in range(N+1):
            solution[0][j] = ic(j*self.dx)
            solution[1][j] = (1-alpha) * ic(j*self.dx) \
                             + alpha/2 * (ic((j-1) * self.dx) + ic((j+1)*self.dx)) \
                             + self.dt * iv(j*self.dx)
        return solution
    
    def solve(self, c, f, g, bf, ic, iv, max_t, max_x):
        if max_t <= 0 or max_x <= 0:
            raise InputError("您在求解方程时输入的区间上界为非正值，请重新输入！")
        self.type = 'Wave'
        M = round(max_t/self.dt)
        N = round(max_x/self.dx)
        alpha = (self.dt*c/self.dx)**2 
        t, x, T, X = self.grid(M, N)
        solution_Mat = self.init_solution(alpha, M, N, bf, ic, iv)
        coef_Mat_1 = np.diag([1+alpha] * (N-1)) \
                     + np.diag([-alpha/2] * (N-2), 1) \
                     + np.diag([-alpha/2] * (N-2), -1)
        coef_Mat_2 = np.diag([2-alpha] * (N-1)) \
                     + np.diag([alpha/2] * (N-2), 1) \
                     + np.diag([alpha/2] * (N-2), -1)
        coef_Mat_3 = np.linalg.inv(coef_Mat_1)
        for i in range(2, M+1):
            const = solution_Mat[i-1, 1:-1].dot(coef_Mat_2) \
                           - solution_Mat[i-2, 1:-1] + (self.dt ** 2) \
                           * (np.array(list(map(f, solution_Mat[i-1, 1:-1]))) \
                           + np.array([g(i*self.dt, j) for j in x[1 : -1]])) 
            const[0] += alpha/2 * (solution_Mat[i-1][0] + solution_Mat[i][0])
            const[-1] += alpha/2 * (solution_Mat[i-1][-1] + solution_Mat[i][-1])
            solution_Mat[i, 1:-1] = const.dot(coef_Mat_3)
        if self.fig:    
            self.figure_3D(T, X, solution_Mat.T, 'Wave Equation')  
        s = interpolate.RectBivariateSpline(t, x, solution_Mat, kx=3, ky=3)
        return s,solution_Mat
    
    
    
class Poisson(PDE):
    
    def init_solution(self, M, N, bf): 
        solution = np.zeros([M+1, N+1])
        for i in range(M+1):
            solution[i][0] = bf(i*self.dt, 0)
            solution[i][-1] = bf(i*self.dt, N*self.dx)
        for j in range(N+1):
            solution[0][j] = bf(0, j*self.dx)
            solution[-1][j] = bf(M*self.dt, j*self.dx)
        return solution
    
    def solve(self, f, bf, max_t, max_x):
        if max_t <= 0 or max_x <= 0:
            raise InputError("您在求解方程时输入的区间上界为非正值，请重新输入！")
        self.type = 'Poisson'
        M = round(max_t/self.dt)
        N = round(max_x/self.dx)
        t, x, T, X = self.grid(M, N)
        I_0M = np.ones(M-1)
        I_1M = np.ones(M-2)
        I_1N = np.ones(N-2)
        I_sparse = sparse.eye(N-1, format = 'csc')
        J_sparse = sparse.diags([I_1N, I_1N], [-1, 1], format = 'csc')
        solution_Mat = self.init_solution(M, N, bf)
        coef_Mat_1 = sparse.diags([-1/self.dt**2 * I_1M, 
                                   2 * (1/self.dt**2 + 1/self.dx**2) 
                                   * I_0M, -1/self.dt**2 * I_1M], [-1, 0, 1],
                                  format = 'csc')
        coef_Mat_2 = sparse.diags(-1/self.dx**2 * I_0M, format = 'csc')
        coef_Mat_3 = sparse.kron(I_sparse, coef_Mat_1, format = 'csc') + \
                     sparse.kron(J_sparse, coef_Mat_2, format = 'csc')
        coef_Mat_4 = np.diag([-1/ self.dt**2] * (M-1)) 
        const = np.zeros([M-1, N-1])
        for i in range(0, M-1):
            for j in range(N-1):
                const[i][j] =  -f(t[i+1], x[j+1])        
        const[0,:] += solution_Mat[0, 1:-1]/self.dt**2
        const[-1, :] += solution_Mat[-1, 1:-1] / self.dt**2
        const[:, 0] -= coef_Mat_4.dot(solution_Mat[1:-1, 0])
        const[:, -1] -= coef_Mat_4.dot(solution_Mat[1:-1, -1])
        const = const.T.flatten()
        solution_Mat[1:-1, 1:-1] = spsolve(coef_Mat_3, const).reshape(N-1, M-1).T
        if self.fig:
            self.figure_3D(T, X, solution_Mat.T, 'Poisson Equation')  
        s = interpolate.RectBivariateSpline(t, x, solution_Mat, kx=3, ky=3)
        return s,solution_Mat



def test_heat():
    t = np.linspace(0, 1, 1000)
    x = np.linspace(0, 1, 1000)
    T, X = np.meshgrid(t, x) 
    def ic(x):
        return math.cosh(x)
    def bf(t, x):
        return math.exp(2*t) * math.cosh(x)
    def f_exact(t, x):
        return  math.exp(2*t)*math.cosh(x)
    h = Heat(0.01, 0.01, 0)
    s, M = h.solve(1, lambda x: x, lambda x,y: 0, bf, ic, 1,1)
    h.error_test(s, f_exact, 1, 1)
    
    
def test_wave():
    t = np.linspace(0, 1, 1000)
    x = np.linspace(0, 1, 1000)
    T, X = np.meshgrid(t, x) 
    def ic(x):
        return math.sin(x)
    def iv(x):
        return math.cos(x)
    def bf(t,x):
        return math.exp(t)*math.sin(x)
    def f_exact(t, x):
        return  math.exp(t)*math.sin(x)
    h = Wave(0.01, 0.01, 0)
    s, M = h.solve(1, lambda u: u, lambda t,x: x , bf, ic, iv, 1,1)
    h.error_test(s, f_exact, 1, 1)

    
def test_poisson():
    def f(x,y):
        return -2*math.pi**2*math.sin(math.pi*x)*math.sin(math.pi*y)
    def bf(x,y):
        return 0
    def f_exact(x,y):
        return math.sin(math.pi*x)*math.sin(math.pi*y)
    h = Poisson(0.01, 0.01, 0)
    s, M = h.solve(f, bf, 1,1)
    h.error_test(s , f_exact, 1, 1)
 
    
def test_all():
    test_wave()
    test_heat()
    test_poisson()
    
if __name__ == '__main__':
    start_time = time.time()
    test_all()
    end_time = time.time()
    print(end_time - start_time)
 






        


   
        
