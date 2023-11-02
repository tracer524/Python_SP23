import numpy as np
from scipy import linalg

# 生成矩阵B和向量b
B = np.random.rand(6, 4)
b = np.random.rand(6, 1)

# 计算最小二乘解
B_t = np.transpose(B)
B_inv = np.linalg.inv(np.matmul(B_t, B))
x_ls = np.matmul(np.matmul(B_inv, B_t), b)

# 计算矩阵B的奇异值分解
U, s, Vh = np.linalg.svd(B)

# 计算矩阵B的LU分解
P, L, U = linalg.lu(B)

# 打印结果
print("矩阵B:\n", B)
print("向量b:\n", b)
print("线性方程组 Bx=b 的最小二乘解 x:\n", x_ls)
print("矩阵B的奇异值分解:")
print("U:\n", U)
print("s:\n", s)
print("Vh:\n", Vh)
print("矩阵B的LU分解:")
print("P:\n", P)
print("L:\n", L)
print("U:\n", U)
