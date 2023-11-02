# -*- coding: utf-8 -*-
"""
Created on Sun Nov  6 11:45:37 2022

@author: 潘先生1
"""

import math
 
def floyd(dis):
    node_num = len(dis)
    for i in range(node_num):        
        for j in range(node_num):    
            for k in range(node_num): 
                dis[j][k] = min(dis[j][i] + dis[i][k],dis[j][k])

if __name__ == '__main__':
    V =int(input('顶点个数:')) #顶点数
    cost = [[float(math.inf) for _ in range(V)] for _ in range(V)]
    # cost[u][v]表示边e=(u,v)的长度，不存在时设为INF，初始化时统一设置为inf
    cost=[]
    n=V#n为矩阵的维数，与顶点个数相等
    print('请输入邻接矩阵')
    for i in range(0,n):
        cost.append(list(map(float,input().split())))#这里采用map函数对输入数据进行类型转换
    floyd(cost)
    print(cost)