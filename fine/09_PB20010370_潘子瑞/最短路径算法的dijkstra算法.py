# -*- coding: utf-8 -*-
"""
Created on Sat Dec  3 16:52:28 2022

@author: 潘先生1
"""
import math
def dijkstra(s):
    distance[s] = 0
    while True:
        v = -1# v在这里相当于是一个指标，对包含起点s做统一处理
        for u in range(V):# 从未使用过的顶点中选择一个距离最小的顶点
            if not used[u] and (v == -1 or distance[u] < distance[v]):
                v = u
        if v == -1: # 说明所有顶点都添加到S中了！
            break
        # 将选定的顶点加入到S中, 同时进行距离更新
        used[v] = True
        # 更新U中各个顶点到起点s的距离。
        for u in range(V):
            distance[u] = min(distance[u], distance[v] + cost[v][u])

if __name__ == '__main__':
    V =int(input('顶点个数:')) #顶点数
    used = [False for _ in range(V)] #标记数组：used[v]值为False说明改顶点还没有访问过，在S中，否则在U中
    distance = [float(math.inf) for _ in range(V)]
    # distance[i]表示从源点s到ｉ的最短距离，distance[s]=0
    #cost = [[float(math.inf) for _ in range(V)] for _ in range(V)]
    # cost[u][v]表示边e=(u,v)的长度，不存在时设为INF，初始化时统一设置为inf
    cost=[]
    n=V#n为矩阵的维数，与顶点个数相等
    print('请输入邻接矩阵')
    for i in range(0,n):
        cost.append(list(map(float,input().split())))#这里采用map函数对输入数据进行类型转换
    s = int(input('请输入一个起始点：'))#指定一个起始点
    dijkstra(s)
    print(distance)