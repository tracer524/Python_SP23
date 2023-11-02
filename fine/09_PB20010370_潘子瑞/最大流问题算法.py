# -*- coding: utf-8 -*-
"""
Created on Mon Dec  5 20:16:40 2022

@author: 潘先生1
"""

from queue import Queue
import numpy as np
import math
#初始化
m = int(input('请输入顶点个数:'))#顶点个数
limit=[]
maxflow = [[0 for i in range(m)] for j in range(m)]
#记录最大流图，初始都为0
flow = [0 for i in range(m)]
#记录增广路径前进过程记录的最小流量
pre = [float('inf') for i in range(m)]
#记录增广路径每个节点的前驱
q = Queue()

#输入流量限制图
print('请输入邻接矩阵:')
for i in range(0,m):
    limit.append(list(map(int,input().split())))#这里采用map函数对输入数据进行类型转换

def Find(s,t):#Find找寻增广路径
    q.empty()#清空队列

    for i in range(m):
        pre[i] = float('inf')

    flow[s] = float('inf')

    q.put(s)
    while(not q.empty()):
        index = q.get()
        if(index == t):
            break
        for i in range(m):
            if( (i!=s) & (limit[index][i]>0) & (pre[i]==float('inf')) ):
                pre[i] = index
                flow[i] = min(flow[index],limit[index][i]) 
                q.put(i)
    if(pre[t] == float('inf')):
        #汇点的前驱还是初始值，说明已无增广路径
        return -1
    else:
        return flow[t]##增广路径增加的最小流量

def max_flow(s,t):
    augmentflow = 0#当前寻找到的增广路径的最小通过流量
    sumflow = 0#记录最大流，一直累加
    while(True):
        augmentflow = Find(s,t)
        if(augmentflow == -1):#返回-1说明已没有增广路径
            break
        k = t
        while(k!=s):#k回溯到起点，停止
            prev = pre[k]#走的方向是从prev到k
            maxflow[prev][k] += augmentflow
            limit[prev][k] -= augmentflow#前进方向消耗掉了
            limit[k][prev] += augmentflow#反向边
            k = prev
        sumflow += augmentflow
    return sumflow

result=max_flow(0,m-1)    
print('最大流值为：',result)
print(maxflow) #最大流图